import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_causal_experiment import CausalExperiment


def parse_args():
    parser = argparse.ArgumentParser(description="Export patient-level SurvivEHR outputs from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .ckpt file")
    parser.add_argument("--path-to-db", required=True, help="Path to the SQLite database used by FastEHR")
    parser.add_argument("--path-to-ds", required=True, help="Path to the built FastEHR dataset directory")
    parser.add_argument("--meta-information-path", default=None, help="Optional path to meta_information.pickle")
    parser.add_argument("--output-dir", default="outputs/patient_outputs", help="Directory to write output files")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def _index_to_file_row(file_row_count_dict, index):
    remaining = int(index)
    for file_path, row_count in file_row_count_dict.items():
        row_count = int(row_count)
        if remaining >= row_count:
            remaining -= row_count
        else:
            return file_path, remaining
    raise IndexError(f"Index {index} out of range for file_row_count_dict")


def _get_patient_metadata(file_path, row_index):
    row = pq.read_table(file_path).to_pandas().loc[row_index]
    return {
        "patient_id": int(row["PATIENT_ID"]),
        "practice_id": int(row["PRACTICE_ID"]),
        "country": str(row["COUNTRY"]),
        "health_auth": str(row["HEALTH_AUTH"]),
    }


def _single_item_batch(sample_dict, device):
    tokens = sample_dict["tokens"]
    attention_mask = torch.ones_like(tokens)
    batch = {
        "static_covariates": sample_dict["static_covariates"].unsqueeze(0).to(device),
        "tokens": tokens.unsqueeze(0).to(device),
        "ages": sample_dict["ages"].unsqueeze(0).to(device),
        "values": sample_dict["values"].unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
    }
    return batch


def _predict_patient(experiment, dm, sample_dict, top_k):
    with torch.no_grad():
        outputs, _, _ = experiment(
            sample_dict,
            is_generation=True,
            return_loss=False,
            return_generation=True,
        )

    surv = outputs["surv"]
    surv_cdf = surv["surv_CDF"]

    event_scores = np.asarray([float(cdf[0, -1]) for cdf in surv_cdf], dtype=float)
    top_indices = np.argsort(event_scores)[::-1][:top_k]

    top_event_ids = [int(idx + 1) for idx in top_indices]
    top_event_tokens = [dm.decode([event_id]) for event_id in top_event_ids]
    top_event_scores = [float(event_scores[idx]) for idx in top_indices]

    return {
        "pred_top1_event_id": top_event_ids[0] if top_event_ids else None,
        "pred_top1_event_token": top_event_tokens[0] if top_event_tokens else None,
        "pred_top1_score": top_event_scores[0] if top_event_scores else None,
        "pred_topk_event_ids": json.dumps(top_event_ids),
        "pred_topk_event_tokens": json.dumps(top_event_tokens),
        "pred_topk_scores": json.dumps(top_event_scores),
    }


def export_split(experiment, dm, split_name, output_dir, top_k):
    split_to_dataset = {
        "train": dm.train_set,
        "val": dm.val_set,
        "test": dm.test_set,
    }
    dataset = split_to_dataset[split_name]

    rows = []
    for idx in range(len(dataset)):
        file_path, row_index = _index_to_file_row(dataset.file_row_count_dict, idx)
        metadata = _get_patient_metadata(file_path, row_index)

        sample = dataset.getitem(idx)
        batch = _single_item_batch(sample, next(experiment.parameters()).device)
        pred = _predict_patient(experiment, dm, batch, top_k=top_k)

        rows.append({
            "split": split_name,
            "sample_index": int(idx),
            **metadata,
            **pred,
        })

    split_df = pd.DataFrame(rows)
    split_csv = output_dir / f"patient_predictions_{split_name}.csv"
    split_parquet = output_dir / f"patient_predictions_{split_name}.parquet"
    split_df.to_csv(split_csv, index=False)
    split_df.to_parquet(split_parquet, index=False)

    return split_df


def export_model_artifacts(experiment, checkpoint_path, output_dir):
    artifacts_dir = output_dir / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_copy = artifacts_dir / Path(checkpoint_path).name
    shutil.copy2(checkpoint_path, ckpt_copy)

    state_dict_path = artifacts_dir / "causal_experiment_state_dict.pt"
    torch.save(experiment.state_dict(), state_dict_path)

    reload_instructions = artifacts_dir / "reload_instructions.txt"
    reload_instructions.write_text(
        "from SurvivEHR.examples.modelling.SurvivEHR.setup_causal_experiment import CausalExperiment\n"
        f"model = CausalExperiment.load_from_checkpoint('{ckpt_copy}', weights_only=False, map_location='cpu')\n"
        "model.eval()\n"
    )


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    dm = FoundationalDataModule(
        path_to_db=args.path_to_db,
        path_to_ds=args.path_to_ds,
        load=True,
        tokenizer="tabular",
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        min_workers=0,
        overwrite_meta_information=args.meta_information_path,
        supervised=False,
        global_diagnoses=False,
        repeating_events=True,
    )

    experiment = CausalExperiment.load_from_checkpoint(
        str(checkpoint_path),
        weights_only=False,
        map_location=device,
    )
    experiment = experiment.to(device)
    experiment.eval()

    split_frames = []
    for split in args.splits:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        split_frames.append(export_split(experiment, dm, split, output_dir, args.top_k))

    all_df = pd.concat(split_frames, ignore_index=True)
    all_df.to_csv(output_dir / "patient_predictions_all.csv", index=False)
    all_df.to_parquet(output_dir / "patient_predictions_all.parquet", index=False)

    export_model_artifacts(experiment, str(checkpoint_path), output_dir)

    summary = {
        "checkpoint": str(checkpoint_path),
        "rows_exported": int(len(all_df)),
        "splits": sorted(set(all_df["split"].tolist())),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
