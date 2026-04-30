import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import FineTuneExperiment


def parse_args():
    parser = argparse.ArgumentParser(description="Export patient-level fine-tune outputs from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained fine-tune .ckpt file")
    parser.add_argument("--path-to-db", required=True, help="Path to the SQLite database used by FastEHR")
    parser.add_argument("--path-to-ds", required=True, help="Path to the built supervised FastEHR dataset directory")
    parser.add_argument("--meta-information-path", default=None, help="Optional path to meta_information.pickle")
    parser.add_argument("--output-dir", default="outputs/patient_outputs", help="Directory to write output files")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def _predict_batch(experiment, batch, outcome_tokens):
    model_batch = {k: (v.to(next(experiment.parameters()).device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        outputs, _, _ = experiment(
            model_batch,
            is_generation=True,
            return_loss=False,
            return_generation=True,
        )

    surv = outputs["surv"]
    surv_cdf = surv["surv_CDF"]
    batch_size = int(model_batch["tokens"].shape[0])

    rows = []
    for sample_idx in range(batch_size):
        risk_scores = np.asarray([float(cdf[sample_idx, -1]) for cdf in surv_cdf], dtype=float)
        top_idx = int(np.argmax(risk_scores)) if len(risk_scores) > 0 else None
        rows.append({
            "pred_top1_risk_index": int(top_idx + 1) if top_idx is not None else None,
            "pred_top1_outcome_token": int(outcome_tokens[top_idx]) if top_idx is not None else None,
            "pred_top1_score": float(risk_scores[top_idx]) if top_idx is not None else None,
            "pred_all_scores": json.dumps([float(v) for v in risk_scores.tolist()]),
            "pred_outcome_tokens": json.dumps([int(v) for v in outcome_tokens]),
            "target_token": int(model_batch["target_token"][sample_idx].detach().cpu().item()),
            "target_age_delta": float(model_batch["target_age_delta"][sample_idx].detach().cpu().item()),
            "target_value": float(model_batch["target_value"][sample_idx].detach().cpu().item()),
        })
    return rows


def export_split(experiment, dm, split_name, output_dir, outcome_tokens):
    split_to_loader = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }
    loader = split_to_loader[split_name]

    rows = []
    sample_counter = 0
    for batch_idx, batch in enumerate(loader):
        batch_preds = _predict_batch(experiment, batch, outcome_tokens=outcome_tokens)
        for in_batch_idx, pred in enumerate(batch_preds):
            rows.append({
                "split": split_name,
                "sample_index": int(sample_counter),
                "batch_index": int(batch_idx),
                "in_batch_index": int(in_batch_idx),
                **pred,
            })
            sample_counter += 1

    split_df = pd.DataFrame(rows)
    split_csv = output_dir / f"finetune_patient_predictions_{split_name}.csv"
    split_parquet = output_dir / f"finetune_patient_predictions_{split_name}.parquet"
    split_df.to_csv(split_csv, index=False)
    split_df.to_parquet(split_parquet, index=False)

    return split_df


def export_model_artifacts(experiment, checkpoint_path, output_dir):
    artifacts_dir = output_dir / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_copy = artifacts_dir / Path(checkpoint_path).name
    shutil.copy2(checkpoint_path, ckpt_copy)

    state_dict_path = artifacts_dir / "finetune_experiment_state_dict.pt"
    torch.save(experiment.state_dict(), state_dict_path)

    reload_instructions = artifacts_dir / "reload_instructions.txt"
    reload_instructions.write_text(
        "from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import FineTuneExperiment\n"
        f"model = FineTuneExperiment.load_from_checkpoint('{ckpt_copy}', weights_only=False, map_location='cpu')\n"
        "model.eval()\n"
    )


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    experiment = FineTuneExperiment.load_from_checkpoint(
        str(checkpoint_path),
        weights_only=False,
        map_location=device,
    )
    experiment = experiment.to(device)
    experiment.eval()

    outcome_tokens = [int(v) for v in experiment.hparams.outcome_tokens]

    dm = FoundationalDataModule(
        path_to_db=args.path_to_db,
        path_to_ds=args.path_to_ds,
        load=True,
        tokenizer="tabular",
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        min_workers=0,
        overwrite_meta_information=args.meta_information_path,
        supervised=True,
        outcome_list=outcome_tokens,
        global_diagnoses=False,
        repeating_events=True,
    )

    split_frames = []
    for split in args.splits:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        split_frames.append(export_split(experiment, dm, split, output_dir, outcome_tokens=outcome_tokens))

    all_df = pd.concat(split_frames, ignore_index=True)
    all_df.to_csv(output_dir / "finetune_patient_predictions_all.csv", index=False)
    all_df.to_parquet(output_dir / "finetune_patient_predictions_all.parquet", index=False)

    export_model_artifacts(experiment, str(checkpoint_path), output_dir)

    summary = {
        "checkpoint": str(checkpoint_path),
        "rows_exported": int(len(all_df)),
        "splits": sorted(set(all_df["split"].tolist())),
        "outcome_tokens": outcome_tokens,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
