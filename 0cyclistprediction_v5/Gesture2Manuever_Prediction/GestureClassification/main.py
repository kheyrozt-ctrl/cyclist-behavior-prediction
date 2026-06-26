from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GestureClassification.newsplit_dataloader import create_and_save_dataloaders
from GestureClassification.newsplit_head_model_definition import HeadBiLSTMClassifier
from GestureClassification.newsplit_leg_model_definition import LegBiLSTMClassifier
from GestureClassification.newsplit_upper_model_definition import UpperBiLSTMClassifier
from GestureClassification.newsplit_model_training import train_all_folds


def main(argv=None):
    # Parse command-line arguments.
    release_root = Path(__file__).resolve().parents[2]
    run_root = release_root.parent
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data-dir", type=Path, default=run_root / "release_data" / "GestureDataset", help="directory created by GestureCrop_newsplit.py")
    parser.add_argument("--output-dir", type=Path, default=run_root / "release_runs")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--branch", choices=["head", "upper", "leg", "all"], default="all")
    args = parser.parse_args(argv)

    branches = ("head", "upper", "leg") if args.branch == "all" else (args.branch,)
    model_classes = {
        "head": HeadBiLSTMClassifier,
        "upper": UpperBiLSTMClassifier,
        "leg": LegBiLSTMClassifier,
    }
    folds = range(1, 6) if args.fold is None else (args.fold,)
    for fold_number in folds:
        for branch in branches:
            start_time = time.time()
            # Create DataLoaders for this fold and branch.
            dataloaders = create_and_save_dataloaders(args.data_dir, branch, fold_number, batch_size=32)
            # Train the model.
            train_all_folds(
                dataloaders,
                args.output_dir / "stage1" / branch / f"fold_{fold_number}",
                model_classes[branch],
                fold_number,
                branch=branch,
            )
            total_time = time.time() - start_time
            model_save_dir = args.output_dir / "stage1" / branch / f"fold_{fold_number}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            label = {"head": "Head", "upper": "Upper", "leg": "Leg"}[branch]
            print(f"{label} training time for fold {fold_number}: {total_time:.2f} seconds")
            (model_save_dir / "training_time.txt").write_text(
                f"{label} training time for fold {fold_number}: {total_time:.2f} seconds\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
