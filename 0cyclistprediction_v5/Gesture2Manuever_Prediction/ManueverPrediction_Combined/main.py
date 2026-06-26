from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ManueverPrediction_Combined.dataloader import create_and_save_dataloaders
from ManueverPrediction_Combined.model_training import train_all_folds


def parse_arguments(argv=None):
    # Parse command-line arguments.
    release_root = Path(__file__).resolve().parents[2]
    run_root = release_root.parent
    parser = argparse.ArgumentParser(description="Train maneuver classifier")
    parser.add_argument("--data-dir", type=Path, default=run_root / "release_data" / "ManueverDataset", help="directory created by ManueverCrop_combine.py")
    parser.add_argument("--output-dir", type=Path, default=run_root / "release_runs")
    parser.add_argument("--stage1-checkpoint-dir", type=Path, default=run_root / "release_runs")
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    folds = range(1, 6) if args.fold is None else (args.fold,)
    for fold_number in folds:
        model_save_dir = args.output_dir / "stage2" / f"fold_{fold_number}"
        start_time = time.time()
        # Create DataLoaders for this fold.
        dataloaders = create_and_save_dataloaders(args.data_dir, fold_number, batch_size=32)
        # Train the maneuver prediction model.
        train_all_folds(
            dataloaders,
            model_save_dir,
            fold_number,
            stage1_checkpoint_dir=args.stage1_checkpoint_dir,
        )
        total_time = time.time() - start_time
        print(f"Total training time for fold {fold_number}: {total_time:.2f} seconds")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        (model_save_dir / "training_time.txt").write_text(
            f"Total training time for fold {fold_number}: {total_time:.2f} seconds\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
