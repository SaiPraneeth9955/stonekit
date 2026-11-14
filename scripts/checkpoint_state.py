"""Classify an nnU-Net fold folder for the E-CV launcher (scripts/train_cv_folds.sh).

Prints one word and exits 0:
  complete  checkpoint_final.pth loads, has current_epoch == expected epochs, and its weights are finite
  resume    no final checkpoint; checkpoint_latest.pth loads with 0 < current_epoch < expected and finite weights
  fresh     no checkpoint of either kind
Exits 2 (and prints the reason) for anything else, such as an unreadable or inconsistent checkpoint,
so the launcher stops instead of silently restarting or skipping a fold.
"""

import argparse
import sys
from pathlib import Path

import torch


def inspect(path: Path) -> tuple[int, bool]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["network_weights"]
    finite = all(torch.isfinite(t).all().item() for t in weights.values() if torch.is_floating_point(t))
    return int(checkpoint["current_epoch"]), finite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold_folder", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    final, latest = args.fold_folder / "checkpoint_final.pth", args.fold_folder / "checkpoint_latest.pth"
    try:
        if final.exists():
            epoch, finite = inspect(final)
            if epoch == args.epochs and finite:
                print("complete")
                return
            print(f"bad final checkpoint: epoch {epoch}, finite weights {finite}")
            sys.exit(2)
        if latest.exists():
            epoch, finite = inspect(latest)
            if 0 < epoch < args.epochs and finite:
                print("resume")
                return
            print(f"bad latest checkpoint: epoch {epoch}, finite weights {finite}")
            sys.exit(2)
    except Exception as error:  # noqa: BLE001  (any unreadable checkpoint must stop the launcher)
        print(f"unreadable checkpoint: {error!r}")
        sys.exit(2)
    print("fresh")


if __name__ == "__main__":
    main()
