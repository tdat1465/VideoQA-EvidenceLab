"""Fetch pinned author source, then exercise the CPU selector (no GPU/models)."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from evidencelab.ridge_adapter import fetch_ridge, load_ridge, selector_args
from evidencelab.longvideo_config import LongVideoConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    frame, weight = load_ridge(fetch_ridge(args.target))
    weights = {key: 5 for key in weight.WEIGHT_KEYS}
    chosen = frame.gradselect([.1, .1, .2, .9, .8, .3, .1, .2, .5, .1], 4,
                              weights, selector_args(LongVideoConfig(method="ridge")))
    assert len(chosen) == 4 and chosen == sorted(set(chosen))
    print(f"Pinned RIDGE source verified; CPU selection PASS: {chosen}")


if __name__ == "__main__":
    main()
