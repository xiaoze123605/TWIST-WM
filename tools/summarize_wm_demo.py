#!/usr/bin/env python
"""Print the compact clean/corrupt/WM comparison used by the demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    required = ("joint_rmse", "root_error", "max_tilt")
    missing = [name for name in required if name not in summary]
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the TWIST + Motion-WM demo")
    parser.add_argument("--clean", type=Path, default=Path("clean_summary.json"))
    parser.add_argument("--corrupt", type=Path, default=Path("corrupt_summary.json"))
    parser.add_argument("--wm", type=Path, default=Path("wm_summary.json"))
    args = parser.parse_args()

    rows = (
        ("Clean", load_summary(args.clean)),
        ("Corrupt", load_summary(args.corrupt)),
        ("Corrupt + WM", load_summary(args.wm)),
    )
    print(f"{'Mode':<16} {'Joint RMSE':>12} {'Root error':>12} {'Max tilt':>12}")
    for label, metrics in rows:
        print(
            f"{label:<16} {metrics['joint_rmse']:>12.5f} "
            f"{metrics['root_error']:>12.5f} {metrics['max_tilt']:>12.5f}"
        )

    corrupt = rows[1][1]
    wm = rows[2][1]
    print(
        "WM improvement: "
        f"joint_rmse={corrupt['joint_rmse'] - wm['joint_rmse']:+.5f}, "
        f"root_error={corrupt['root_error'] - wm['root_error']:+.5f}, "
        f"max_tilt={corrupt['max_tilt'] - wm['max_tilt']:+.5f} "
        "(positive is better)"
    )


if __name__ == "__main__":
    main()
