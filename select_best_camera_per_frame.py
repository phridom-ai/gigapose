#!/usr/bin/env python3
"""
Select the highest-scoring camera prediction for each frame (im_id).
"""

import argparse
from collections import Counter
from pathlib import Path

from src.utils.inout import load_bop_results, save_bop_results


def default_output_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_best_camera.csv")


def select_best_camera_per_frame(input_path, output_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_output_path(input_path)

    results = load_bop_results(input_path)
    if not results:
        raise ValueError(f"No predictions found in {input_path}")

    best_by_im_id = {}
    for result in results:
        key = result["im_id"]
        current_best = best_by_im_id.get(key)
        if current_best is None:
            best_by_im_id[key] = result
            continue

        candidate_key = (result["score"], -result["scene_id"])
        current_key = (current_best["score"], -current_best["scene_id"])
        if candidate_key > current_key:
            best_by_im_id[key] = result

    selected = [best_by_im_id[key] for key in sorted(best_by_im_id.keys())]
    save_bop_results(output_path, selected)

    scene_counts = Counter(result["scene_id"] for result in selected)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_input_rows": len(results),
        "num_selected_rows": len(selected),
        "scene_counts": dict(sorted(scene_counts.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick the highest-scoring camera result for each im_id"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input BOP-format CSV with multiple scene_id rows per im_id",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path; defaults to <input>_best_camera.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = select_best_camera_per_frame(
        input_path=args.input,
        output_path=args.output,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
