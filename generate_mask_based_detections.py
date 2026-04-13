#!/usr/bin/env python3
"""
Generate GigaPose detection JSON from BOP mask_visib/mask files and test targets.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def load_binary_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    return (mask > 0).astype(np.uint8)


def encode_binary_mask(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = list(rle["size"])
    return rle


def expand_targets_by_instance(targets_path):
    targets = json.loads(Path(targets_path).read_text())
    expanded = defaultdict(list)
    for row in targets:
        key = (int(row["scene_id"]), int(row["im_id"]))
        obj_id = int(row["obj_id"])
        inst_count = int(row["inst_count"])
        expanded[key].extend([obj_id] * inst_count)
    return expanded


def list_instance_mask_paths(mask_dir, im_id):
    return sorted(mask_dir.glob(f"{im_id:06d}_*.png"))


def mask_score(vis_mask_path, full_mask_path):
    vis_mask = load_binary_mask(vis_mask_path)
    vis_area = int(vis_mask.sum())
    if vis_area == 0:
        raise ValueError(f"Visible mask is empty: {vis_mask_path}")

    if full_mask_path is None or not full_mask_path.exists():
        return vis_mask, 1.0

    full_mask = load_binary_mask(full_mask_path)
    full_area = int(full_mask.sum())
    if full_area == 0:
        return vis_mask, 1.0
    return vis_mask, float(vis_area / full_area)


def generate_detections(dataset_root, split="test", targets_path=None, output_path=None):
    dataset_root = Path(dataset_root)
    dataset_name = dataset_root.name
    split_root = dataset_root / split
    if targets_path is None:
        targets_path = dataset_root / "test_targets_bop19.json"
    else:
        targets_path = Path(targets_path)

    if output_path is None:
        output_path = (
            dataset_root.parent
            / "default_detections"
            / "core19_model_based_unseen"
            / dataset_name
            / f"{dataset_name}-{split}.json"
        )
    else:
        output_path = Path(output_path)

    targets_by_image = expand_targets_by_instance(targets_path)
    detections = []

    for (scene_id, im_id) in sorted(targets_by_image.keys()):
        scene_dir = split_root / f"{scene_id:06d}"
        mask_visib_dir = scene_dir / "mask_visib"
        mask_dir = scene_dir / "mask"
        vis_paths = list_instance_mask_paths(mask_visib_dir, im_id)
        obj_ids = targets_by_image[(scene_id, im_id)]

        if len(vis_paths) != len(obj_ids):
            raise ValueError(
                f"Mismatch for scene_id={scene_id} im_id={im_id}: "
                f"{len(vis_paths)} mask_visib files vs {len(obj_ids)} expanded targets"
            )

        for instance_idx, (vis_path, obj_id) in enumerate(zip(vis_paths, obj_ids)):
            full_mask_path = mask_dir / vis_path.name
            vis_mask, score = mask_score(vis_path, full_mask_path)
            rle = encode_binary_mask(vis_mask)
            bbox = [int(v) for v in mask_utils.toBbox(rle).tolist()]

            detections.append(
                {
                    "scene_id": scene_id,
                    "image_id": im_id,
                    "category_id": obj_id,
                    "score": score,
                    "bbox": bbox,
                    "segmentation": rle,
                    "time": 0.0,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(detections, indent=2))

    return {
        "dataset_root": str(dataset_root),
        "targets_path": str(targets_path),
        "output_path": str(output_path),
        "num_detections": len(detections),
        "num_images": len(targets_by_image),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GigaPose detections from BOP masks and test_targets_bop19.json"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root, e.g. gigaPose_datasets/datasets/hope",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split containing scene directories with mask_visib/",
    )
    parser.add_argument(
        "--targets-path",
        help="Optional path to test_targets_bop19.json",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = generate_detections(
        dataset_root=args.dataset_root,
        split=args.split,
        targets_path=args.targets_path,
        output_path=args.output,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
