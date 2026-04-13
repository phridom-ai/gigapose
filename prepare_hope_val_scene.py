#!/usr/bin/env python3
"""
Prepare a single HOPE validation scene as a standalone GigaPose test dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from convert_to_webdataset import convert_dataset_to_webdataset


SCENE_SUBDIRS = ("rgb", "depth", "mask", "mask_visib")
SCENE_FILES = ("scene_camera.json", "scene_gt.json", "scene_gt_info.json")


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_scene(source_scene_dir: Path, target_scene_dir: Path) -> None:
    target_scene_dir.mkdir(parents=True, exist_ok=True)
    for name in SCENE_SUBDIRS:
        shutil.copytree(source_scene_dir / name, target_scene_dir / name)
    for name in SCENE_FILES:
        shutil.copy2(source_scene_dir / name, target_scene_dir / name)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.array(Image.open(mask_path)) > 0).astype(np.uint8)


def encode_binary_mask(binary_mask: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = list(rle["size"])
    return rle


def score_visible_mask(vis_mask_path: Path, full_mask_path: Path) -> tuple[np.ndarray, float]:
    vis_mask = load_binary_mask(vis_mask_path)
    vis_area = int(vis_mask.sum())
    if vis_area == 0:
        raise ValueError(f"Visible mask is empty: {vis_mask_path}")

    if not full_mask_path.exists():
        return vis_mask, 1.0

    full_mask = load_binary_mask(full_mask_path)
    full_area = int(full_mask.sum())
    if full_area == 0:
        return vis_mask, 1.0
    return vis_mask, float(vis_area / full_area)


def build_test_targets_bop19(scene_gt: dict, scene_id: int) -> list[dict]:
    targets = []
    for im_id_str in sorted(scene_gt.keys(), key=lambda value: int(value)):
        im_id = int(im_id_str)
        obj_counts = Counter(int(entry["obj_id"]) for entry in scene_gt[im_id_str])
        for obj_id in sorted(obj_counts.keys()):
            targets.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": obj_id,
                    "inst_count": int(obj_counts[obj_id]),
                }
            )
    return targets


def build_test_targets_bop24(scene_gt: dict, scene_id: int) -> list[dict]:
    targets = []
    for im_id_str in sorted(scene_gt.keys(), key=lambda value: int(value)):
        targets.append({"scene_id": scene_id, "im_id": int(im_id_str)})
    return targets


def build_gt_detections(scene_dir: Path, scene_gt: dict, scene_id: int) -> list[dict]:
    detections = []
    mask_dir = scene_dir / "mask"
    mask_visib_dir = scene_dir / "mask_visib"

    for im_id_str in sorted(scene_gt.keys(), key=lambda value: int(value)):
        im_id = int(im_id_str)
        gt_entries = scene_gt[im_id_str]
        mask_paths = sorted(mask_visib_dir.glob(f"{im_id:06d}_*.png"))
        if len(mask_paths) != len(gt_entries):
            raise ValueError(
                f"Scene {scene_dir.name} image {im_id:06d}: "
                f"{len(gt_entries)} GT instances but {len(mask_paths)} mask_visib files"
            )

        for gt_entry, vis_mask_path in zip(gt_entries, mask_paths):
            full_mask_path = mask_dir / vis_mask_path.name
            vis_mask, score = score_visible_mask(vis_mask_path, full_mask_path)
            rle = encode_binary_mask(vis_mask)
            bbox = [int(value) for value in mask_utils.toBbox(rle).tolist()]
            detections.append(
                {
                    "scene_id": scene_id,
                    "image_id": im_id,
                    "category_id": int(gt_entry["obj_id"]),
                    "score": score,
                    "bbox": bbox,
                    "segmentation": rle,
                    "time": 0.0,
                }
            )

    return detections


def select_obj_ids(scene_gt: dict, keep_obj_ids: list[int] | None, max_objects: int | None) -> list[int]:
    available_obj_ids = sorted(
        {int(entry["obj_id"]) for entries in scene_gt.values() for entry in entries}
    )
    if keep_obj_ids:
        requested = sorted(dict.fromkeys(int(obj_id) for obj_id in keep_obj_ids))
        missing = [obj_id for obj_id in requested if obj_id not in available_obj_ids]
        if missing:
            raise ValueError(
                f"Requested object ids are not present in the scene: {missing}. "
                f"Available ids: {available_obj_ids}"
            )
        return requested

    if max_objects is None or max_objects >= len(available_obj_ids):
        return available_obj_ids

    first_im_id = sorted(scene_gt.keys(), key=lambda value: int(value))[0]
    first_counts = Counter(int(entry["obj_id"]) for entry in scene_gt[first_im_id])
    single_instance_ids = sorted(obj_id for obj_id, count in first_counts.items() if count == 1)
    repeated_ids = sorted(obj_id for obj_id, count in first_counts.items() if count > 1)
    selected = (single_instance_ids + repeated_ids)[:max_objects]
    return selected


def filter_scene_annotations(
    scene_gt: dict,
    scene_gt_info: dict,
    obj_id_mapping: dict[int, int],
) -> tuple[dict, dict, dict]:
    filtered_gt = {}
    filtered_gt_info = {}
    kept_indices = {}

    for im_id_str in sorted(scene_gt.keys(), key=lambda value: int(value)):
        gt_entries = scene_gt[im_id_str]
        gt_info_entries = scene_gt_info[im_id_str]
        if len(gt_entries) != len(gt_info_entries):
            raise ValueError(
                f"scene_gt and scene_gt_info length mismatch for image {im_id_str}: "
                f"{len(gt_entries)} vs {len(gt_info_entries)}"
            )

        kept_gt_entries = []
        kept_gt_info_entries = []
        kept_idx_list = []
        for idx, (gt_entry, gt_info_entry) in enumerate(zip(gt_entries, gt_info_entries)):
            original_obj_id = int(gt_entry["obj_id"])
            if original_obj_id not in obj_id_mapping:
                continue
            remapped_entry = dict(gt_entry)
            remapped_entry["obj_id"] = int(obj_id_mapping[original_obj_id])
            kept_gt_entries.append(remapped_entry)
            kept_gt_info_entries.append(gt_info_entry)
            kept_idx_list.append(idx)

        filtered_gt[im_id_str] = kept_gt_entries
        filtered_gt_info[im_id_str] = kept_gt_info_entries
        kept_indices[im_id_str] = kept_idx_list

    return filtered_gt, filtered_gt_info, kept_indices


def rewrite_filtered_masks(scene_dir: Path, kept_indices: dict) -> None:
    for subdir_name in ("mask", "mask_visib"):
        source_dir = scene_dir / subdir_name
        tmp_dir = scene_dir / f"_{subdir_name}_filtered"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for im_id_str in sorted(kept_indices.keys(), key=lambda value: int(value)):
            im_id = int(im_id_str)
            original_paths = sorted(source_dir.glob(f"{im_id:06d}_*.png"))
            selected_indices = kept_indices[im_id_str]
            if original_paths and len(original_paths) <= max(selected_indices, default=-1):
                raise ValueError(
                    f"Not enough {subdir_name} files for image {im_id:06d}: "
                    f"have {len(original_paths)}, need index {max(selected_indices)}"
                )
            for new_idx, original_idx in enumerate(selected_indices):
                shutil.copy2(
                    original_paths[original_idx],
                    tmp_dir / f"{im_id:06d}_{new_idx:06d}.png",
                )

        shutil.rmtree(source_dir)
        tmp_dir.rename(source_dir)


def build_obj_id_mapping(selected_obj_ids: list[int], remap_to_contiguous: bool) -> dict[int, int]:
    if remap_to_contiguous:
        return {int(original_obj_id): idx + 1 for idx, original_obj_id in enumerate(selected_obj_ids)}
    return {int(obj_id): int(obj_id) for obj_id in selected_obj_ids}


def build_template_subset(
    source_template_dir: Path,
    target_template_dir: Path,
    obj_id_mapping: dict[int, int],
) -> None:
    target_template_dir.mkdir(parents=True, exist_ok=True)
    (target_template_dir / "object_poses").mkdir(parents=True, exist_ok=True)
    source_preprocessed_dir = source_template_dir / "preprocessed"
    if source_preprocessed_dir.exists():
        (target_template_dir / "preprocessed").mkdir(parents=True, exist_ok=True)

    for original_obj_id, new_obj_id in obj_id_mapping.items():
        target_object_dir = target_template_dir / f"{new_obj_id:06d}"
        target_object_dir.symlink_to((source_template_dir / f"{original_obj_id:06d}").resolve())

        source_pose = source_template_dir / "object_poses" / f"{original_obj_id:06d}.npy"
        target_pose = target_template_dir / "object_poses" / f"{new_obj_id:06d}.npy"
        target_pose.symlink_to(source_pose.resolve())

        source_preprocessed = source_template_dir / "preprocessed" / f"{original_obj_id:06d}.npz"
        if source_preprocessed.exists():
            target_preprocessed = target_template_dir / "preprocessed" / f"{new_obj_id:06d}.npz"
            target_preprocessed.symlink_to(source_preprocessed.resolve())


def build_model_subset(
    source_dataset_root: Path,
    target_models_dir: Path,
    obj_id_mapping: dict[int, int],
) -> dict:
    source_models_dir = source_dataset_root / "models"
    source_models_info = load_json(source_dataset_root / "models_eval" / "models_info.json")

    target_models_dir.mkdir(parents=True, exist_ok=True)
    remapped_models_info = {}
    for original_obj_id, new_obj_id in obj_id_mapping.items():
        shutil.copy2(
            source_models_dir / f"obj_{original_obj_id:06d}.ply",
            target_models_dir / f"obj_{new_obj_id:06d}.ply",
        )

        source_texture = source_models_dir / f"obj_{original_obj_id:06d}.png"
        if source_texture.exists():
            shutil.copy2(source_texture, target_models_dir / source_texture.name)
            shutil.copy2(source_texture, target_models_dir / f"obj_{new_obj_id:06d}.png")

        remapped_models_info[str(new_obj_id)] = source_models_info[str(original_obj_id)]

    save_json(target_models_dir / "models_info.json", remapped_models_info)
    return remapped_models_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone HOPE val/000001 into a standalone GigaPose-compatible test dataset."
    )
    parser.add_argument(
        "--datasets-root",
        default="gigaPose_datasets/datasets",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--source-dataset",
        default="hope",
        help="Source dataset name under --datasets-root.",
    )
    parser.add_argument("--source-split", default="val", help="Source split name.")
    parser.add_argument(
        "--source-scene-id",
        type=int,
        default=1,
        help="Source scene id inside the source split.",
    )
    parser.add_argument(
        "--target-dataset",
        default="hope_val_000001",
        help="Target dataset name to create under --datasets-root.",
    )
    parser.add_argument("--target-split", default="test", help="Target split name.")
    parser.add_argument(
        "--target-scene-id",
        type=int,
        default=1,
        help="Target scene id inside the target split.",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum number of samples per generated webdataset shard.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Optional limit on the number of object ids kept in the cloned scene.",
    )
    parser.add_argument(
        "--keep-obj-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit object ids to keep in the cloned scene.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the target dataset and custom detection file if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    datasets_root = (repo_root / args.datasets_root).resolve()

    source_dataset_root = datasets_root / args.source_dataset
    source_scene_dir = (
        source_dataset_root / args.source_split / f"{args.source_scene_id:06d}"
    )
    if not source_scene_dir.exists():
        raise FileNotFoundError(f"Source scene not found: {source_scene_dir}")

    target_dataset_root = datasets_root / args.target_dataset
    target_scene_dir = target_dataset_root / args.target_split / f"{args.target_scene_id:06d}"
    detection_file = (
        datasets_root
        / "default_detections"
        / "core19_model_based_unseen"
        / args.target_dataset
        / f"{args.target_dataset}-{args.target_split}.json"
    )
    template_root = datasets_root / "templates"
    target_template_link = template_root / args.target_dataset
    source_template_dir = template_root / args.source_dataset
    filtering_active = args.max_objects is not None or args.keep_obj_ids is not None

    if target_dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Target dataset already exists: {target_dataset_root}. Use --overwrite to replace it."
            )
        remove_path(target_dataset_root)

    if detection_file.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Target detection file already exists: {detection_file}. Use --overwrite to replace it."
            )
        remove_path(detection_file.parent)

    if target_template_link.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Target template link already exists: {target_template_link}. Use --overwrite to replace it."
            )
        remove_path(target_template_link)

    target_dataset_root.mkdir(parents=True, exist_ok=True)
    (target_dataset_root / args.target_split).mkdir(parents=True, exist_ok=True)

    copy_scene(source_scene_dir, target_scene_dir)

    scene_gt = load_json(target_scene_dir / "scene_gt.json")
    scene_gt_info = load_json(target_scene_dir / "scene_gt_info.json")
    selected_obj_ids = select_obj_ids(
        scene_gt=scene_gt,
        keep_obj_ids=args.keep_obj_ids,
        max_objects=args.max_objects,
    )
    obj_id_mapping = build_obj_id_mapping(
        selected_obj_ids=selected_obj_ids,
        remap_to_contiguous=filtering_active,
    )
    filtered_scene_gt, filtered_scene_gt_info, kept_indices = filter_scene_annotations(
        scene_gt=scene_gt,
        scene_gt_info=scene_gt_info,
        obj_id_mapping=obj_id_mapping,
    )
    rewrite_filtered_masks(target_scene_dir, kept_indices)
    save_json(target_scene_dir / "scene_gt.json", filtered_scene_gt)
    save_json(target_scene_dir / "scene_gt_info.json", filtered_scene_gt_info)

    if filtering_active:
        build_model_subset(
            source_dataset_root=source_dataset_root,
            target_models_dir=target_dataset_root / "models",
            obj_id_mapping=obj_id_mapping,
        )
        build_template_subset(
            source_template_dir=source_template_dir,
            target_template_dir=target_template_link,
            obj_id_mapping=obj_id_mapping,
        )
    else:
        shutil.copytree(
            source_dataset_root / "models",
            target_dataset_root / "models",
            ignore=shutil.ignore_patterns("models_info copy.json"),
        )
        full_models_info = load_json(source_dataset_root / "models_eval" / "models_info.json")
        save_json(target_dataset_root / "models" / "models_info.json", full_models_info)
        target_template_link.parent.mkdir(parents=True, exist_ok=True)
        target_template_link.symlink_to(source_template_dir.name)

    save_json(
        target_dataset_root / "object_id_mapping.json",
        {
            "new_to_original": {str(new_obj_id): int(original_obj_id) for original_obj_id, new_obj_id in obj_id_mapping.items()},
            "original_to_new": {str(original_obj_id): int(new_obj_id) for original_obj_id, new_obj_id in obj_id_mapping.items()},
        },
    )

    save_json(
        target_dataset_root / "test_targets_bop19.json",
        build_test_targets_bop19(filtered_scene_gt, scene_id=args.target_scene_id),
    )
    save_json(
        target_dataset_root / "test_targets_bop24.json",
        build_test_targets_bop24(filtered_scene_gt, scene_id=args.target_scene_id),
    )
    save_json(
        detection_file,
        build_gt_detections(target_scene_dir, filtered_scene_gt, scene_id=args.target_scene_id),
    )

    webdataset_result = convert_dataset_to_webdataset(
        dataset_root=target_dataset_root,
        split=args.target_split,
        maxcount=args.maxcount,
    )

    summary = {
        "target_dataset_root": str(target_dataset_root),
        "target_scene_dir": str(target_scene_dir),
        "template_link": str(target_template_link),
        "detection_file": str(detection_file),
        "selected_original_obj_ids": selected_obj_ids,
        "object_id_mapping": {str(original_obj_id): int(new_obj_id) for original_obj_id, new_obj_id in obj_id_mapping.items()},
        "num_filtered_instances": sum(len(entries) for entries in filtered_scene_gt.values()),
        "webdataset": webdataset_result,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
