#!/usr/bin/env python3
"""
Convert a BOP-style test split into WebDataset shards for GigaPose.
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def find_image_path(rgb_dir, image_id):
    stem = f"{int(image_id):06d}"
    for suffix in IMAGE_SUFFIXES:
        path = rgb_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find RGB image for image_id={stem} in {rgb_dir}")


def iter_scene_dirs(test_dir):
    scene_dirs = [path for path in sorted(test_dir.iterdir()) if path.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {test_dir}")
    return scene_dirs


def build_key_to_shard_map(output_dir):
    key_to_shard = {}
    for shard_path in sorted(output_dir.glob("shard-*.tar")):
        with tarfile.open(shard_path) as tar:
            names = tar.getnames()
        keys = {name.split(".")[0] for name in names}
        for key in keys:
            key_to_shard[key] = shard_path.name
    return key_to_shard


def cleanup_previous_outputs(output_dir):
    for shard_path in output_dir.glob("shard-*.tar"):
        shard_path.unlink()
    key_to_shard_path = output_dir / "key_to_shard.json"
    if key_to_shard_path.exists():
        key_to_shard_path.unlink()


def convert_dataset_to_webdataset(dataset_root, split="test", maxcount=1000):
    dataset_root = Path(dataset_root)
    test_dir = dataset_root / split
    output_dir = test_dir
    if maxcount < 1:
        raise ValueError("maxcount must be >= 1")

    cleanup_previous_outputs(output_dir)

    num_samples = 0
    num_shards = 0
    samples_in_shard = 0
    tar = None
    current_shard_name = None
    key_to_shard = {}

    def _close_tar():
        nonlocal tar
        if tar is not None:
            tar.close()
            tar = None

    def _open_next_shard():
        nonlocal tar, num_shards, samples_in_shard, current_shard_name
        _close_tar()
        current_shard_name = f"shard-{num_shards:06d}.tar"
        tar = tarfile.open(output_dir / current_shard_name, "w")
        num_shards += 1
        samples_in_shard = 0

    for scene_dir in iter_scene_dirs(test_dir):
        scene_id = int(scene_dir.name)
        rgb_dir = scene_dir / "rgb"
        scene_camera_path = scene_dir / "scene_camera.json"
        if not rgb_dir.exists():
            raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")
        if not scene_camera_path.exists():
            raise FileNotFoundError(f"Missing scene_camera.json: {scene_camera_path}")

        scene_camera = json.loads(scene_camera_path.read_text())
        image_ids = sorted(int(image_id) for image_id in scene_camera.keys())
        for image_id in image_ids:
            if tar is None or samples_in_shard >= maxcount:
                _open_next_shard()
            rgb_path = find_image_path(rgb_dir, image_id)
            key = f"{scene_id:06d}_{image_id:06d}"
            suffix = rgb_path.suffix.lower()
            if suffix == ".jpeg":
                suffix = ".jpg"
            with open(rgb_path, "rb") as f:
                rgb_data = f.read()
            camera_json = json.dumps(scene_camera[str(image_id)]).encode("utf-8")

            rgb_info = tarfile.TarInfo(name=f"{key}.rgb{suffix}")
            rgb_info.size = len(rgb_data)
            tar.addfile(rgb_info, fileobj=io.BytesIO(rgb_data))

            camera_info = tarfile.TarInfo(name=f"{key}.camera.json")
            camera_info.size = len(camera_json)
            tar.addfile(camera_info, fileobj=io.BytesIO(camera_json))

            key_to_shard[key] = current_shard_name
            samples_in_shard += 1
            num_samples += 1

    _close_tar()
    key_to_shard_path = output_dir / "key_to_shard.json"
    key_to_shard_path.write_text(json.dumps(key_to_shard, indent=2))

    return {
        "num_samples": num_samples,
        "num_shards": num_shards,
        "key_to_shard_path": str(key_to_shard_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a custom BOP test split to WebDataset")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root, e.g. gigaPose_datasets/datasets/my_dataset",
    )
    parser.add_argument("--split", default="test", help="Dataset split to convert")
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum number of samples per shard",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = convert_dataset_to_webdataset(
        dataset_root=args.dataset_root,
        split=args.split,
        maxcount=args.maxcount,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
