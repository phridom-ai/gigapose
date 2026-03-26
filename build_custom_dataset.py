#!/usr/bin/env python3
"""
Build a custom GigaPose dataset from per-camera image folders.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from convert_to_webdataset import convert_dataset_to_webdataset
from generate_scene_camera_from_calibration import generate_scene_camera, load_calibration
from undistort_images import collect_image_paths, undistort_rgb_images
from visualize_detections import visualize_detections


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def resolve_path(path_str, repo_root):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def recreate_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_static_assets(source_dataset_root, dataset_root):
    shutil.copytree(source_dataset_root / "models", dataset_root / "models")
    shutil.copy2(
        source_dataset_root / "systemCalibration.pkl",
        dataset_root / "systemCalibration.pkl",
    )


def normalize_output_suffix(image_path):
    return image_path.suffix.lower()


def copy_scene_images(source_dir, destination_rgb_dir):
    image_paths = collect_image_paths(source_dir)
    destination_rgb_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, image_path in enumerate(image_paths):
        suffix = normalize_output_suffix(image_path)
        dst = destination_rgb_dir / f"{idx:06d}{suffix}"
        shutil.copy2(image_path, dst)
        copied.append(dst)
    return copied


def discover_camera_dirs(input_root, calibration_serials):
    input_root = Path(input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    recognized = []
    ignored = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in calibration_serials:
            image_paths = [
                child
                for child in sorted(path.iterdir())
                if child.is_file() and child.suffix in IMAGE_SUFFIXES
            ]
            if not image_paths:
                raise ValueError(
                    f"Recognized camera folder {path} contains no images"
                )
            recognized.append((path.name, path, image_paths))
        else:
            ignored.append(path.name)
    return recognized, ignored


def run_cnos_batch_inference(
    repo_root,
    template_dir,
    image_dir,
    output_file,
    scene_id,
    stability_score_thresh,
    conf_threshold,
    num_max_dets,
):
    command = [
        "python3",
        "-m",
        "src.scripts.inference_custom_batch",
        "--template_dir",
        str(template_dir),
        "--image_dir",
        str(image_dir),
        "--output_file",
        str(output_file),
        "--stability_score_thresh",
        str(stability_score_thresh),
        "--conf_threshold",
        str(conf_threshold),
        "--num_max_dets",
        str(num_max_dets),
        "--scene_id",
        str(scene_id),
    ]
    subprocess.run(command, cwd=repo_root / "cnos", check=True)


def merge_detection_files(scene_detection_files, merged_output_file):
    merged = []
    for detection_file in scene_detection_files:
        merged.extend(json.loads(Path(detection_file).read_text()))
    merged = sorted(
        merged,
        key=lambda det: (
            int(det["scene_id"]),
            int(det["image_id"]),
            int(det.get("category_id", det.get("obj_id", 0))),
        ),
    )
    merged_output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_output_file.write_text(json.dumps(merged, indent=2))
    return merged


def write_manifest(dataset_root, manifest):
    manifest_path = dataset_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a custom multi-camera dataset for GigaPose"
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory whose immediate subdirectories are camera serials containing raw images",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the generated dataset under gigaPose_datasets/datasets/",
    )
    parser.add_argument(
        "--source-dataset-root",
        required=True,
        help="Existing dataset root used as the source of models/ and systemCalibration.pkl",
    )
    parser.add_argument(
        "--template-dir",
        help="CNOS custom template directory containing flat *.png templates",
    )
    parser.add_argument(
        "--datasets-root",
        default="gigaPose_datasets/datasets",
        help="Root directory containing datasets and default_detections",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha passed to undistortion and undistorted scene-camera generation",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="depth_scale written to scene_camera.json",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum number of samples per WebDataset shard",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.2,
        help="CNOS stability score threshold",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="CNOS confidence threshold",
    )
    parser.add_argument(
        "--num-max-dets",
        type=int,
        default=1,
        help="Maximum detections per image returned by CNOS",
    )
    parser.add_argument(
        "--visualize-detections",
        action="store_true",
        help="Generate per-scene detection visualization overlays",
    )
    parser.add_argument(
        "--skip-detections",
        action="store_true",
        help="Skip the CNOS detection and visualization stages",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the target dataset and detection outputs if they already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    datasets_root = resolve_path(args.datasets_root, repo_root)
    source_dataset_root = resolve_path(args.source_dataset_root, repo_root)
    input_root = resolve_path(args.input_root, repo_root)
    template_dir = (
        resolve_path(args.template_dir, repo_root) if args.template_dir else None
    )

    if args.visualize_detections and args.skip_detections:
        raise ValueError("--visualize-detections cannot be used with --skip-detections")
    if not args.skip_detections and template_dir is None:
        raise ValueError("--template-dir is required unless --skip-detections is used")
    if template_dir is not None and not template_dir.exists():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    if not (source_dataset_root / "models").exists():
        raise FileNotFoundError(
            f"Missing models directory in source dataset: {source_dataset_root / 'models'}"
        )
    if not (source_dataset_root / "systemCalibration.pkl").exists():
        raise FileNotFoundError(
            "Missing systemCalibration.pkl in source dataset root"
        )

    dataset_root = datasets_root / args.dataset_name
    detection_dir = (
        datasets_root
        / "default_detections"
        / "core19_model_based_unseen"
        / args.dataset_name
    )
    detection_file = detection_dir / f"{args.dataset_name}-test.json"

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Target dataset already exists: {dataset_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(dataset_root)
    if detection_dir.exists() and not args.skip_detections:
        if not args.overwrite:
            raise FileExistsError(
                f"Detection output already exists: {detection_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(detection_dir)

    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "test").mkdir(parents=True, exist_ok=True)
    copy_static_assets(source_dataset_root, dataset_root)

    calibration = load_calibration(dataset_root / "systemCalibration.pkl")
    recognized, ignored = discover_camera_dirs(input_root, calibration.keys())
    if not recognized:
        raise ValueError(
            f"No camera folders matching calibration serials were found in {input_root}"
        )

    scenes = []
    for scene_id, (camera_serial, source_dir, source_images) in enumerate(recognized):
        scene_dir = dataset_root / "test" / f"{scene_id:06d}"
        rgb_dir = scene_dir / "rgb"
        copied_images = copy_scene_images(source_dir, rgb_dir)
        undistort_result = undistort_rgb_images(
            rgb_dir=rgb_dir,
            calibration_pkl=dataset_root / "systemCalibration.pkl",
            camera_serial=camera_serial,
            alpha=args.alpha,
        )
        scene_camera_path, scene_camera, _ = generate_scene_camera(
            calibration_pkl=dataset_root / "systemCalibration.pkl",
            camera_serial=camera_serial,
            output=scene_dir / "scene_camera.json",
            rgb_dir=rgb_dir,
            depth_scale=args.depth_scale,
            undistort=True,
            alpha=args.alpha,
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "camera_serial": camera_serial,
                "source_dir": str(source_dir),
                "rgb_dir": str(rgb_dir),
                "num_source_images": len(source_images),
                "num_copied_images": len(copied_images),
                "scene_camera_path": str(scene_camera_path),
                "num_scene_camera_entries": len(scene_camera),
                "undistortion": undistort_result,
            }
        )

    webdataset_result = convert_dataset_to_webdataset(
        dataset_root=dataset_root,
        split="test",
        maxcount=args.maxcount,
    )

    scene_detection_files = []
    merged_detections = None
    if not args.skip_detections:
        tmp_detection_dir = dataset_root / "_tmp_scene_detections"
        recreate_directory(tmp_detection_dir)
        for scene in scenes:
            scene_detection_file = tmp_detection_dir / f"scene_{scene['scene_id']:06d}.json"
            run_cnos_batch_inference(
                repo_root=repo_root,
                template_dir=template_dir,
                image_dir=Path(scene["rgb_dir"]),
                output_file=scene_detection_file,
                scene_id=scene["scene_id"],
                stability_score_thresh=args.stability_score_thresh,
                conf_threshold=args.conf_threshold,
                num_max_dets=args.num_max_dets,
            )
            scene_detection_files.append(scene_detection_file)

        merged_detections = merge_detection_files(scene_detection_files, detection_file)
        shutil.rmtree(tmp_detection_dir)

        if args.visualize_detections:
            visualize_result = visualize_detections(
                dataset_root=dataset_root,
                detection_file=detection_file,
            )
        else:
            visualize_result = None
    else:
        visualize_result = None

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "source_dataset_root": str(source_dataset_root),
        "input_root": str(input_root),
        "template_dir": str(template_dir) if template_dir else None,
        "ignored_input_subdirs": ignored,
        "alpha": args.alpha,
        "depth_scale": args.depth_scale,
        "webdataset": webdataset_result,
        "detections": {
            "enabled": not args.skip_detections,
            "output_file": str(detection_file) if not args.skip_detections else None,
            "num_detections": len(merged_detections) if merged_detections is not None else None,
            "stability_score_thresh": args.stability_score_thresh,
            "conf_threshold": args.conf_threshold,
            "num_max_dets": args.num_max_dets,
        },
        "visualizations": visualize_result,
        "scenes": scenes,
    }
    manifest["manifest_path"] = str(dataset_root / "build_manifest.json")
    manifest_path = write_manifest(dataset_root, manifest)
    manifest["manifest_path"] = str(manifest_path)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
