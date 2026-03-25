#!/usr/bin/env python3
"""
Generate a BOP-style scene_camera.json from a systemCalibration.pkl file.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_calibration(calibration_path):
    with open(calibration_path, "rb") as f:
        calibration = pickle.load(f)
    if not isinstance(calibration, dict):
        raise ValueError(f"Expected calibration pickle to contain a dict, got {type(calibration).__name__}")
    return {str(key): value for key, value in calibration.items()}


def get_image_ids(rgb_dir):
    image_paths = sorted(list(Path(rgb_dir).glob("*.jpg")) + list(Path(rgb_dir).glob("*.png")))
    if not image_paths:
        raise ValueError(f"No .jpg or .png files found in {rgb_dir}")

    image_ids = []
    for image_path in image_paths:
        try:
            image_ids.append(int(image_path.stem))
        except ValueError as exc:
            raise ValueError(
                f"Image filename {image_path.name} does not have an integer stem required for BOP image ids"
            ) from exc
    return image_ids


def flatten_K(K):
    return np.asarray(K, dtype=np.float64).reshape(3, 3).reshape(-1).tolist()


def compute_output_K(entry, undistort, alpha):
    K = np.asarray(entry["K"], dtype=np.float64).reshape(3, 3)
    if not undistort:
        return K

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for --undistort") from exc

    D = np.asarray(entry["D"], dtype=np.float64).reshape(-1)
    width = int(entry["imgW"])
    height = int(entry["imgH"])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha, (width, height))
    return new_K


def build_scene_camera(image_ids, K, depth_scale):
    cam_K = flatten_K(K)
    return {
        str(image_id): {
            "cam_K": cam_K,
            "depth_scale": depth_scale,
        }
        for image_id in image_ids
    }


def main():
    parser = argparse.ArgumentParser(description="Generate scene_camera.json from systemCalibration.pkl")
    parser.add_argument(
        "--calibration_pkl",
        required=True,
        help="Path to systemCalibration.pkl",
    )
    parser.add_argument(
        "--camera_serial",
        required=True,
        help="Camera serial number to extract from the calibration file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for scene_camera.json",
    )
    parser.add_argument(
        "--rgb_dir",
        help="Directory of RGB images; filenames are used as BOP image ids",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        help="If rgb_dir is not provided, generate entries for image ids 0..num_images-1",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="depth_scale value written to each scene_camera entry",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Use cv2.getOptimalNewCameraMatrix to generate an undistorted K from calibration K and D",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha passed to cv2.getOptimalNewCameraMatrix when --undistort is used",
    )
    parser.add_argument(
        "--list_serials",
        action="store_true",
        help="Print available camera serials and exit",
    )
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_pkl)

    if args.list_serials:
        for serial in sorted(calibration.keys()):
            print(serial)
        return

    if args.rgb_dir:
        image_ids = get_image_ids(args.rgb_dir)
    elif args.num_images is not None:
        if args.num_images < 1:
            raise ValueError("--num_images must be >= 1")
        image_ids = list(range(args.num_images))
    else:
        raise ValueError("Provide either --rgb_dir or --num_images")

    if args.camera_serial not in calibration:
        available = ", ".join(sorted(calibration.keys()))
        raise KeyError(f"Camera serial {args.camera_serial!r} not found. Available serials: {available}")

    entry = calibration[args.camera_serial]
    K = compute_output_K(entry, undistort=args.undistort, alpha=args.alpha)
    scene_camera = build_scene_camera(image_ids, K, args.depth_scale)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_camera, f, indent=2)

    print(f"Camera serial: {args.camera_serial}")
    print(f"Images: {len(image_ids)}")
    print(f"Undistort: {args.undistort}")
    print(f"Output: {output_path}")
    print("K:")
    print(np.asarray(K))


if __name__ == "__main__":
    main()
