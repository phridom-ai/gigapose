#!/usr/bin/env python3
"""
Undistort RGB images for a single camera scene.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from generate_scene_camera_from_calibration import load_calibration

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_image_paths(rgb_dir):
    rgb_dir = Path(rgb_dir)
    image_paths = [
        path
        for path in sorted(rgb_dir.iterdir())
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise ValueError(f"No images found in {rgb_dir}")
    return image_paths


def compute_new_camera_matrix(calibration_entry, width, height, alpha):
    original_K = np.asarray(calibration_entry["K"], dtype=np.float64).reshape(3, 3)
    distortion_coeffs = np.asarray(calibration_entry["D"], dtype=np.float64).reshape(-1)
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        original_K,
        distortion_coeffs,
        (width, height),
        alpha,
        (width, height),
    )
    return original_K, distortion_coeffs, new_K, roi


def update_scene_camera_intrinsics(scene_camera_path, new_K):
    scene_camera_path = Path(scene_camera_path)
    scene_camera = json.loads(scene_camera_path.read_text())
    new_K_flat = np.asarray(new_K, dtype=np.float64).reshape(-1).tolist()
    for cam_id in scene_camera.keys():
        scene_camera[cam_id]["cam_K"] = new_K_flat
    scene_camera_path.write_text(json.dumps(scene_camera, indent=2))


def undistort_rgb_images(
    rgb_dir,
    calibration_pkl,
    camera_serial,
    alpha=0.0,
    scene_camera_path=None,
    backup_dir=None,
):
    rgb_dir = Path(rgb_dir)
    calibration = load_calibration(calibration_pkl)
    if camera_serial not in calibration:
        available = ", ".join(sorted(calibration.keys()))
        raise KeyError(
            f"Camera serial {camera_serial!r} not found. Available serials: {available}"
        )

    image_paths = collect_image_paths(rgb_dir)
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise ValueError(f"Could not read image {image_paths[0]}")
    height, width = first_img.shape[:2]

    original_K, distortion_coeffs, new_K, roi = compute_new_camera_matrix(
        calibration[camera_serial], width=width, height=height, alpha=alpha
    )

    backup_path = None
    if backup_dir is not None:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc=f"Undistorting {rgb_dir.name}"):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        if backup_path is not None:
            shutil.copy2(image_path, backup_path / image_path.name)
        undistorted_img = cv2.undistort(
            img, original_K, distortion_coeffs, None, new_K
        )
        cv2.imwrite(str(image_path), undistorted_img)

    if scene_camera_path is not None:
        update_scene_camera_intrinsics(scene_camera_path, new_K)

    return {
        "image_count": len(image_paths),
        "image_size": [width, height],
        "original_K": original_K.tolist(),
        "distortion_coeffs": distortion_coeffs.tolist(),
        "new_K": np.asarray(new_K, dtype=np.float64).tolist(),
        "roi": list(roi),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Undistort RGB images for one scene")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB images")
    parser.add_argument(
        "--calibration-pkl",
        required=True,
        help="Path to systemCalibration.pkl",
    )
    parser.add_argument(
        "--camera-serial",
        required=True,
        help="Camera serial present in the calibration pickle",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha passed to cv2.getOptimalNewCameraMatrix",
    )
    parser.add_argument(
        "--scene-camera",
        help="Optional scene_camera.json to update with the undistorted intrinsics",
    )
    parser.add_argument(
        "--backup-dir",
        help="Optional directory where original images are copied before undistortion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = undistort_rgb_images(
        rgb_dir=args.rgb_dir,
        calibration_pkl=args.calibration_pkl,
        camera_serial=args.camera_serial,
        alpha=args.alpha,
        scene_camera_path=args.scene_camera,
        backup_dir=args.backup_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
