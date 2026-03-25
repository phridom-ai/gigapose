#!/usr/bin/env python3
"""
Undistort images and update camera intrinsics for GigaPose.
Removes lens distortion and generates new camera matrix.
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ==================== CONFIGURATION ====================

# Input/Output directories
RGB_DIR = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/test/000000/rgb")
SCENE_CAMERA_PATH = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/test/000000/scene_camera.json")
SYSTEM_CALIBRATION_PATH = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/systemCalibration.pkl")
CAMERA_SERIAL = "233500779"

# Alpha parameter for getOptimalNewCameraMatrix
# 0 = crop to remove all black pixels (loses field of view)
# 1 = keep all pixels (may have black borders)
ALPHA = 0.0

# Whether to save a backup of original images
SAVE_BACKUP = True

# ==================== MAIN SCRIPT ====================

def undistort_images():
    """Undistort all images in the RGB directory and update camera matrix."""
    
    print("="*70)
    print("Image Undistortion Script for GigaPose")
    print("="*70)
    print(f"\nInput directory: {RGB_DIR}")
    print(f"Camera config:   {SCENE_CAMERA_PATH}")
    print(f"Calibration pkl: {SYSTEM_CALIBRATION_PATH}")
    print(f"Camera serial:   {CAMERA_SERIAL}")
    
    # Verify directories exist
    if not RGB_DIR.exists():
        print(f"\n❌ Error: RGB directory not found: {RGB_DIR}")
        return
    
    if not SCENE_CAMERA_PATH.exists():
        print(f"\n❌ Error: scene_camera.json not found: {SCENE_CAMERA_PATH}")
        return

    if not SYSTEM_CALIBRATION_PATH.exists():
        print(f"\n❌ Error: systemCalibration.pkl not found: {SYSTEM_CALIBRATION_PATH}")
        return

    # Load original camera intrinsics and distortion coefficients from calibration file
    with open(SYSTEM_CALIBRATION_PATH, "rb") as f:
        calibration = pickle.load(f)

    calibration = {str(key): value for key, value in calibration.items()}
    if CAMERA_SERIAL not in calibration:
        available = ", ".join(sorted(calibration.keys()))
        print(f"\n❌ Error: Camera serial {CAMERA_SERIAL} not found in calibration file")
        print(f"Available serials: {available}")
        return

    camera_calibration = calibration[CAMERA_SERIAL]
    original_K = np.asarray(camera_calibration["K"], dtype=np.float64).reshape(3, 3)
    distortion_coeffs = np.asarray(camera_calibration["D"], dtype=np.float64).reshape(-1)
    
    # Get list of images
    image_files = sorted(list(RGB_DIR.glob("*.jpg")) + list(RGB_DIR.glob("*.png")))
    if not image_files:
        print(f"\n❌ Error: No images found in {RGB_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Load first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"\n❌ Error: Could not read image: {image_files[0]}")
        return
    
    h, w = first_img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Compute optimal new camera matrix
    print(f"\nComputing optimal camera matrix (alpha={ALPHA})...")
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        original_K,
        distortion_coeffs,
        (w, h), 
        ALPHA, 
        (w, h)
    )
    
    print("\nOriginal K matrix:")
    print(original_K)
    print("\nDistortion coefficients:")
    print(distortion_coeffs)
    print("\nNew K matrix (after undistortion):")
    print(new_K)
    print(f"\nROI (region of interest): x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Create backup directory if needed
    if SAVE_BACKUP:
        backup_dir = RGB_DIR.parent / "rgb_distorted_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"\nBackup directory: {backup_dir}")
    
    # Undistort all images
    print(f"\nUndistorting {len(image_files)} images...")
    for img_path in tqdm(image_files, desc="Processing"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n⚠️  Warning: Could not read {img_path}, skipping")
            continue
        
        # Backup original if requested
        if SAVE_BACKUP:
            backup_path = backup_dir / img_path.name
            cv2.imwrite(str(backup_path), img)
        
        # Undistort
        undistorted_img = cv2.undistort(img, original_K, distortion_coeffs, None, new_K)
        
        # Save undistorted image (overwrite original)
        cv2.imwrite(str(img_path), undistorted_img)
    
    # Update scene_camera.json with new K matrix
    print(f"\nUpdating camera intrinsics in {SCENE_CAMERA_PATH}...")
    with open(SCENE_CAMERA_PATH, 'r') as f:
        scene_camera = json.load(f)
    
    # Backup original scene_camera.json
    if SAVE_BACKUP:
        backup_camera_path = SCENE_CAMERA_PATH.parent / "scene_camera_distorted_backup.json"
        with open(backup_camera_path, 'w') as f:
            json.dump(scene_camera, f, indent=2)
        print(f"Backed up original camera config to: {backup_camera_path}")
    
    # Update all camera entries with new K matrix
    new_K_flat = new_K.flatten().tolist()
    for cam_id in scene_camera.keys():
        scene_camera[cam_id]["cam_K"] = new_K_flat
    
    # Save updated scene_camera.json
    with open(SCENE_CAMERA_PATH, 'w') as f:
        json.dump(scene_camera, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"✓ Undistorted {len(image_files)} images")
    print(f"✓ Updated camera intrinsics in scene_camera.json")
    if SAVE_BACKUP:
        print(f"✓ Originals backed up to:")
        print(f"  - {backup_dir}")
        print(f"  - {backup_camera_path}")
    print("\n📋 Next steps:")
    print("  1. Re-run your detection/annotation script on the undistorted images")
    print("  2. Regenerate WebDataset: python convert_to_webdataset.py")
    print("  3. Test with GigaPose: python test.py test_dataset_name=my_objects ...")
    print("\n💡 Tip: Check a few undistorted images visually to verify quality")
    print("="*70)


if __name__ == "__main__":
    undistort_images()
