#!/usr/bin/env python3
"""
Rename images in a directory to sequential format: 000000.jpg, 000001.jpg, etc.
"""

import shutil
from pathlib import Path

# ==================== CONFIGURATION ====================

RGB_DIR = Path("gigaPose_datasets/datasets/measuring_tape_distorted/test/000000/rgb")

# Image format (jpg or png)
OUTPUT_EXTENSION = ".jpg"

# Number of digits for zero-padding (6 = 000000, 000001, ...)
PADDING = 6

# ==================== MAIN SCRIPT ====================

def rename_images_sequential():
    """Rename all images to sequential format."""
    
    print("="*70)
    print("Sequential Image Renaming Script")
    print("="*70)
    print(f"\nDirectory: {RGB_DIR}")
    
    if not RGB_DIR.exists():
        print(f"\n❌ Error: Directory not found: {RGB_DIR}")
        return
    
    # Get all image files (jpg, png, jpeg)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(RGB_DIR.glob(ext))
    
    if not image_files:
        print(f"\n❌ Error: No images found in {RGB_DIR}")
        return
    
    # Sort by name (natural sort if possible)
    image_files.sort()
    
    print(f"\nFound {len(image_files)} images")
    print(f"Will rename to: {str(0).zfill(PADDING)}{OUTPUT_EXTENSION}, "
          f"{str(1).zfill(PADDING)}{OUTPUT_EXTENSION}, ...")
    
    # Create temporary directory for staging
    temp_dir = RGB_DIR / "_temp_rename"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\nStep 1: Moving images to temporary directory...")
    # Move all files to temp directory first (avoid conflicts)
    temp_files = []
    for img_path in image_files:
        temp_path = temp_dir / img_path.name
        shutil.move(str(img_path), str(temp_path))
        temp_files.append(temp_path)
    
    print(f"Step 2: Renaming and moving back...")
    # Rename and move back with sequential names
    for idx, temp_path in enumerate(temp_files):
        new_name = f"{str(idx).zfill(PADDING)}{OUTPUT_EXTENSION}"
        new_path = RGB_DIR / new_name
        shutil.move(str(temp_path), str(new_path))
        
        if idx < 5 or idx >= len(temp_files) - 3:
            print(f"  {temp_path.name:30s} → {new_name}")
        elif idx == 5:
            print(f"  ... ({len(temp_files) - 8} more) ...")
    
    # Remove temporary directory
    temp_dir.rmdir()
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"✓ Renamed {len(temp_files)} images")
    print(f"✓ Format: {str(0).zfill(PADDING)}{OUTPUT_EXTENSION} to "
          f"{str(len(temp_files)-1).zfill(PADDING)}{OUTPUT_EXTENSION}")
    print("="*70)


if __name__ == "__main__":
    rename_images_sequential()
