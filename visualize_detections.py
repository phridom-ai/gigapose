#!/usr/bin/env python3
"""
Visualize detection bounding boxes and segmentation masks.
Helps verify that detections align correctly with images.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from pycocotools import mask as mask_utils

# ==================== CONFIGURATION ====================

# Paths
DETECTION_FILE = Path("gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/measuring_tape_undistorted/measuring_tape_undistorted-test.json")
RGB_DIR = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/test/000000/rgb")
OUTPUT_DIR = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/test/000000/detection_visualizations")

# Visualization settings
BBOX_COLOR = (0, 255, 0)      # Green for bounding box
BBOX_THICKNESS = 2
MASK_COLOR = (255, 0, 0)      # Red for mask
MASK_ALPHA = 0.4              # Transparency for mask overlay
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND = (0, 0, 0)   # Black background for text

# How many images to visualize (None = all)
MAX_IMAGES = None  # Set to a number like 10 to limit output

# ==================== MAIN SCRIPT ====================

def draw_detection_on_image(image, detection):
    """Draw bounding box and mask on image."""
    
    # Create a copy to draw on
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Get bbox and convert to integers
    bbox = detection['bbox']  # [x, y, width, height]
    x, y, width, height = [int(v) for v in bbox]
    
    # Draw bounding box
    cv2.rectangle(vis_image, (x, y), (x + width, y + height), BBOX_COLOR, BBOX_THICKNESS)
    
    # Draw segmentation mask if available
    if 'segmentation' in detection and detection['segmentation']:
        try:
            # Decode RLE mask
            if isinstance(detection['segmentation'], dict):
                # Already in RLE format
                rle = detection['segmentation']
            else:
                # Convert from list format
                rle = {
                    'size': detection['segmentation']['size'],
                    'counts': detection['segmentation']['counts']
                }
            
            # Decode mask
            mask = mask_utils.decode(rle)
            
            # Create colored mask overlay
            mask_overlay = vis_image.copy()
            mask_overlay[mask > 0] = MASK_COLOR
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1 - MASK_ALPHA, mask_overlay, MASK_ALPHA, 0)
            
        except Exception as e:
            print(f"  ⚠️  Could not decode mask: {e}")
    
    # Add text with detection info
    score = detection.get('score', 0.0)
    obj_id = detection.get('category_id', detection.get('obj_id', '?'))
    text = f"Obj:{obj_id} Score:{score:.3f}"
    
    # Get text size for background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Draw text background
    cv2.rectangle(vis_image, (x, y - text_h - 10), (x + text_w + 10, y), TEXT_BACKGROUND, -1)
    
    # Draw text
    cv2.putText(vis_image, text, (x + 5, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
    return vis_image


def visualize_detections():
    """Main function to visualize all detections."""
    
    print("="*70)
    print("Detection Visualization Script")
    print("="*70)
    print(f"\nDetection file: {DETECTION_FILE}")
    print(f"Image directory: {RGB_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check files exist
    if not DETECTION_FILE.exists():
        print(f"\n❌ Error: Detection file not found: {DETECTION_FILE}")
        return
    
    if not RGB_DIR.exists():
        print(f"\n❌ Error: RGB directory not found: {RGB_DIR}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load detections
    print("\nLoading detections...")
    with open(DETECTION_FILE, 'r') as f:
        detections = json.load(f)
    
    print(f"Loaded {len(detections)} detections")
    
    # Group detections by image
    detections_by_image = {}
    for det in detections:
        scene_id = det['scene_id']
        image_id = det['image_id']
        
        # Create image key
        img_key = f"{scene_id:06d}_{image_id:06d}"
        
        if img_key not in detections_by_image:
            detections_by_image[img_key] = []
        detections_by_image[img_key].append(det)
    
    print(f"Detections span {len(detections_by_image)} unique images")
    
    # Process each image
    print(f"\nVisualizing detections...")
    processed = 0
    
    for img_key, dets in sorted(detections_by_image.items()):
        if MAX_IMAGES is not None and processed >= MAX_IMAGES:
            print(f"\nReached limit of {MAX_IMAGES} images, stopping.")
            break
        
        # Parse image key
        scene_id_str, image_id_str = img_key.split('_')
        image_id = int(image_id_str)
        
        # Load image
        img_path = RGB_DIR / f"{image_id:06d}.jpg"
        if not img_path.exists():
            img_path = RGB_DIR / f"{image_id:06d}.png"
        
        if not img_path.exists():
            print(f"  ⚠️  Image not found: {img_path.name}, skipping")
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not read image: {img_path.name}, skipping")
            continue
        
        # Draw all detections for this image
        vis_image = image.copy()
        for det in dets:
            vis_image = draw_detection_on_image(vis_image, det)
        
        # Save visualization
        output_path = OUTPUT_DIR / f"{image_id:06d}_detections.jpg"
        cv2.imwrite(str(output_path), vis_image)
        
        processed += 1
        print(f"  [{processed:3d}] {img_path.name} → {output_path.name} ({len(dets)} detection(s))")
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"✓ Visualized {processed} images")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print("\n💡 Next steps:")
    print(f"  1. Check the images in {OUTPUT_DIR}")
    print(f"  2. Verify bounding boxes (green) align with objects")
    print(f"  3. Verify masks (red overlay) cover the objects correctly")
    print(f"  4. If detections look wrong, you need to regenerate them")
    print("="*70)


if __name__ == "__main__":
    try:
        visualize_detections()
    except ImportError as e:
        if 'pycocotools' in str(e):
            print("\n❌ Error: pycocotools not installed")
            print("\nInstall with:")
            print("  pip install pycocotools")
        else:
            raise
