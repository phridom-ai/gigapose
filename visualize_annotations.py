#!/usr/bin/env python3
"""
Visualize original annotations from object_0.json.
Shows bounding boxes and segmentation masks from raw annotations.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from pycocotools import mask as mask_utils

# ==================== CONFIGURATION ====================

# Paths
ANNOTATION_FILE = Path("gigaPose_datasets/datasets/my_objects/test/object_0.json")
RGB_DIR = Path("gigaPose_datasets/datasets/my_objects/test/000000/rgb")
OUTPUT_DIR = Path("gigaPose_datasets/datasets/my_objects/test/000000/annotation_visualizations")

# Filter by camera index (set to None to visualize all cameras)
CAMERA_IDX_FILTER = 0  # Only show camera_idx=0

# Visualization settings
BBOX_COLOR = (0, 255, 255)    # Yellow for bounding box
BBOX_THICKNESS = 2
MASK_COLOR = (255, 0, 255)    # Magenta for mask
MASK_ALPHA = 0.4              # Transparency for mask overlay
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND = (0, 0, 0)   # Black background for text

# How many images to visualize (None = all)
MAX_IMAGES = None  # Set to a number like 10 to limit output

# ==================== MAIN SCRIPT ====================

def decode_rle_segmentation(segmentation, image_shape):
    """Decode RLE segmentation to binary mask."""
    try:
        # Check if it's already in COCO RLE format
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            return mask_utils.decode(segmentation)
        
        # If it's a list RLE format
        if isinstance(segmentation, list):
            # Create RLE dict
            rle = {
                'size': list(image_shape),
                'counts': segmentation
            }
            return mask_utils.decode(rle)
        
        return None
    except Exception as e:
        print(f"    ⚠️  Could not decode segmentation: {e}")
        return None


def draw_annotation_on_image(image, bbox, segmentation):
    """Draw bounding box and mask on image."""
    
    # Create a copy to draw on
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Get bbox - format is [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw bounding box
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
    
    # Draw segmentation mask if available
    if segmentation:
        mask = decode_rle_segmentation(segmentation, (h, w))
        
        if mask is not None:
            # Create colored mask overlay
            mask_overlay = vis_image.copy()
            mask_overlay[mask > 0] = MASK_COLOR
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1 - MASK_ALPHA, mask_overlay, MASK_ALPHA, 0)
    
    # Add text with bbox dimensions
    width = x2 - x1
    height = y2 - y1
    text = f"BBox: {width}x{height}px"
    
    # Get text size for background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Draw text background
    cv2.rectangle(vis_image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), TEXT_BACKGROUND, -1)
    
    # Draw text
    cv2.putText(vis_image, text, (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
    return vis_image


def visualize_annotations():
    """Main function to visualize all annotations."""
    
    print("="*70)
    print("Original Annotation Visualization Script")
    print("="*70)
    print(f"\nAnnotation file: {ANNOTATION_FILE}")
    print(f"Image directory: {RGB_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if CAMERA_IDX_FILTER is not None:
        print(f"Filter: camera_idx={CAMERA_IDX_FILTER}")
    
    # Check files exist
    if not ANNOTATION_FILE.exists():
        print(f"\n❌ Error: Annotation file not found: {ANNOTATION_FILE}")
        return
    
    if not RGB_DIR.exists():
        print(f"\n❌ Error: RGB directory not found: {RGB_DIR}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("\nLoading annotations...")
    with open(ANNOTATION_FILE, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Loaded {len(frames)} frames")
    
    # Process frames
    print(f"\nVisualizing annotations...")
    processed = 0
    skipped = 0
    
    for frame in frames:
        if MAX_IMAGES is not None and processed >= MAX_IMAGES:
            print(f"\nReached limit of {MAX_IMAGES} images, stopping.")
            break
        
        frame_idx = frame['frame_idx']
        instances = frame['instances']
        
        # Filter instances by camera_idx if specified
        if CAMERA_IDX_FILTER is not None:
            instances = [inst for inst in instances if inst.get('camera_idx') == CAMERA_IDX_FILTER]
        
        if not instances:
            skipped += 1
            continue
        
        # Load image
        img_path = RGB_DIR / f"{frame_idx:06d}.jpg"
        if not img_path.exists():
            img_path = RGB_DIR / f"{frame_idx:06d}.png"
        
        if not img_path.exists():
            print(f"  ⚠️  Image not found: {frame_idx:06d}.jpg, skipping")
            skipped += 1
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not read image: {img_path.name}, skipping")
            skipped += 1
            continue
        
        # Draw all instances for this frame
        vis_image = image.copy()
        for inst in instances:
            bbox = inst['bbox']
            segmentation = inst.get('segmentation', None)
            vis_image = draw_annotation_on_image(vis_image, bbox, segmentation)
        
        # Save visualization
        output_path = OUTPUT_DIR / f"{frame_idx:06d}_annotations.jpg"
        cv2.imwrite(str(output_path), vis_image)
        
        processed += 1
        print(f"  [{processed:3d}] Frame {frame_idx:06d} → {output_path.name} ({len(instances)} annotation(s))")
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"✓ Visualized {processed} images")
    if skipped > 0:
        print(f"  (Skipped {skipped} frames - no matching camera_idx or missing images)")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print("\n💡 Compare with detection visualizations:")
    print(f"  Original annotations: {OUTPUT_DIR}")
    print(f"  Detection file:       gigaPose_datasets/datasets/my_objects/test/000000/detection_visualizations")
    print("\n  - Yellow bbox + Magenta mask = Original annotations (object_0.json)")
    print("  - Green bbox + Red mask = Detection file (my_objects-test.json)")
    print("\n  If they don't match, your detections are out of sync with the images!")
    print("="*70)


if __name__ == "__main__":
    try:
        visualize_annotations()
    except ImportError as e:
        if 'pycocotools' in str(e):
            print("\n❌ Error: pycocotools not installed")
            print("\nInstall with:")
            print("  pip install pycocotools")
        else:
            raise
