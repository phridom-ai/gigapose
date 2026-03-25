#!/usr/bin/env python3
"""
Convert object_0.json annotations to GigaPose detection format.
Uses undistorted_cropped annotations for undistorted images.
"""
import json
import numpy as np
import os

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("WARNING: pycocotools not found. Install with: pip install pycocotools")
    HAS_PYCOCOTOOLS = False

# ==================== CONFIGURATION ====================
USE_UNDISTORTED = True  # Use undistorted_cropped data
CAMERA_IDX_FILTER = 0
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1440
# =======================================================

# Load your annotations
print("="*70)
print("Annotation to Detection Converter")
print(f"Using: {'undistorted_cropped' if USE_UNDISTORTED else 'distorted'} annotations")
print("="*70)
print("\nLoading annotations from object_0.json...")
with open('gigaPose_datasets/datasets/my_objects/test/object_0.json', 'r') as f:
    data = json.load(f)

print(f"Found {len(data['frames'])} total frames")

# Output detections as a FLAT LIST (not dict!)
detections = []

# Process only frames with camera_idx=0 (first 60 images)
processed_count = 0
for frame in data['frames']:
    if processed_count >= 60:
        break
        
    frame_idx = frame['frame_idx']
    found_camera_0 = False
    
    for inst in frame['instances']:
        # Only process detections for camera_idx 0
        if inst['camera_idx'] != CAMERA_IDX_FILTER:
            continue
        
        found_camera_0 = True
        
        # Get annotation data (undistorted_cropped or distorted)
        if USE_UNDISTORTED and 'undistorted_cropped' in inst:
            annotation_data = inst['undistorted_cropped']
        elif 'distorted' in inst:
            annotation_data = inst['distorted']
        else:
            annotation_data = inst
        
        # Get bbox and segmentation
        bbox_original = annotation_data.get('bbox', inst.get('bbox'))
        segmentation = annotation_data.get('segmentation', inst.get('segmentation'))
        
        # Convert bbox: [x1, y1, x2, y2] -> [x, y, width, height]
        x1, y1, x2, y2 = bbox_original
        bbox = [x1, y1, x2 - x1, y2 - y1]
        
        # Convert segmentation: list RLE -> string RLE (COCO format)
        h, w = segmentation['size']
        counts = segmentation['counts']
        
        # Decode RLE list to binary mask
        binary_mask = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        flag = 0  # Start with 0
        for count in counts:
            binary_mask[idx:idx+count] = flag
            idx += count
            flag = 1 - flag
        binary_mask = binary_mask.reshape((h, w), order='F')  # Column-major (Fortran order)
        
        if HAS_PYCOCOTOOLS:
            # Re-encode to COCO RLE format (with full image size)
            # Image dimensions: width=1440, height=1080
            full_mask = np.zeros((1080, 1440), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = binary_mask
            rle = mask_utils.encode(np.asfortranarray(full_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
        else:
            # Fallback: use the cropped RLE but convert list to fake string
            # This might not work perfectly but allows testing
            rle = {
                'size': [1080, 1440],
                'counts': str(counts)  # Simple conversion
            }
        
        detection = {
            "scene_id": 0,
            "image_id": processed_count,  # Sequential 0-59
            "category_id": inst['object_idx'] + 1,  # Map object_idx to category_id
            "bbox": bbox,
            "score": 1.0,  # No score in your data, using 1.0
            "time": 0.0,
            "segmentation": rle
        }
        detections.append(detection)
    
    if found_camera_0:
        if processed_count % 10 == 0:
            print(f"Processed image {processed_count} (frame_idx={frame_idx})...")
        processed_count += 1

print(f"Total detections: {len(detections)} for {processed_count} images")

# Save as a FLAT LIST
output_file = "gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/my_objects/my_objects-test.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(detections, f, indent=2)

print(f"\n{'='*60}")
print(f"SUCCESS!")
print(f"{'='*60}")
print(f"Saved {len(detections)} detections to:")
print(f"  {output_file}")
print(f"\nNext steps:")
print(f"  1. Run: python test.py test_dataset_name=my_objects run_id=exp1 test_setting=detection")
