#!/usr/bin/env python3
"""
Fix COCO RLE encoding in detection JSON file.
Converts uncompressed RLE (list of ints) to compressed COCO RLE format (string).
"""

import json
import numpy as np

# Try to import pycocotools
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("WARNING: pycocotools not available. Install it with:")
    print("  pip install pycocotools --break-system-packages")
    print("  or use a virtual environment")


def encode_mask_to_coco_rle(uncompressed_rle, size):
    """
    Convert uncompressed RLE (list of counts) to COCO compressed RLE format.
    
    The input RLE already represents a full-size mask, so we just need to:
    1. Decode RLE list to binary mask
    2. Encode using pycocotools to get compressed string format
    
    Args:
        uncompressed_rle: List of integers representing run lengths
        size: [height, width] of the mask (should be full image size)
    
    Returns:
        dict with 'counts' as compressed string and 'size' as list
    """
    if not HAS_PYCOCOTOOLS:
        raise ImportError("pycocotools is required for proper RLE encoding")
    
    # Decode RLE list to binary mask
    h, w = size
    binary_mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    flag = 0  # Start with 0 (background)
    
    for count in uncompressed_rle:
        binary_mask[idx:idx+count] = flag
        idx += count
        flag = 1 - flag  # Alternate between 0 and 1
    
    # Reshape to 2D mask (Column-major/Fortran order)
    binary_mask = binary_mask.reshape((h, w), order='F')
    
    # Encode to COCO RLE format
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    
    # Convert bytes to string
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def fix_detection_file(input_path, output_path=None):
    """
    Fix RLE encoding in detection JSON file.
    
    Converts uncompressed RLE (list) to compressed COCO RLE (string).
    The RLE already represents full-size masks.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (defaults to input_path if None)
    """
    if not HAS_PYCOCOTOOLS:
        print("\n❌ ERROR: pycocotools is required!")
        print("Install it with one of these methods:")
        print("  1. pip install pycocotools --break-system-packages")
        print("  2. Use a virtual environment")
        print("  3. apt install python3-pycocotools")
        return
    
    if output_path is None:
        output_path = input_path
    
    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        detections = json.load(f)
    
    print(f"Total detections: {len(detections)}")
    
    # Fix each detection's segmentation mask
    fixed_count = 0
    for i, det in enumerate(detections):
        if 'segmentation' in det:
            seg = det['segmentation']
            
            # Check if counts is a list (needs fixing)
            if isinstance(seg['counts'], list):
                # Convert uncompressed RLE to COCO RLE
                # The RLE already represents full-size mask
                coco_rle = encode_mask_to_coco_rle(seg['counts'], seg['size'])
                det['segmentation'] = coco_rle
                fixed_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Fixed {i + 1}/{len(detections)} detections...")
    
    print(f"\nFixed {fixed_count} segmentation masks")
    
    # Save fixed detections
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print("✓ Done!")
    
    # Verify the fix
    print("\n--- Verification ---")
    with open(output_path, 'r') as f:
        fixed_data = json.load(f)
    
    example_seg = fixed_data[0]['segmentation']
    print(f"Example segmentation format:")
    print(f"  - Has 'size': {('size' in example_seg)}")
    print(f"  - Has 'counts': {('counts' in example_seg)}")
    print(f"  - Counts type: {type(example_seg['counts'])}")
    print(f"  - Counts length: {len(example_seg['counts']) if isinstance(example_seg['counts'], str) else 'N/A'}")
    print(f"  - Size: {example_seg['size']}")
    
    if isinstance(example_seg['counts'], str):
        print(f"  - First 50 chars: {example_seg['counts'][:50]}...")
        print("\n✓ Format looks correct (COCO RLE string format)")
    else:
        print("\n✗ Warning: counts is still not a string!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix COCO RLE encoding in detection JSON")
    parser.add_argument(
        "--input",
        default="gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/my_objects/my_objects-test.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (defaults to overwriting input)"
    )
    
    args = parser.parse_args()
    
    fix_detection_file(args.input, args.output)
