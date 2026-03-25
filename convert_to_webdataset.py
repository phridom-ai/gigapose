#!/usr/bin/env python3
"""
Convert BOP format to WebDataset format for GigaPose.
Creates .tar shards and key_to_shard.json mapping.
"""

import json
import tarfile
from pathlib import Path
import io

# Paths
test_dir = Path("gigaPose_datasets/datasets/measuring_tape_undistorted/test")
scene_dir = test_dir / "000000"
rgb_dir = scene_dir / "rgb"
output_dir = test_dir

# Load scene_camera.json
with open(scene_dir / "scene_camera.json", 'r') as f:
    scene_camera = json.load(f)

# Create a single tar shard with all images
shard_path = output_dir / "shard-000000.tar"
key_to_shard = {}

print(f"Creating WebDataset shard: {shard_path}")

# Remove old tar if exists
if shard_path.exists():
    shard_path.unlink()

with tarfile.open(shard_path, 'w') as tar:
    for img_id in range(60):
        img_id_str = f"{img_id:06d}"
        image_key = f"000000_{img_id_str}"
        
        # Add RGB image
        rgb_path = rgb_dir / f"{img_id_str}.jpg"
        if rgb_path.exists():
            # Read RGB data
            with open(rgb_path, 'rb') as f:
                rgb_data = f.read()
            
            # Add RGB to tar
            rgb_info = tarfile.TarInfo(name=f"{image_key}.rgb.jpg")
            rgb_info.size = len(rgb_data)
            tar.addfile(rgb_info, fileobj=io.BytesIO(rgb_data))
            
            # Add camera info
            camera_data = scene_camera[str(img_id)]
            camera_json = json.dumps(camera_data).encode('utf-8')
            camera_info = tarfile.TarInfo(name=f"{image_key}.camera.json")
            camera_info.size = len(camera_json)
            tar.addfile(camera_info, fileobj=io.BytesIO(camera_json))
            
            # Map this key to the shard
            key_to_shard[image_key] = "shard-000000.tar"
            
            if img_id % 10 == 0:
                print(f"  Added {image_key}")

# Save key_to_shard mapping
key_to_shard_path = output_dir / "key_to_shard.json"
with open(key_to_shard_path, 'w') as f:
    json.dump(key_to_shard, f, indent=2)

print(f"\n{'='*60}")
print(f"SUCCESS!")
print(f"{'='*60}")
print(f"Created WebDataset files:")
print(f"  {shard_path}")
print(f"  {key_to_shard_path}")
print(f"\nProcessed {len(key_to_shard)} images")
print(f"\nYou can now run:")
print(f"  python test.py test_dataset_name=my_objects run_id=exp1 test_setting=detection")
