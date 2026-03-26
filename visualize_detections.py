#!/usr/bin/env python3
"""
Visualize detection bounding boxes and segmentation masks for a custom dataset.
"""

import argparse
import json
from pathlib import Path

import cv2
from pycocotools import mask as mask_utils


BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
MASK_COLOR = (255, 0, 0)
MASK_ALPHA = 0.4
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def find_image_path(rgb_dir, image_id):
    stem = f"{int(image_id):06d}"
    for suffix in IMAGE_SUFFIXES:
        path = rgb_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image {stem} in {rgb_dir}")


def decode_segmentation(segmentation):
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            height, width = segmentation["size"]
            segmentation = mask_utils.frPyObjects(segmentation, height, width)
    mask = mask_utils.decode(segmentation)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def draw_detection_on_image(image, detection):
    vis_image = image.copy()
    bbox = detection["bbox"]
    x, y, width, height = [int(v) for v in bbox]
    cv2.rectangle(
        vis_image,
        (x, y),
        (x + width, y + height),
        BBOX_COLOR,
        BBOX_THICKNESS,
    )

    if detection.get("segmentation"):
        mask = decode_segmentation(detection["segmentation"])
        mask_overlay = vis_image.copy()
        mask_overlay[mask > 0] = MASK_COLOR
        vis_image = cv2.addWeighted(
            vis_image, 1 - MASK_ALPHA, mask_overlay, MASK_ALPHA, 0
        )

    score = detection.get("score", 0.0)
    obj_id = detection.get("category_id", detection.get("obj_id", "?"))
    text = f"Obj:{obj_id} Score:{score:.3f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_top = max(0, y - text_h - 10)
    cv2.rectangle(
        vis_image,
        (x, text_top),
        (x + text_w + 10, y),
        TEXT_BACKGROUND,
        -1,
    )
    cv2.putText(
        vis_image,
        text,
        (x + 5, max(text_h, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
    )
    return vis_image


def default_detection_file(dataset_root):
    dataset_root = Path(dataset_root)
    dataset_name = dataset_root.name
    return (
        dataset_root.parent
        / "default_detections"
        / "core19_model_based_unseen"
        / dataset_name
        / f"{dataset_name}-test.json"
    )


def visualize_detections(dataset_root, detection_file=None, max_images=None):
    dataset_root = Path(dataset_root)
    detection_file = Path(detection_file) if detection_file else default_detection_file(dataset_root)

    detections = json.loads(detection_file.read_text())
    detections_by_image = {}
    for det in detections:
        key = (int(det["scene_id"]), int(det["image_id"]))
        detections_by_image.setdefault(key, []).append(det)

    processed = 0
    for (scene_id, image_id) in sorted(detections_by_image.keys()):
        if max_images is not None and processed >= max_images:
            break
        scene_dir = dataset_root / "test" / f"{scene_id:06d}"
        rgb_dir = scene_dir / "rgb"
        output_dir = scene_dir / "detection_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = find_image_path(rgb_dir, image_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image {image_path}")

        vis_image = image.copy()
        for det in detections_by_image[(scene_id, image_id)]:
            vis_image = draw_detection_on_image(vis_image, det)

        output_path = output_dir / f"{image_id:06d}_detections.jpg"
        cv2.imwrite(str(output_path), vis_image)
        processed += 1

    return {
        "detection_file": str(detection_file),
        "processed_images": processed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize detection overlays for a custom dataset")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root, e.g. gigaPose_datasets/datasets/my_dataset",
    )
    parser.add_argument(
        "--detection-file",
        help="Optional detection JSON; defaults to the dataset's custom default-detection path",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional cap on the number of images to visualize",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = visualize_detections(
        dataset_root=args.dataset_root,
        detection_file=args.detection_file,
        max_images=args.max_images,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
