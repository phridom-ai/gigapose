#!/usr/bin/env python3
"""
Visualize 6D pose estimation results by overlaying 3D object models onto input images.
"""

import os
import csv
import json
import pickle
import struct
import numpy as np
import cv2
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_ply_model(ply_path):
    """Load vertex xyz coordinates from an ASCII or binary little-endian PLY."""
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: {ply_path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        fmt = None
        vertex_count = None
        vertex_properties = []
        in_vertex_block = False

        for line in header_lines:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
                in_vertex_block = True
            elif line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex_block = False
            elif in_vertex_block and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("List properties are not supported in vertex elements")
                vertex_properties.append((parts[2], parts[1]))

        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"Unsupported PLY format: {fmt}")
        if vertex_count is None:
            raise ValueError(f"PLY file has no vertex element: {ply_path}")

        prop_names = [name for name, _ in vertex_properties]
        xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]

        if fmt == "ascii":
            vertices = np.empty((vertex_count, 3), dtype=np.float32)
            for idx in range(vertex_count):
                parts = f.readline().decode("ascii").strip().split()
                vertices[idx] = [float(parts[xyz_indices[0]]), float(parts[xyz_indices[1]]), float(parts[xyz_indices[2]])]
            return vertices

        type_map = {
            "char": "b",
            "uchar": "B",
            "short": "h",
            "ushort": "H",
            "int": "i",
            "uint": "I",
            "float": "f",
            "double": "d",
        }
        try:
            struct_fmt = "<" + "".join(type_map[prop_type] for _, prop_type in vertex_properties)
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY property type: {exc.args[0]}") from exc

        stride = struct.calcsize(struct_fmt)
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        for idx in range(vertex_count):
            row = struct.unpack(struct_fmt, f.read(stride))
            vertices[idx] = [row[xyz_indices[0]], row[xyz_indices[1]], row[xyz_indices[2]]]
        return vertices


def project_points(points_3d, K, R, t, distortion_coeffs=None):
    """
    Project 3D points to 2D image plane.
    
    Args:
        points_3d: (N, 3) array of 3D points
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector in mm
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
    """
    if distortion_coeffs is not None:
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
        points_2d, _ = cv2.projectPoints(
            np.asarray(points_3d, dtype=np.float64),
            rvec,
            tvec,
            np.asarray(K, dtype=np.float64),
            np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1),
        )
        return points_2d.reshape(-1, 2).astype(np.float32)

    # Transform points: X_cam = R * X_obj + t
    points_cam = (R @ points_3d.T).T + t

    # Avoid invalid divisions for points behind the camera.
    valid_z = points_cam[:, 2] > 1e-6
    points_2d = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    if not np.any(valid_z):
        return points_2d

    # Project to image plane
    projected = (K @ points_cam[valid_z].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    points_2d[valid_z] = projected

    return points_2d


def draw_wireframe(image, points_2d, color=(0, 255, 0), thickness=1):
    """
    Draw wireframe by connecting nearby points.
    
    Args:
        image: Input image
        points_2d: (N, 2) array of projected 2D points
        color: BGR color tuple
        thickness: Line thickness
    """
    # Filter points within image bounds
    h, w = image.shape[:2]
    valid_mask = (
        np.isfinite(points_2d[:, 0]) &
        np.isfinite(points_2d[:, 1]) &
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    valid_points = points_2d[valid_mask]
    
    # Draw points
    for pt in valid_points:
        cv2.circle(image, tuple(pt.astype(int)), 1, color, -1)
    
    return image


def draw_bbox_3d(image, vertices_3d, K, R, t, color=(0, 255, 0), thickness=2, distortion_coeffs=None):
    """
    Draw 3D bounding box of the object.
    
    Args:
        image: Input image
        vertices_3d: (N, 3) object vertices
        K: Camera intrinsic matrix
        R: Rotation matrix
        t: Translation vector
        color: BGR color
        thickness: Line thickness
    """
    # Compute 3D bounding box corners
    min_coords = vertices_3d.min(axis=0)
    max_coords = vertices_3d.max(axis=0)
    
    # 8 corners of the bounding box
    corners_3d = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
    ])
    
    # Project to 2D
    corners_2d = project_points(corners_3d, K, R, t, distortion_coeffs=distortion_coeffs)
    if not np.all(np.isfinite(corners_2d)):
        return image
    corners_2d = corners_2d.astype(int)
    
    # Draw the 12 edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    for start_idx, end_idx in edges:
        pt1 = tuple(corners_2d[start_idx])
        pt2 = tuple(corners_2d[end_idx])
        cv2.line(image, pt1, pt2, color, thickness)
    
    return image


def draw_axes(image, K, R, t, size=50, thickness=3, distortion_coeffs=None):
    """
    Draw coordinate axes at the object origin.
    
    Args:
        image: Input image
        K: Camera intrinsic matrix
        R: Rotation matrix
        t: Translation vector
        size: Length of axes in mm
        thickness: Line thickness
    """
    # Define axes endpoints
    axes_3d = np.array([
        [0, 0, 0],
        [size, 0, 0],  # X-axis (red)
        [0, size, 0],  # Y-axis (green)
        [0, 0, size],  # Z-axis (blue)
    ])
    
    # Project to 2D
    axes_2d = project_points(axes_3d, K, R, t, distortion_coeffs=distortion_coeffs)
    if not np.all(np.isfinite(axes_2d)):
        return image
    axes_2d = axes_2d.astype(int)
    
    # Draw axes
    origin = tuple(axes_2d[0])
    cv2.line(image, origin, tuple(axes_2d[1]), (0, 0, 255), thickness)  # X: red
    cv2.line(image, origin, tuple(axes_2d[2]), (0, 255, 0), thickness)  # Y: green
    cv2.line(image, origin, tuple(axes_2d[3]), (255, 0, 0), thickness)  # Z: blue
    
    return image


def draw_2d_bbox(image, bbox, color=(0, 255, 255), thickness=2, label=None):
    """Draw a 2D bbox in xywh format."""
    x, y, width, height = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
    if label:
        text_origin = (x, max(20, y - 8))
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def load_detection_bboxes(detections_json_path):
    """Load detection bboxes keyed by (scene_id, image_id, category_id)."""
    with open(detections_json_path, "r") as f:
        detections = json.load(f)

    bbox_by_key = {}
    for det in detections:
        key = (int(det["scene_id"]), int(det["image_id"]), int(det["category_id"]))
        bbox_by_key[key] = det["bbox"]
    return bbox_by_key


def load_calibration_entry(calibration_pkl_path, camera_serial):
    with open(calibration_pkl_path, "rb") as f:
        calibration = pickle.load(f)
    calibration = {str(key): value for key, value in calibration.items()}
    if camera_serial not in calibration:
        available = ", ".join(sorted(calibration.keys()))
        raise KeyError(
            f"Camera serial {camera_serial!r} not found in calibration. Available serials: {available}"
        )
    entry = calibration[camera_serial]
    K = np.asarray(entry["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(entry["D"], dtype=np.float64).reshape(-1)
    return K, D


def parse_csv_line(line):
    """Parse a line from the CSV results file."""
    scene_id = int(line['scene_id'])
    im_id = int(line['im_id'])
    obj_id = int(line['obj_id'])
    score = float(line['score'])
    
    # Parse rotation matrix (9 values)
    R_flat = [float(x) for x in line['R'].split()]
    R = np.array(R_flat).reshape(3, 3)
    
    # Parse translation vector (3 values)
    t_flat = [float(x) for x in line['t'].split()]
    t = np.array(t_flat)
    
    time = float(line['time'])
    
    return {
        'scene_id': scene_id,
        'im_id': im_id,
        'obj_id': obj_id,
        'score': score,
        'R': R,
        't': t,
        'time': time
    }


def visualize_poses(
    csv_path,
    images_dir,
    models_dir,
    camera_json_path,
    output_dir,
    detections_json_path=None,
    calibration_pkl_path=None,
    camera_serial=None,
    score_threshold=0.0,
    vis_type='bbox',  # 'bbox', 'wireframe', 'axes', 'all'
    max_images=None,
    draw_detection_bbox=True,
):
    """
    Visualize pose estimation results.
    
    Args:
        csv_path: Path to CSV file with pose predictions
        images_dir: Directory with RGB images
        models_dir: Directory with object 3D models (.ply)
        camera_json_path: Path to scene_camera.json with camera intrinsics
        output_dir: Output directory for visualizations
        score_threshold: Minimum score to visualize
        vis_type: Type of visualization ('bbox', 'wireframe', 'axes', 'all')
        max_images: Maximum number of images to process (None = all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera intrinsics
    print(f"Loading camera intrinsics from {camera_json_path}")
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)
    
    distortion_coeffs = None
    calibration_K = None
    if calibration_pkl_path:
        if not camera_serial:
            raise ValueError("--camera_serial is required when --calibration_pkl is provided")
        print(f"Loading distortion calibration from {calibration_pkl_path} for camera {camera_serial}")
        calibration_K, distortion_coeffs = load_calibration_entry(calibration_pkl_path, camera_serial)

    # Load object models
    print(f"Loading object models from {models_dir}")
    models = {}
    for model_file in Path(models_dir).glob("*.ply"):
        obj_id = int(model_file.stem.replace('obj_', ''))
        models[obj_id] = load_ply_model(str(model_file))
        print(f"  Loaded object {obj_id}: {len(models[obj_id])} vertices")

    detection_bboxes = {}
    if detections_json_path:
        print(f"Loading detection bounding boxes from {detections_json_path}")
        detection_bboxes = load_detection_bboxes(detections_json_path)
    
    # Read pose predictions
    print(f"Reading pose predictions from {csv_path}")
    predictions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            pred = parse_csv_line(line)
            if pred['score'] >= score_threshold:
                predictions.append(pred)
    
    print(f"Found {len(predictions)} predictions (score >= {score_threshold})")
    
    # Group predictions by image
    image_predictions = {}
    for pred in predictions:
        im_id = pred['im_id']
        if im_id not in image_predictions:
            image_predictions[im_id] = []
        image_predictions[im_id].append(pred)
    
    print(f"Processing {len(image_predictions)} images")
    
    # Process each image
    processed = 0
    for im_id in tqdm(sorted(image_predictions.keys())):
        if max_images and processed >= max_images:
            break
        
        # Load image
        image_path = os.path.join(images_dir, f"{im_id:06d}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir, f"{im_id:06d}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {im_id:06d} not found, skipping")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load {image_path}, skipping")
            continue
        
        # Get camera intrinsics for this image
        cam_info = camera_data[str(im_id)]
        K = np.array(cam_info['cam_K']).reshape(3, 3)
        if calibration_K is not None and not np.allclose(K, calibration_K, atol=1e-6):
            print(
                f"Warning: scene_camera K for image {im_id:06d} does not match calibration K. "
                "Using scene_camera K with calibration distortion coefficients."
            )
        
        # Overlay each prediction
        for pred_idx, pred in enumerate(image_predictions[im_id]):
            obj_id = pred['obj_id']
            R = pred['R']
            t = pred['t']
            score = pred['score']
            
            if obj_id not in models:
                print(f"Warning: Model for object {obj_id} not found, skipping")
                continue
            
            vertices = models[obj_id]
            
            # Choose color based on score (green = high, red = low)
            color_ratio = min(1.0, max(0.0, score))
            color = (0, int(255 * color_ratio), int(255 * (1 - color_ratio)))
            
            # Draw visualization
            if vis_type in ['bbox', 'all']:
                image = draw_bbox_3d(
                    image, vertices, K, R, t, color=color, thickness=2, distortion_coeffs=distortion_coeffs
                )
            
            if vis_type in ['axes', 'all']:
                image = draw_axes(image, K, R, t, size=50, thickness=3, distortion_coeffs=distortion_coeffs)
            
            if vis_type in ['wireframe', 'all']:
                points_2d = project_points(vertices[::10], K, R, t, distortion_coeffs=distortion_coeffs)  # Subsample for speed
                image = draw_wireframe(image, points_2d, color=color, thickness=1)

            if draw_detection_bbox:
                det_key = (pred['scene_id'], im_id, obj_id)
                det_bbox = detection_bboxes.get(det_key)
                if det_bbox is not None:
                    image = draw_2d_bbox(
                        image,
                        det_bbox,
                        color=(0, 255, 255),
                        thickness=2,
                        label=f"Det obj {obj_id}",
                    )
            
            # Draw score text
            text = f"Obj {obj_id}: {score:.3f}"
            cv2.putText(
                image,
                text,
                (10, 30 + pred_idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{im_id:06d}_pose.jpg")
        cv2.imwrite(output_path, image)
        
        processed += 1
    
    print(f"\n✓ Saved {processed} visualizations to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 6D pose estimation results")
    parser.add_argument(
        "--csv",
        default="gigaPose_datasets/results/large_cnos_exp_1/predictions/large-pbrreal-rgb-mmodel_my_objects-test_cnos_exp_1.csv",
        help="Path to CSV file with pose predictions"
    )
    parser.add_argument(
        "--images_dir",
        default="gigaPose_datasets/datasets/my_objects/test/000000/rgb",
        help="Directory with RGB images"
    )
    parser.add_argument(
        "--models_dir",
        default="gigaPose_datasets/datasets/my_objects/models",
        help="Directory with object 3D models (.ply)"
    )
    parser.add_argument(
        "--camera_json",
        default="gigaPose_datasets/datasets/my_objects/test/000000/scene_camera.json",
        help="Path to scene_camera.json"
    )
    parser.add_argument(
        "--output_dir",
        default="pose_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--calibration_pkl",
        default=None,
        help="Optional path to systemCalibration.pkl for distortion-aware projection"
    )
    parser.add_argument(
        "--camera_serial",
        default=None,
        help="Camera serial number to use with --calibration_pkl"
    )
    parser.add_argument(
        "--detections_json",
        default="gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/my_objects/my_objects-test.json",
        help="Path to detection json used for 2D bbox overlay"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum score to visualize (default: 0.0)"
    )
    parser.add_argument(
        "--vis_type",
        choices=['bbox', 'wireframe', 'axes', 'all'],
        default='all',
        help="Visualization type (default: all)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    parser.add_argument(
        "--no_detection_bbox",
        action="store_true",
        help="Disable 2D detection bbox overlay"
    )
    
    args = parser.parse_args()
    
    visualize_poses(
        csv_path=args.csv,
        images_dir=args.images_dir,
        models_dir=args.models_dir,
        camera_json_path=args.camera_json,
        output_dir=args.output_dir,
        detections_json_path=args.detections_json,
        calibration_pkl_path=args.calibration_pkl,
        camera_serial=args.camera_serial,
        score_threshold=args.score_threshold,
        vis_type=args.vis_type,
        max_images=args.max_images,
        draw_detection_bbox=not args.no_detection_bbox,
    )
