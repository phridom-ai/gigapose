#!/usr/bin/env python3
"""
Visualize GT and predicted 6D poses in Rerun using cached GLB assets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


GT_ASSET_COLOR = [80, 230, 80, 255]
PRED_ASSET_COLOR = [255, 150, 60, 255]
GT_DRAW_COLOR = (60, 220, 60)
PRED_DRAW_COLOR = (40, 140, 255)
GLB_EXPORT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GT and predicted 6D poses for a BOP scene in Rerun."
    )
    parser.add_argument("--csv", required=True, help="Path to a BOP-format prediction CSV.")
    parser.add_argument(
        "--dataset-root",
        default="gigaPose_datasets/datasets/hope_val_000001",
        help="Dataset root containing models/ and the selected split.",
    )
    parser.add_argument("--split", default="test", help="Split name inside --dataset-root.")
    parser.add_argument(
        "--scene-id",
        type=int,
        default=1,
        help="Scene id inside the selected split.",
    )
    parser.add_argument(
        "--im-id",
        type=int,
        default=None,
        help="Optional single image id to visualize.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to visualize.",
    )
    parser.add_argument(
        "--glb-cache-dir",
        default=None,
        help="Directory for cached GLB meshes. Defaults to <dataset-root>/_rerun_glb_cache.",
    )
    parser.add_argument(
        "--save-rrd",
        default=None,
        help="Optional output path for a saved Rerun recording.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the Rerun viewer process.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def require_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'rerun-sdk'. Install it in your environment to use this viewer."
        ) from exc
    return rr


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'trimesh'. Install it in your environment to export GLB assets for Rerun."
        ) from exc
    return trimesh


def load_ply_mesh(ply_path: Path):
    trimesh = require_trimesh()
    mesh = trimesh.load(str(ply_path), force="mesh", process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"PLY mesh contains no geometry: {ply_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {ply_path}: {type(mesh)!r}")
    return mesh


def ensure_glb_cache_entry(model_dir: Path, glb_cache_dir: Path, obj_id: int) -> tuple[Path, np.ndarray]:
    model_name = f"obj_{obj_id:06d}"
    ply_path = model_dir / f"{model_name}.ply"
    glb_path = glb_cache_dir / f"{model_name}.v{GLB_EXPORT_VERSION}.glb"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing mesh for obj_id={obj_id}: {ply_path}")

    mesh_mm = load_ply_mesh(ply_path)
    vertices_mm = np.asarray(mesh_mm.vertices, dtype=np.float32)
    if not glb_path.exists() or glb_path.stat().st_mtime < ply_path.stat().st_mtime:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_m = mesh_mm.copy()
        mesh_m.apply_scale(0.001)
        glb_bytes = mesh_m.export(file_type="glb")
        glb_path.write_bytes(glb_bytes)

    return glb_path, vertices_mm


def parse_prediction_row(row: dict) -> dict:
    return {
        "scene_id": int(row["scene_id"]),
        "im_id": int(row["im_id"]),
        "obj_id": int(row["obj_id"]),
        "score": float(row["score"]),
        "R": np.asarray([float(value) for value in row["R"].split()], dtype=np.float32).reshape(3, 3),
        "t": np.asarray([float(value) for value in row["t"].split()], dtype=np.float32),
        "time": float(row["time"]),
    }


def load_predictions(csv_path: Path, scene_id: int) -> tuple[dict, list[str]]:
    predictions = {}
    warnings = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = parse_prediction_row(row)
            if pred["scene_id"] != scene_id:
                continue
            key = (pred["scene_id"], pred["im_id"], pred["obj_id"])
            previous = predictions.get(key)
            if previous is None or pred["score"] > previous["score"]:
                if previous is not None:
                    warnings.append(
                        f"Multiple predictions for scene={scene_id} im_id={pred['im_id']} obj_id={pred['obj_id']}; "
                        f"kept score {pred['score']:.4f} over {previous['score']:.4f}."
                    )
                predictions[key] = pred
    return predictions, warnings


def project_points(points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points_cam = (R @ points_3d.T).T + t.reshape(1, 3)
    valid_z = points_cam[:, 2] > 1e-6
    points_2d = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    if not np.any(valid_z):
        return points_2d
    projected = (K @ points_cam[valid_z].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    points_2d[valid_z] = projected
    return points_2d


def draw_bbox_3d(image: np.ndarray, vertices_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, color) -> None:
    min_coords = vertices_3d.min(axis=0)
    max_coords = vertices_3d.max(axis=0)
    corners_3d = np.asarray(
        [
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
        ],
        dtype=np.float32,
    )
    corners_2d = project_points(corners_3d, K, R, t)
    if not np.all(np.isfinite(corners_2d)):
        return
    corners_2d = corners_2d.astype(np.int32)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start_idx, end_idx in edges:
        cv2.line(image, tuple(corners_2d[start_idx]), tuple(corners_2d[end_idx]), color, 2)


def draw_axes(image: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, size_mm: float = 50.0) -> None:
    axes_3d = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [size_mm, 0.0, 0.0],
            [0.0, size_mm, 0.0],
            [0.0, 0.0, size_mm],
        ],
        dtype=np.float32,
    )
    axes_2d = project_points(axes_3d, K, R, t)
    if not np.all(np.isfinite(axes_2d)):
        return
    axes_2d = axes_2d.astype(np.int32)
    origin = tuple(axes_2d[0])
    cv2.line(image, origin, tuple(axes_2d[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(axes_2d[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(axes_2d[3]), (255, 0, 0), 2)


def find_rgb_path(scene_dir: Path, im_id: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = scene_dir / "rgb" / f"{im_id:06d}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find RGB image for im_id={im_id:06d} in {scene_dir / 'rgb'}")


def format_warning(message: str) -> str:
    return f"[warning] {message}"


def main() -> None:
    args = parse_args()
    rr = require_rerun()

    dataset_root = Path(args.dataset_root)
    csv_path = Path(args.csv)
    scene_dir = dataset_root / args.split / f"{args.scene_id:06d}"
    glb_cache_dir = Path(args.glb_cache_dir) if args.glb_cache_dir else dataset_root / "_rerun_glb_cache"
    model_dir = dataset_root / "models"

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    scene_camera = load_json(scene_dir / "scene_camera.json")
    scene_gt = load_json(scene_dir / "scene_gt.json")
    predictions, prediction_warnings = load_predictions(csv_path, scene_id=args.scene_id)

    image_ids = sorted(int(image_id) for image_id in scene_camera.keys())
    if args.im_id is not None:
        image_ids = [image_id for image_id in image_ids if image_id == args.im_id]
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    if not image_ids:
        raise ValueError("No images selected for visualization.")

    gt_object_ids = {
        int(entry["obj_id"])
        for im_id in image_ids
        for entry in scene_gt.get(str(im_id), [])
    }
    pred_object_ids = {
        int(pred["obj_id"])
        for (_, im_id, _), pred in predictions.items()
        if im_id in image_ids
    }
    object_ids = sorted(gt_object_ids | pred_object_ids)

    model_cache = {}
    for obj_id in object_ids:
        glb_path, vertices_mm = ensure_glb_cache_entry(model_dir, glb_cache_dir, obj_id)
        model_cache[obj_id] = {"glb_path": glb_path, "vertices_mm": vertices_mm}

    rr.init("gigapose_pose_comparison", spawn=not args.no_spawn)
    if args.save_rrd is not None:
        if not hasattr(rr, "save"):
            raise RuntimeError("This rerun installation does not expose rr.save for recording export.")
        rr.save(str(args.save_rrd))

    if hasattr(rr, "ViewCoordinates") and hasattr(rr.ViewCoordinates, "RDF"):
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for warning in prediction_warnings:
        print(format_warning(warning))

    unmatched_predictions = []
    processed_images = 0
    for im_id in image_ids:
        rr.set_time("frame", sequence=im_id)
        if hasattr(rr, "Clear"):
            rr.log("world/gt", rr.Clear(recursive=True))
            rr.log("world/pred", rr.Clear(recursive=True))

        rgb_path = find_rgb_path(scene_dir, im_id)
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {rgb_path}")
        overlay_bgr = image_bgr.copy()
        height, width = image_bgr.shape[:2]

        cam_info = scene_camera[str(im_id)]
        K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)

        translation_list = []
        mat3x3_list = []
        gt_entries = scene_gt.get(str(im_id), [])
        gt_counts = defaultdict(int)
        gt_obj_ids_in_frame = defaultdict(int)
        for gt_entry in gt_entries:
            obj_id = int(gt_entry["obj_id"])
            gt_instance_idx = gt_counts[obj_id]
            gt_counts[obj_id] += 1
            gt_obj_ids_in_frame[obj_id] += 1
            entity_path = f"world/gt/obj_{obj_id:06d}/inst_{gt_instance_idx:02d}"
            mesh_path = f"{entity_path}/mesh"

            translation_list.append((np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32) * 0.001).tolist())
            mat3x3_list.append(np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3).tolist())
            draw_bbox_3d(
                overlay_bgr,
                model_cache[obj_id]["vertices_mm"],
                K,
                np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32),
                GT_DRAW_COLOR,
            )
            draw_axes(
                overlay_bgr,
                K,
                np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32),
            )

        rr.log(
            "/world/gt/obj_000001/inst_00/mesh",
            rr.Asset3D(path="/home/luthov/hoi/gigapose_integration/gigapose/gigaPose_datasets/datasets/hope_val_000001/_rerun_glb_cache/obj_000001.v2.glb"),
            # rr.Asset3D(path=str(model_cache[obj_id]["glb_path"])),
            static=True,
        )
        rr.log(
            "/world/gt/obj_000001/inst_00",
            rr.Transform3D(
                translation=translation_list[0], # (np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32) * 0.001).tolist(),
                mat3x3=mat3x3_list[0], # np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3).tolist(),
            )
        )

        rr.log(
            "/world/gt/obj_000002/inst_00/mesh",
            rr.Asset3D(path="/home/luthov/hoi/gigapose_integration/gigapose/gigaPose_datasets/datasets/hope_val_000001/_rerun_glb_cache/obj_000002.v2.glb"),
            # rr.Asset3D(path=str(model_cache[obj_id]["glb_path"])),
            static=True,
        )
        rr.log(
            "/world/gt/obj_000002/inst_00",
            rr.Transform3D(
                translation=translation_list[1], # (np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32) * 0.001).tolist(),
                mat3x3=mat3x3_list[1], # np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3).tolist(),
            )
        )
        # rr.log(
        #     "world/camera",
        #     rr.Pinhole(
        #         image_from_camera=K.tolist(),
        #         resolution=[width, height],
        #     ),
        # )

        # # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # rr.log(
        #         "world/camera",
        #         rr.Transform3D(
        #             translation=[0.0, 0.0, 0.0],
        #             mat3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        #         ),
        #         static=True,
        # )
        
        # rr.log("world/camera/raw", rr.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))

    #     frame_predictions = [
    #         pred
    #         for (scene_id, pred_im_id, _), pred in predictions.items()
    #         if scene_id == args.scene_id and pred_im_id == im_id
    #     ]
    #     for pred in sorted(frame_predictions, key=lambda value: value["obj_id"]):
    #         obj_id = int(pred["obj_id"])
    #         entity_path = f"world/pred/obj_{obj_id:06d}"
    #         mesh_path = f"{entity_path}/mesh"
    #         rr.log(
    #             mesh_path,
    #             rr.Asset3D(path=str(model_cache[obj_id]["glb_path"]), albedo_factor=PRED_ASSET_COLOR),
    #             static=True,
    #         )
    #         rr.log(
    #             entity_path,
    #             rr.Transform3D(
    #                 translation=(pred["t"] * 0.001).tolist(),
    #                 mat3x3=pred["R"].tolist(),
    #             ),
    #         )
    #         draw_bbox_3d(
    #             overlay_bgr,
    #             model_cache[obj_id]["vertices_mm"],
    #             K,
    #             pred["R"],
    #             pred["t"],
    #             PRED_DRAW_COLOR,
    #         )
    #         draw_axes(overlay_bgr, K, pred["R"], pred["t"])
    #         if gt_obj_ids_in_frame[obj_id] == 0:
    #             unmatched_predictions.append(
    #                 f"scene={args.scene_id} im_id={im_id} obj_id={obj_id} has no GT match in scene_gt.json."
    #             )

    #     rr.log("world/camera/overlay", rr.Image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)))
    #     processed_images += 1

    # for warning in unmatched_predictions:
    #     print(format_warning(warning))

    # print(
    #     json.dumps(
    #         {
    #             "dataset_root": str(dataset_root),
    #             "scene_dir": str(scene_dir),
    #             "csv_path": str(csv_path),
    #             "glb_cache_dir": str(glb_cache_dir),
    #             "num_images": processed_images,
    #             "num_prediction_rows": len(predictions),
    #             "num_unmatched_predictions": len(unmatched_predictions),
    #         },
    #         indent=2,
    #     )
    # )


if __name__ == "__main__":
    main()
