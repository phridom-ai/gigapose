#!/usr/bin/env python3
"""
Visualize predicted object poses from a BOP-format CSV in Rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


GLB_EXPORT_VERSION = 1
GT_ALBEDO_FACTOR = [80, 230, 80, 255]
PRED_ALBEDO_FACTOR = [255, 140, 60, 255]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BOP-format pose predictions in Rerun."
    )
    parser.add_argument("--csv", required=True, help="Path to the prediction CSV.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing models/ and the selected split.",
    )
    parser.add_argument("--split", default="test", help="Split name inside dataset-root.")
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Optional scene id filter. Required if the CSV contains multiple scenes.",
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
        help="Optional GLB cache directory. Defaults to <dataset-root>/_rerun_glb_cache.",
    )
    parser.add_argument(
        "--mesh-units",
        choices=("mm", "m"),
        default="mm",
        help="Units used by meshes in models/. GLBs are exported in meters.",
    )
    parser.add_argument(
        "--recenter-ply",
        action="store_true",
        help="Create a centered copy of each PLY before GLB export.",
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


def require_rerun():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'rerun-sdk'. Install it in your environment to use this viewer."
        ) from exc
    return rr, rrb


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'trimesh'. Install it in your environment to convert PLY meshes to GLB."
        ) from exc
    return trimesh


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _ensure_centered_ply(mesh_path: Path) -> Path:
    if mesh_path.suffix.lower() != ".ply":
        return mesh_path
    if mesh_path.stem.endswith("_centered"):
        return mesh_path

    trimesh = require_trimesh()
    output_path = mesh_path.with_name(f"{mesh_path.stem}_centered{mesh_path.suffix}")
    try:
        if (
            output_path.exists()
            and output_path.stat().st_mtime_ns >= mesh_path.stat().st_mtime_ns
        ):
            return output_path
    except FileNotFoundError:
        pass

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"PLY mesh contains no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    bbox_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bbox_center)
    mesh.export(output_path)
    return output_path


def load_mesh(mesh_path: Path):
    trimesh = require_trimesh()
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Mesh contains no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)!r}")
    return mesh


def infer_scale_to_meters(mesh_units: str) -> float:
    if mesh_units == "m":
        return 1.0
    if mesh_units == "mm":
        return 0.001
    raise ValueError(f"Unsupported mesh_units={mesh_units!r}")


def find_model_path(model_dir: Path, obj_id: int) -> Path:
    base_name = f"obj_{obj_id:06d}"
    for suffix in (".ply", ".obj"):
        candidate = model_dir / f"{base_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing object mesh for obj_id={obj_id}: expected {base_name}.ply/.obj in {model_dir}")


def ensure_glb_cache_entry(
    model_dir: Path,
    glb_cache_dir: Path,
    obj_id: int,
    mesh_units: str,
    recenter_ply: bool,
) -> Path:
    source_path = find_model_path(model_dir, obj_id)
    mesh_path = _ensure_centered_ply(source_path) if recenter_ply else source_path
    glb_path = glb_cache_dir / f"obj_{obj_id:06d}.v{GLB_EXPORT_VERSION}.glb"

    source_mtime = mesh_path.stat().st_mtime_ns
    if glb_path.exists() and glb_path.stat().st_mtime_ns >= source_mtime:
        return glb_path

    mesh = load_mesh(mesh_path).copy()
    scale = infer_scale_to_meters(mesh_units)
    if scale != 1.0:
        mesh.apply_scale(scale)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(mesh.export(file_type="glb"))
    return glb_path


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


def load_predictions(csv_path: Path) -> list[dict]:
    predictions = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(parse_prediction_row(row))
    if not predictions:
        raise ValueError(f"No predictions found in {csv_path}")
    return predictions


def infer_scene_id(predictions: list[dict], requested_scene_id: int | None) -> int:
    scene_ids = sorted({pred["scene_id"] for pred in predictions})
    if requested_scene_id is not None:
        if requested_scene_id not in scene_ids:
            raise ValueError(
                f"Requested scene_id={requested_scene_id} is not present in CSV. Available scene ids: {scene_ids}"
            )
        return requested_scene_id
    if len(scene_ids) != 1:
        raise ValueError(f"CSV contains multiple scene ids {scene_ids}; pass --scene-id explicitly.")
    return scene_ids[0]


def find_rgb_path(scene_dir: Path, im_id: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = scene_dir / "rgb" / f"{im_id:06d}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find RGB image for im_id={im_id:06d} in {scene_dir / 'rgb'}")


def main() -> None:
    args = parse_args()
    rr, rrb = require_rerun()

    dataset_root = Path(args.dataset_root)
    csv_path = Path(args.csv)
    model_dir = dataset_root / "models"
    glb_cache_dir = Path(args.glb_cache_dir) if args.glb_cache_dir else dataset_root / "_rerun_glb_cache"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {model_dir}")

    predictions = load_predictions(csv_path)
    scene_id = infer_scene_id(predictions, args.scene_id)
    scene_dir = dataset_root / args.split / f"{scene_id:06d}"
    scene_camera = load_json(scene_dir / "scene_camera.json")
    scene_gt_path = scene_dir / "scene_gt.json"
    scene_gt = load_json(scene_gt_path) if scene_gt_path.exists() else {}

    predictions = [pred for pred in predictions if pred["scene_id"] == scene_id]
    predictions_by_image = defaultdict(list)
    object_ids = set()
    for pred in predictions:
        predictions_by_image[pred["im_id"]].append(pred)
        object_ids.add(pred["obj_id"])
    for gt_entries in scene_gt.values():
        for gt_entry in gt_entries:
            object_ids.add(int(gt_entry["obj_id"]))

    for obj_id in sorted(object_ids):
        ensure_glb_cache_entry(
            model_dir=model_dir,
            glb_cache_dir=glb_cache_dir,
            obj_id=obj_id,
            mesh_units=args.mesh_units,
            recenter_ply=args.recenter_ply,
        )

    image_ids = sorted(predictions_by_image.keys())
    if args.im_id is not None:
        image_ids = [im_id for im_id in image_ids if im_id == args.im_id]
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    if not image_ids:
        raise ValueError("No images selected for visualization.")

    rr.init("gigapose_predictions", spawn=not args.no_spawn)
    if args.save_rrd is not None:
        if not hasattr(rr, "save"):
            raise RuntimeError("This rerun installation does not expose rr.save for recording export.")
        rr.save(str(args.save_rrd))

    if hasattr(rr, "ViewCoordinates") and hasattr(rr.ViewCoordinates, "RDF"):
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    logged_assets = set()
    processed_images = 0

    for im_id in image_ids:
        rr.set_time("frame", sequence=im_id)
        # if hasattr(rr, "Clear"):
        #     rr.log("world/gt", rr.Clear(recursive=True))
        #     rr.log("world/pred", rr.Clear(recursive=True))

        rgb_path = find_rgb_path(scene_dir, im_id)
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {rgb_path}")

        cam_info = scene_camera[str(im_id)]
        K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
        height, width = image_bgr.shape[:2]

        rr.log(
            "world/camera",
            rr.Pinhole(image_from_camera=K.tolist(), resolution=[width, height]),
            static=True,
        )
        rr.log(
            "world/camera/rgb",
            rr.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
        )

        per_object_counts = defaultdict(int)
        frame_gt_entries = sorted(
            scene_gt.get(str(im_id), []),
            key=lambda gt_entry: int(gt_entry["obj_id"]),
        )
        frame_predictions = sorted(
            predictions_by_image[im_id],
            key=lambda pred: (pred["obj_id"], -pred["score"]),
        )

        for gt_entry in frame_gt_entries:
            obj_id = int(gt_entry["obj_id"])
            inst_idx = per_object_counts[("gt", obj_id)]
            per_object_counts[("gt", obj_id)] += 1

            entity_path = f"world/gt/obj_{obj_id:06d}/inst_{inst_idx:02d}"
            glb_path = glb_cache_dir / f"obj_{obj_id:06d}.v{GLB_EXPORT_VERSION}.glb"

            if entity_path not in logged_assets:
                rr.log(
                    entity_path,
                    rr.Asset3D(path=str(glb_path), albedo_factor=PRED_ALBEDO_FACTOR),
                    static=True,
                )
                logged_assets.add(entity_path)

            rr.log(
                entity_path,
                rr.Transform3D(
                    translation=(np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32) * 0.001).tolist(),
                    mat3x3=np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3).tolist(),
                ),
            )

        for pred in frame_predictions:
            obj_id = pred["obj_id"]
            inst_idx = per_object_counts[("pred", obj_id)]
            per_object_counts[("pred", obj_id)] += 1

            entity_path = f"world/pred/obj_{obj_id:06d}/inst_{inst_idx:02d}"
            glb_path = glb_cache_dir / f"obj_{obj_id:06d}.v{GLB_EXPORT_VERSION}.glb"

            if entity_path not in logged_assets:
                rr.log(
                    entity_path,
                    rr.Asset3D(path=str(glb_path), albedo_factor=GT_ALBEDO_FACTOR),
                    static=True,
                )
                logged_assets.add(entity_path)

            rr.log(
                entity_path,
                rr.Transform3D(
                    translation=(pred["t"] * 0.001).tolist(),
                    mat3x3=pred["R"].tolist(),
                ),
            )

        processed_images += 1

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="3D Scene",
                contents=["/**"],
            ),
            rrb.Spatial2DView(
                origin="world/camera",
                name="Camera",
                contents=["/**"],
            ),
        )
    )
    if hasattr(rr, "send_blueprint"):
        rr.send_blueprint(blueprint)

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "scene_dir": str(scene_dir),
                "csv_path": str(csv_path),
                "glb_cache_dir": str(glb_cache_dir),
                "scene_id": scene_id,
                "num_images": processed_images,
                "num_prediction_rows": len(predictions),
                "num_gt_frames": len(scene_gt),
                "num_objects": len(object_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
