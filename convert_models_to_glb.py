#!/usr/bin/env python3
"""
Convert object models in a dataset models directory to GLB files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def log_step(message: str) -> None:
    print(message, flush=True)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


def _ensure_centered_ply(mesh_path: Path) -> Path:
    if mesh_path.suffix.lower() != ".ply":
        return mesh_path
    if mesh_path.stem.endswith("_centered"):
        return mesh_path

    try:
        import trimesh
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"PLY recentering requires trimesh for {mesh_path}: {exc}"
        ) from exc

    output_path = mesh_path.with_name(f"{mesh_path.stem}_centered{mesh_path.suffix}")
    try:
        if (
            output_path.exists()
            and output_path.stat().st_mtime_ns >= mesh_path.stat().st_mtime_ns
        ):
            return output_path
    except Exception:  # noqa: BLE001
        pass

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError("mesh contains no geometry")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        bbox_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-bbox_center)
        mesh.export(output_path)
        log_step(
            "Prepared centered mesh: "
            f"source={_display_path(mesh_path)} output={_display_path(output_path)}"
        )
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to recenter PLY mesh {mesh_path}: {exc}") from exc


def load_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Mesh contains no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)!r}")
    return mesh


def infer_scale_to_meters(mesh_path: Path, mesh_units: str) -> float:
    if mesh_units == "m":
        return 1.0
    if mesh_units == "mm":
        return 0.001
    raise ValueError(f"Unsupported mesh_units={mesh_units!r}, expected 'mm' or 'm'")


def convert_mesh_to_glb(
    mesh_path: Path,
    output_dir: Path,
    mesh_units: str,
    recenter_ply: bool,
    overwrite: bool,
) -> Path:
    source_mesh_path = _ensure_centered_ply(mesh_path) if recenter_ply else mesh_path
    output_path = output_dir / f"{mesh_path.stem}.glb"

    if output_path.exists() and not overwrite:
        log_step(f"Skipping existing GLB: {output_path}")
        return output_path

    mesh = load_mesh(source_mesh_path).copy()
    scale = infer_scale_to_meters(source_mesh_path, mesh_units)
    if scale != 1.0:
        mesh.apply_scale(scale)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mesh.export(file_type="glb"))
    log_step(
        "Exported GLB: "
        f"source={_display_path(source_mesh_path)} output={_display_path(output_path)}"
    )
    return output_path


def iter_model_paths(models_dir: Path) -> list[Path]:
    model_paths = []
    for suffix in (".ply", ".obj"):
        model_paths.extend(sorted(models_dir.glob(f"obj_*{suffix}")))
    return model_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dataset object models to GLB files."
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Directory containing object models such as obj_000001.ply",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for GLBs. Defaults to <models-dir>/glb",
    )
    parser.add_argument(
        "--mesh-units",
        choices=("mm", "m"),
        default="mm",
        help="Units of the source meshes before export. GLB is exported in meters.",
    )
    parser.add_argument(
        "--recenter-ply",
        action="store_true",
        help="Create a centered copy of each PLY before GLB export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GLBs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir) if args.output_dir else models_dir / "glb"

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    model_paths = iter_model_paths(models_dir)
    if not model_paths:
        raise FileNotFoundError(f"No obj_*.ply or obj_*.obj files found in {models_dir}")

    exported = []
    for mesh_path in model_paths:
        exported.append(
            convert_mesh_to_glb(
                mesh_path=mesh_path,
                output_dir=output_dir,
                mesh_units=args.mesh_units,
                recenter_ply=args.recenter_ply,
                overwrite=args.overwrite,
            )
        )

    print(
        {
            "models_dir": str(models_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "num_exported": len(exported),
        }
    )


if __name__ == "__main__":
    main()
