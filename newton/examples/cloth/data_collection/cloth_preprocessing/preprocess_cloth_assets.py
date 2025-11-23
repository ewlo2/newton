# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path
import argparse
from typing import Dict, List

import numpy as np
import trimesh
import pymeshlab

try:
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

DEFAULT_OUTPUT_MESH_NAME = "mesh_centered_remeshed.obj"


def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    s = math.sin(angle_rad / 2.0)
    w = math.cos(angle_rad / 2.0)
    x, y, z = axis * s
    return np.array([x, y, z, w], dtype=np.float32)


def rotate_points(points: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    q = quat_xyzw
    w = q[3]
    x, y, z = q[0], q[1], q[2]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return (R @ points.T).T


def compute_quadrant_indices(verts: np.ndarray, keep_ratio: float = 0.1) -> Dict[str, List[int]]:
    xy = verts[:, :2]
    cx, cy = np.median(xy[:, 0]), np.median(xy[:, 1])
    N = verts.shape[0]
    k = max(1, int(keep_ratio * N))
    d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2

    def top_left_mask():
        return (xy[:, 0] < cx) & (xy[:, 1] >= cy)

    def top_right_mask():
        return (xy[:, 0] >= cx) & (xy[:, 1] >= cy)

    def bottom_left_mask():
        return (xy[:, 0] < cx) & (xy[:, 1] < cy)

    def bottom_right_mask():
        return (xy[:, 0] >= cx) & (xy[:, 1] < cy)

    out: Dict[str, List[int]] = {}
    for name, mask_fn in [
        ("top_left", top_left_mask),
        ("top_right", top_right_mask),
        ("bottom_left", bottom_left_mask),
        ("bottom_right", bottom_right_mask),
    ]:
        mask = mask_fn()
        idx = np.nonzero(mask)[0]
        if idx.size <= k:
            chosen = idx
        else:
            order = np.argsort(-d2[idx])
            chosen = idx[order[:k]]
        out[name] = [int(i) for i in chosen.tolist()]
    return out


def remesh(vertices: np.ndarray, faces: np.ndarray, percent: float) -> tuple[np.ndarray, np.ndarray]:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(percent))
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


essential_keys = [
    ("source_mesh_file", None),
    ("orientation.base_rotation.axis", None),
    ("orientation.base_rotation.degrees", None),
    ("downsampling.perform_remeshing", True),
    ("downsampling.edge_length_percentage", None),
]


def _get_meta_val(meta: Dict, dotted: str):
    cur = meta
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_metadata(meta: Dict, folder: Path) -> tuple[bool, str]:
    # Ensure essential fields exist and are valid
    for key, required_val in essential_keys:
        val = _get_meta_val(meta, key)
        if val is None:
            return False, f"Missing metadata field: {key}"
        if required_val is not None and val != required_val:
            return False, f"Metadata field {key} must be {required_val}, got {val}"
    # Check source mesh exists
    src = folder / str(_get_meta_val(meta, "source_mesh_file"))
    if not src.exists():
        return False, f"Source mesh not found: {src}"
    return True, "OK"


def preprocess_folder(folder: Path, visualize: bool = False) -> None:
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        print(f"[skip] No metadata.json in {folder}")
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[skip] Cannot parse metadata: {meta_path}")
        return

    ok, reason = validate_metadata(meta, folder)
    if not ok:
        print(f"[skip] {folder.name}: {reason}")
        return

    # Load source mesh
    mesh_path = folder / str(meta["source_mesh_file"])
    mesh = trimesh.load_mesh(str(mesh_path))
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)

    # STL axis swap
    if mesh_path.suffix.lower() == ".stl":
        V = V[:, [0, 2, 1]]

    # Remesh using metadata percentage
    percent = float(meta["downsampling"]["edge_length_percentage"])
    V_rm, F_rm = remesh(V, F, percent)
    V_rm = V_rm.astype(np.float32)
    F_rm = F_rm.astype(np.int32)

    # Center by centroid
    centroid = np.mean(V_rm, axis=0, dtype=np.float32)
    V_centered = (V_rm - centroid).astype(np.float32)

    # Compute laid-flat vertices using metadata base rotation
    base = meta["orientation"]["base_rotation"]
    axis = np.asarray(base["axis"], dtype=np.float32)
    deg = float(base["degrees"])
    qx = axis_angle_quat(axis, math.radians(deg))
    V_flat = rotate_points(V_centered, qx)

    # Quadrants
    quadrants = compute_quadrant_indices(V_flat, keep_ratio=0.5)

    # Save centered+remeshed OBJ
    out_mesh_path = folder / DEFAULT_OUTPUT_MESH_NAME
    tm = trimesh.Trimesh(vertices=V_centered, faces=F_rm, process=False)
    tm.export(out_mesh_path)
    print(f"[ok] Wrote {out_mesh_path}")

    # Save indices
    indices_filename = "quadrant_indices.json"
    with open(folder / indices_filename, "w") as f:
        json.dump(quadrants, f, indent=2)
    print(f"[ok] Wrote {folder / indices_filename}")

    # Update metadata (do not overwrite user-provided cloth parameters)
    meta["mesh_file"] = out_mesh_path.name
    meta["quadrant_indices_file"] = indices_filename
    meta["vertex_count"] = int(V_centered.shape[0])
    preproc = meta.get("preprocessed", {}) if isinstance(meta.get("preprocessed", {}), dict) else {}
    preproc.update({"centered": True, "remeshed": True})
    meta["preprocessed"] = preproc
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] Updated {meta_path}")

    # Optional visualization
    if visualize:
        try:
            if _HAS_O3D:
                _visualize_o3d(V_flat, F_rm, quadrants)
            else:
                _visualize_trimesh(V_flat, F_rm, quadrants)
        except Exception as e:
            print(f"[warn] Visualization failed for {folder.name}: {e}")


def _visualize_trimesh(V: np.ndarray, F: np.ndarray, quadrants: Dict[str, List[int]]) -> None:
    colors = np.tile(np.array([180, 180, 200, 255], dtype=np.uint8), (V.shape[0], 1))
    palette = {
        "top_left": np.array([255, 0, 0, 255], dtype=np.uint8),
        "top_right": np.array([0, 255, 0, 255], dtype=np.uint8),
        "bottom_left": np.array([0, 0, 255, 255], dtype=np.uint8),
        "bottom_right": np.array([255, 165, 0, 255], dtype=np.uint8),
    }
    for name, idxs in quadrants.items():
        if name in palette:
            colors[np.asarray(idxs, dtype=np.int32)] = palette[name]
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    m.visual.vertex_colors = colors
    m.show()


def _visualize_o3d(V: np.ndarray, F: np.ndarray, quadrants: Dict[str, List[int]]) -> None:
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(F.astype(np.int32)),
    )
    mesh_o3d.compute_vertex_normals()
    base_color = np.tile(np.array([[0.7, 0.7, 0.8]], dtype=np.float64), (V.shape[0], 1))
    palette = {
        "top_left": np.array([1.0, 0.0, 0.0]),
        "top_right": np.array([0.0, 1.0, 0.0]),
        "bottom_left": np.array([0.0, 0.0, 1.0]),
        "bottom_right": np.array([1.0, 0.65, 0.0]),
    }
    for name, idxs in quadrants.items():
        if name in palette and len(idxs) > 0:
            base_color[np.asarray(idxs, dtype=np.int32), :] = palette[name]
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(base_color)
    o3d.visualization.draw_geometries([mesh_o3d])


def main():
    parser = argparse.ArgumentParser(description="Preprocess cloth assets (center+remesh, quadrants)")
    print(str(Path(__file__).parent.parent / "assets"))
    parser.add_argument("--assets-root", type=str, default=str(Path(__file__).parent.parent / "assets"), help="Path to assets root directory (defaults to ../assets relative to this file)")
    parser.add_argument("--visualize", action="store_true", help="Visualize quadrant vertices after preprocessing")
    args = parser.parse_args()

    assets_root = Path(args.assets_root)
    if not assets_root.exists():
        print(f"Assets root not found: {assets_root}")
        return

    for sub in sorted(assets_root.iterdir()):
        if sub.is_dir():
            preprocess_folder(sub, visualize=args.visualize)


if __name__ == "__main__":
    main()
