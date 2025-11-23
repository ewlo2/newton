# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import Dict, List

import numpy as np

DEFAULT_BASE_ROT_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_BASE_ROT_DEG = 90.0
DEFAULT_Z_ROT_DEG = 0.0
DEFAULT_CENTER_OFFSET = [0.5, 0.25, 0.25]
DEFAULT_REMESH_PERCENT = 0.65  # consumed by preprocessing script
DEFAULT_CLOTH_PARAMS = {
    "density": 0.2,
    "scale": 1.0,
    "tri_ke": 1e2,
    "tri_ka": 1e2,
    "tri_kd": 1.5e-6,
    "edge_ke": 1e-4,
    "edge_kd": 1e-3,
    "particle_radius": 0.006,
}


def _mesh_candidates(folder: Path) -> List[Path]:
    return list(folder.glob("*.obj")) + list(folder.glob("*.stl"))


def _load_metadata(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _merge_defaults(meta: Dict, folder: Path) -> Dict:
    meta = dict(meta) if meta else {}
    # Set source_mesh_file if missing
    if "source_mesh_file" not in meta:
        candidates = _mesh_candidates(folder)
        if candidates:
            meta["source_mesh_file"] = candidates[0].name
    # Orientation defaults
    if "orientation" not in meta or not isinstance(meta.get("orientation"), dict):
        meta["orientation"] = {}
    ori = meta["orientation"]
    if "base_rotation" not in ori or not isinstance(ori.get("base_rotation"), dict):
        ori["base_rotation"] = {"axis": DEFAULT_BASE_ROT_AXIS.tolist(), "degrees": DEFAULT_BASE_ROT_DEG}
    if "default_z_rotation_degrees" not in ori:
        ori["default_z_rotation_degrees"] = DEFAULT_Z_ROT_DEG
    # Placement hint
    if "center_offset" not in meta:
        meta["center_offset"] = DEFAULT_CENTER_OFFSET
    # Downsampling defaults for preprocessing to consume later
    if "downsampling" not in meta or not isinstance(meta.get("downsampling"), dict):
        meta["downsampling"] = {
            "perform_remeshing": True,
            "method": "isotropic_explicit_remeshing",
            "edge_length_percentage": DEFAULT_REMESH_PERCENT,
        }
    else:
        ds = meta["downsampling"]
        ds.setdefault("perform_remeshing", True)
        ds.setdefault("method", "isotropic_explicit_remeshing")
        ds.setdefault("edge_length_percentage", DEFAULT_REMESH_PERCENT)
    # Cloth physical parameters
    if "cloth_parameters" not in meta or not isinstance(meta.get("cloth_parameters"), dict):
        meta["cloth_parameters"] = DEFAULT_CLOTH_PARAMS
    else:
        for k, v in DEFAULT_CLOTH_PARAMS.items():
            meta["cloth_parameters"].setdefault(k, v)
    return meta


def write_metadata_for_folder(folder: Path, overwrite: bool = False) -> None:
    meta_path = folder / "metadata.json"
    existing = {} if overwrite else _load_metadata(meta_path)
    merged = _merge_defaults(existing, folder)
    meta_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Populate cloth metadata (no preprocessing)")
    parser.add_argument("--assets-root", type=str, default=str(Path(__file__).parent / "assets"), help="Path to assets root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite metadata.json instead of merging defaults")
    args = parser.parse_args()

    assets_root = Path(args.assets_root)
    if not assets_root.exists():
        print(f"Assets root not found: {assets_root}")
        return

    for sub in sorted(assets_root.iterdir()):
        if sub.is_dir():
            write_metadata_for_folder(sub, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
