"""
Simulation environment wrapper for Newton: builds replicated Franka + cloth scene,
handles per-iteration cloth randomization (orientation only), and exposes helpers
for per-env world transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np

from newton import utils as nutils
from newton import geometry as ngeom


@dataclass
class ClothSpec:
    mesh_path: Path
    name: str
    metadata: Dict


@dataclass
class SimulationConfig:
    env_count: int = 4
    env_spacing: float = 1.0
    table_size: Tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_height: float = 0.6
    cloth_size_hint: float = 0.4  # only for placement spacing hints
    cloth_library_dir: Optional[Path] = None  # root directory containing cloth_type_i folders
    random_seed: int = 123
    # Orientation control
    use_default_orientation: bool = False  # if True, defer to metadata default orientation (no sampling)
    rot_sample_deg_min: float = 0.0       # inclusive min degrees for Z-rotation sampling
    rot_sample_deg_max: float = 360.0     # exclusive max degrees for Z-rotation sampling


class SimulationEnv:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.random_seed)
        self.env_offsets = self._compute_env_offsets(cfg.env_count, cfg.env_spacing)
        self._cloth_specs: List[ClothSpec] = []
        if cfg.cloth_library_dir:
            self._cloth_specs = self._load_cloth_library(cfg.cloth_library_dir)

        # Placeholders for external integration (Model/Builder/State)
        self.model = None
        self.builder = None
        self.state = None

    @staticmethod
    def _compute_env_offsets(n: int, spacing: float) -> List[np.ndarray]:
        # Arrange envs in a grid close to square
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        offsets = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                offsets.append(np.array([c * spacing, r * spacing, 0.0], dtype=np.float32))
                idx += 1
        return offsets

    @staticmethod
    def _load_cloth_library(root: Path) -> List[ClothSpec]:
        specs: List[ClothSpec] = []
        root = Path(root)
        if not root.exists():
            return specs
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            # Expect mesh file and optional metadata.json
            mesh_candidates = list(child.glob("*.obj")) + list(child.glob("*.stl"))
            if not mesh_candidates:
                continue
            mesh_path = mesh_candidates[0]
            meta = {}
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            # Prefer metadata-provided mesh_file if present and exists
            if isinstance(meta, dict) and "mesh_file" in meta:
                candidate = child / str(meta.get("mesh_file"))
                if candidate.exists():
                    mesh_path = candidate
            specs.append(ClothSpec(mesh_path=mesh_path, name=child.name, metadata=meta))
        return specs

    def pick_cloth_for_iteration(self) -> Optional[ClothSpec]:
        if not self._cloth_specs:
            return None
        return self.rng.choice(self._cloth_specs)

    def cloth_center_world(self, env_id: int) -> np.ndarray:
        # Center on table top for each env
        offset = self.env_offsets[env_id]
        x = offset[0] + 0.0
        y = offset[1] + 0.0
        z = self.cfg.table_height + 0.5 * self.cfg.table_size[2] + 1e-3
        return np.array([x, y, z], dtype=np.float32)

    def random_cloth_orientation(self) -> Optional[float]:
        """
        Returns:
            Optional[float]:
                - None to indicate 'use default from metadata'
                - Otherwise, a sampled Z-rotation in radians, drawn uniformly from the configured degrees range
        """
        if self.cfg.use_default_orientation:
            return None
        dmin = float(self.cfg.rot_sample_deg_min)
        dmax = float(self.cfg.rot_sample_deg_max)
        # Normalize and guard against invalid ranges
        if not math.isfinite(dmin) or not math.isfinite(dmax):
            dmin, dmax = 0.0, 360.0
        if abs(dmax - dmin) < 1e-9:
            # Zero-width range → effectively default orientation (but explicit angle)
            return float(math.radians(dmin))
        # Sample uniformly in [dmin, dmax)
        d = float(self.rng.uniform(dmin, dmax))
        return float(math.radians(d))

    # ---- Integration hooks (to be wired into actual Newton builder/model) ----

    def build_scene(self, cloth_spec: Optional[ClothSpec]) -> None:
        """
        This function should be integrated with newton.Builder to create:
        - A table per env
        - A Franka per env
        - A cloth per env using cloth_spec, placed at cloth_center with random Z-rotation
        - Replication across envs using env_offsets

        Here we provide the algorithm and data needed; actual Builder calls must
        be written where your Newton project constructs bodies.
        """
        # Pseudocode comments only; integration is repo-specific.
        # Use self.env_offsets, self.cloth_center_world(env), self.random_cloth_orientation()
        pass

    def get_env_offsets(self) -> List[np.ndarray]:
        return self.env_offsets
