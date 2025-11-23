"""
Data collection utilities: mesh sampling, point cloud projection, and timestep buffering.
Saves one file per iteration; can optionally trigger USD/binary viewer output upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DataCollectionConfig:
    """Config for collecting data with offsets around grasp/release events."""

    grasp_offset_steps: int = 5      # Start collecting N steps after grasp
    release_offset_steps: int = 5    # Stop collecting N steps after release
    camera_position: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 2.0], dtype=np.float32))
    camera_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))
    point_sample_count: int = 1000   # Number of points to sample from cloth mesh
    grid: int = 256                  # Z-buffer grid resolution for occlusion
    save_interval: int = 1           # Save every N simulation steps


class PointCloudProjector:
    """
    Projects 3D points to a plane normal to camera_direction, simulating occlusion
    via a simple z-buffer (closest points kept per 2D pixel bin).
    """

    def __init__(self, camera_position: np.ndarray, camera_direction: np.ndarray):
        cp = np.asarray(camera_position, dtype=np.float32)
        cd = np.asarray(camera_direction, dtype=np.float32)
        self.camera_pos = cp
        self.camera_dir = cd / (np.linalg.norm(cd) + 1e-9)
        self._compute_basis()

    def _compute_basis(self):
        # Camera axes: z looks opposite to view dir; choose a stable up
        self.cam_z = -self.camera_dir
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(self.cam_z, up))) > 0.99:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.cam_x = np.cross(up, self.cam_z).astype(np.float32)
        self.cam_x /= (np.linalg.norm(self.cam_x) + 1e-9)
        self.cam_y = np.cross(self.cam_z, self.cam_x).astype(np.float32)

    def project(self, pts3: np.ndarray, grid: int = 256) -> Dict[str, np.ndarray]:
        """
        Project 3D points to 2D using camera basis and keep front-most per cell.
        Returns dict with keys: points_2d (N,2), depth (N,), visible_mask (N,)
        """
        pts3 = np.asarray(pts3, dtype=np.float32)
        rel = pts3 - self.camera_pos[None, :]
        x = rel @ self.cam_x
        y = rel @ self.cam_y
        z = rel @ self.cam_z  # smaller is closer to camera plane

        # Normalize x,y into [0,1] by simple min/max (frame-local)
        # Then bin into a grid and keep smallest z per bin
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        xr = (x - xmin) / (max(xmax - xmin, 1e-6))
        yr = (y - ymin) / (max(ymax - ymin, 1e-6))
        ix = np.clip((xr * (grid - 1)).astype(np.int32), 0, grid - 1)
        iy = np.clip((yr * (grid - 1)).astype(np.int32), 0, grid - 1)
        flat = ix + iy * grid

        # Keep closest per bin
        best_z = {}
        keep = np.zeros_like(z, dtype=bool)
        for i, f in enumerate(flat):
            zi = float(z[i])
            if f not in best_z or zi < best_z[f][0]:
                if f in best_z:
                    keep[best_z[f][1]] = False
                best_z[f] = (zi, i)
                keep[i] = True

        pts2 = np.stack([x[keep], y[keep]], axis=1)
        depth = z[keep]
        return {"points_2d": pts2, "depth": depth, "visible_mask": keep}


class DataCollector:
    """Buffers per-timestep data and saves a single file per iteration."""

    def __init__(self, cfg: DataCollectionConfig, out_dir: str = "output", seed: Optional[int] = 42):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.projector = PointCloudProjector(cfg.camera_position, cfg.camera_direction)
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.buffer: List[Dict] = []

    def should_collect(self, step: int, grasp_step: int, release_step: int) -> bool:
        return (step >= grasp_step + self.cfg.grasp_offset_steps) and (step <= release_step + self.cfg.release_offset_steps)

    def add_step(
        self,
        step_idx: int,
        meshes_world: List[np.ndarray],     # list per env (V,3)
        ee_positions: List[np.ndarray],     # list per env (3,)
        contact_indices: List[np.ndarray],  # list per env (K,) or empty
    ) -> None:
        """Append one timestep worth of data for all environments."""
        step_entry: Dict = {"step": step_idx, "envs": []}
        for env_id, verts in enumerate(meshes_world):
            verts = np.asarray(verts, dtype=np.float32)
            # Sample subset for projection
            if len(verts) > self.cfg.point_sample_count:
                sel = self.rng.choice(len(verts), size=self.cfg.point_sample_count, replace=False)
                sample = verts[sel]
            else:
                sample = verts
                sel = np.arange(len(verts), dtype=np.int32)

            proj = self.projector.project(sample, grid=self.cfg.grid)
            step_entry["envs"].append(
                {
                    "mesh_vertices": verts,                  # Full mesh
                    "sample_indices": sel,                   # Indices of sampled points from full mesh
                    "points_2d": proj["points_2d"],        # Projected visible subset
                    "depth": proj["depth"],               # Depths of visible subset
                    "ee_position": np.asarray(ee_positions[env_id], dtype=np.float32),
                    "contact_vertices": np.asarray(contact_indices[env_id], dtype=np.int32),
                }
            )
        self.buffer.append(step_entry)

    def save_iteration(self, iteration: int) -> Path:
        """Save a single npz file for this iteration. One file contains all timesteps for all envs."""
        path = self.out / f"iteration_{iteration:04d}.npz"
        # Flatten structure into arrays-of-objects for compactness
        steps = [entry["step"] for entry in self.buffer]
        env_count = len(self.buffer[0]["envs"]) if self.buffer else 0

        # Pack per-env sequences
        packed = {"steps": np.array(steps, dtype=np.int32), "env_count": np.array([env_count], dtype=np.int32)}
        for env_id in range(env_count):
            meshes = [entry["envs"][env_id]["mesh_vertices"] for entry in self.buffer]
            pts2d = [entry["envs"][env_id]["points_2d"] for entry in self.buffer]
            depth = [entry["envs"][env_id]["depth"] for entry in self.buffer]
            ee = [entry["envs"][env_id]["ee_position"] for entry in self.buffer]
            contact = [entry["envs"][env_id]["contact_vertices"] for entry in self.buffer]

            packed[f"env_{env_id:02d}_meshes"] = np.array(meshes, dtype=object)
            packed[f"env_{env_id:02d}_points2d"] = np.array(pts2d, dtype=object)
            packed[f"env_{env_id:02d}_depth"] = np.array(depth, dtype=object)
            packed[f"env_{env_id:02d}_ee"] = np.array(ee, dtype=object)
            packed[f"env_{env_id:02d}_contact"] = np.array(contact, dtype=object)

        np.savez_compressed(path, **packed)
        return path
