"""
Trajectory generation utilities for multi-environment cloth manipulation.

Generates simple grasp-lift-drop trajectories with configurable noise.
Each environment receives an independent trajectory seeded from a master seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class TrajectoryConfig:
    """
    Configuration for per-environment trajectory generation.

    All distances are in meters, times in seconds.
    """

    # Noise controls (keep visibly editable/tunable)
    position_noise_std: float = 0.02    # XYZ Gaussian noise on waypoints (m)
    timing_noise_std: float = 0.10      # Gaussian noise on segment durations (s)

    # Waypoint heights
    grasp_height: float = 0.20          # Z at grasp
    lift_height: float = 0.35           # Z at lift (above table)

    # Drop displacement ranges relative to lift (in XY)
    drop_offset_range: Tuple[float, float] = (-0.15, 0.15)

    # Safety caps
    min_segment_time: float = 0.4       # Lower bound on any segment duration

    # Dwell (stationary) durations at grasp and drop (seconds)
    grasp_dwell: float = 0.5
    drop_dwell: float = 0.5


class TrajectoryGenerator:
    """
    Generates simple one-motion trajectories (grasp -> lift -> drop -> open)
    for each environment with small position/timing noise.
    """

    def __init__(self, config: Optional[TrajectoryConfig] = None, seed: Optional[int] = None):
        self.config = config or TrajectoryConfig()
        self.master_rng = np.random.RandomState(seed)

    def _rng_for_env(self, env_id: int) -> np.random.RandomState:
        # Derive a deterministic child RNG per environment for reproducibility
        seed = self.master_rng.randint(0, 2**31 - 1)
        return np.random.RandomState(seed + env_id * 9973)

    def generate_for_env(
        self,
        env_id: int,
        cloth_center_xyz: Tuple[float, float, float],
        cloth_extent_m: float,
    ) -> Dict:
        """
        Generate a single environment trajectory.

        Returns a dict with:
        - grasp_pos, lift_pos, drop_pos: np.ndarray shape (3,)
        - grasp_time, lift_time, drop_time, total_time: floats
        - env_id: int
        """
        rng = self._rng_for_env(env_id)
        cx, cy, cz = cloth_center_xyz
        half = cloth_extent_m * 0.5

        # Grasp on the cloth with small noise
        gx = cx + rng.uniform(-half, half) + rng.normal(0.0, self.config.position_noise_std)
        gy = cy + rng.uniform(-half, half) + rng.normal(0.0, self.config.position_noise_std)
        gz = self.config.grasp_height + rng.normal(0.0, self.config.position_noise_std)
        grasp = np.array([gx, gy, gz], dtype=np.float32)

        # Lift: vertically above the grasp
        lift = grasp.copy()
        lift[2] = self.config.lift_height + rng.normal(0.0, self.config.position_noise_std)

        # Drop: small XY displacement from lift
        doff = self.config.drop_offset_range
        dx = rng.uniform(doff[0], doff[1])
        dy = rng.uniform(doff[0], doff[1])
        drop = lift + np.array([dx, dy, 0.0], dtype=np.float32)

        # Timings with noise and clamped minimums (move durations)
        grasp_time = max(self.config.min_segment_time, 2.0 + rng.normal(0.0, self.config.timing_noise_std))
        lift_time = max(self.config.min_segment_time, 2.0 + rng.normal(0.0, self.config.timing_noise_std))
        drop_time = max(self.config.min_segment_time, 1.5 + rng.normal(0.0, self.config.timing_noise_std))

        # Dwell (stationary) durations
        grasp_dwell = max(0.0, float(self.config.grasp_dwell))
        drop_dwell = max(0.0, float(self.config.drop_dwell))

        total_time = float(grasp_time + grasp_dwell + lift_time + drop_time + drop_dwell)

        return {
            "env_id": env_id,
            # Core phase waypoints
            "grasp_pos": grasp,
            "lift_pos": lift,
            "drop_pos": drop,
            # Pre/Post phase waypoints (explicit, currently equal to their core counterparts)
            # Pre-grasp: approach target at which gripper is still open
            "pre_grasp_pos": grasp.copy(),
            # Post-grasp: same position while gripper transitions to closed during dwell
            "post_grasp_pos": grasp.copy(),
            # Pre-drop: transport target while gripper is still closed
            "pre_drop_pos": drop.copy(),
            # Post-drop: same position while gripper transitions to open during dwell
            "post_drop_pos": drop.copy(),
            "grasp_time": float(grasp_time),
            "grasp_dwell": float(grasp_dwell),
            "lift_time": float(lift_time),
            "drop_time": float(drop_time),
            "drop_dwell": float(drop_dwell),
            "total_time": total_time,
        }

    def generate_batch(
        self,
        n_env: int,
        cloth_centers_xyz: List[Tuple[float, float, float]],
        cloth_extent_m: float,
    ) -> List[Dict]:
        """Generate per-environment trajectories as a list of dicts."""
        assert len(cloth_centers_xyz) == n_env, "cloth_centers_xyz must match n_env"
        return [self.generate_for_env(i, cloth_centers_xyz[i], cloth_extent_m) for i in range(n_env)]
