"""
Collect dataset from the existing Franka + cloth example without modifying core logic.

This script drives the Example class step-by-step, samples cloth vertices,
projects them with occlusion, logs end-effector positions, and saves one .npz per iteration.

Notes:
- Cloth library randomization and per-iteration mesh selection are not wired into Example yet.
  The SimulationEnv stub (simulation_env.py) provides hooks to implement this inside your builder.
- Contact vertex extraction is left as a TODO because the collision structure is model-dependent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import math

import numpy as np
import warp as wp

import newton
from newton.examples.cloth.data_collection.data_collector import DataCollector, DataCollectionConfig
from newton.examples.cloth.data_collection.trajectory_generator import TrajectoryConfig, TrajectoryGenerator
from newton.examples.cloth.data_collection.cloth_franka_env import Example
from newton.examples.cloth.data_collection import kernels as dc_kernels
from newton.examples.cloth.data_collection.simulation_env import SimulationEnv, SimulationConfig


def load_config(config_path: str | None) -> dict | None:
    """Load JSON configuration file if provided."""
    if config_path is None:
        return None
    with open(config_path, 'r') as f:
        return json.load(f)


def get_env_partition_counts(model, n_env: int) -> int:
    # Assumes all particles belong to cloth and are evenly replicated across envs
    assert model.particle_count % n_env == 0, "Particle count not divisible by envs; multiple cloths or mismatch?"
    return model.particle_count // n_env


def extract_meshes_by_env(state, per_env_particle_count: int, n_env: int) -> List[np.ndarray]:
    q = state.particle_q
    if q is None:
        return [np.zeros((0, 3), dtype=np.float32) for _ in range(n_env)]
    q_np = q.numpy()  # (total_particles, 3)
    meshes = []
    for env_id in range(n_env):
        start = env_id * per_env_particle_count
        end = start + per_env_particle_count
        meshes.append(q_np[start:end])
    return meshes


def extract_ee_positions_by_env(example: Example, state) -> List[np.ndarray]:
    """
    Compute per-environment end-effector positions using a Warp kernel.

    This avoids attempting to index `state.body_q` (a Warp array) on the host.
    The kernel composes the body transform with the `endeffector_offset` and
    writes out the translation for each environment thread (wp.tid()).
    """

    n = example.n_env
    # ensure kernel runs on the same device as body_q
    device = state.body_q.device if hasattr(state.body_q, "device") else None
    with wp.ScopedDevice(device):
        out = wp.empty(n, dtype=wp.vec3)
        wp.launch(
            dc_kernels.extract_ee_positions_kernel,
            dim=n,
            inputs=[state.body_q, example.endeffector_offset, example.endeffector_id, example.bodies_per_env],
            outputs=[out],
        )
        out_np = out.numpy()

    # return list of (3,) numpy arrays for downstream code
    return [out_np[i].astype(np.float32) for i in range(n)]


def compute_contact_indices(
    meshes_world: List[np.ndarray],
    ee_positions: List[np.ndarray],
    table_top_z: float = 0.20,
    ee_radius: float = 0.035,
    table_margin: float = 0.003,
) -> List[np.ndarray]:
    """
    Heuristic contact extraction:
    - Robot contact: cloth vertices within ee_radius of the end-effector position
    - Table contact: cloth vertices with z <= table_top_z + table_margin
    Returns per-env arrays of vertex indices (relative to that env's mesh slice)
    """
    out = []
    thresh2 = ee_radius * ee_radius
    z_thresh = table_top_z + table_margin
    for env_id, verts in enumerate(meshes_world):
        if len(verts) == 0:
            out.append(np.array([], dtype=np.int32))
            continue
        ee = ee_positions[env_id]
        # distances to EE
        d2 = np.sum((verts - ee[None, :]) ** 2, axis=1)
        near_ee = d2 <= thresh2
        near_table = verts[:, 2] <= z_thresh
        idx = np.nonzero(near_ee | near_table)[0].astype(np.int32)
        out.append(idx)
    return out


def main():
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--iterations", type=int, default=5, help="Number of simulation iterations")
    parser.add_argument("--n-env", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory to save npz files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory/data sampling")
    parser.add_argument("--save-interval", type=int, default=1, help="Save every N steps")
    parser.add_argument("--cloth-lib", type=str, default=None, help="Path to cloth library root (folders with meshes and optional metadata.json)")
    parser.add_argument("--cloth-source", type=str, choices=["lib", "dir"], default=None, help="Choose to load cloth from 'lib' (library) or 'dir' (single mesh file)")
    parser.add_argument("--mesh-file", type=str, default=None, help="Path to a single cloth mesh file (.obj/.stl) when using --cloth-source dir")
    viewer, args = newton.examples.init(parser)

    # Load config and override args if provided
    config = load_config(args.config)
    
    # Initialize default values
    table_top_z = 0.20
    ee_radius = 0.035
    table_margin = 0.003
    cloth_base = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    cloth_extent = 0.4
    
    if config:
        sim = config.get("simulation", {})
        paths = config.get("paths", {})
        traj_cfg_dict = config.get("trajectory", {})
        dc_cfg_dict = config.get("data_collection", {})
        contact_cfg = config.get("contact", {})
        cloth_cfg = config.get("cloth", {})
        
        # Override args with config values (CLI args still take precedence if explicitly set)
        args.n_env = sim.get("n_env", args.n_env)
        args.iterations = sim.get("iterations", args.iterations)
        args.num_frames = sim.get("num_frames", args.num_frames)
        args.seed = sim.get("seed", args.seed)
        args.save_interval = sim.get("save_interval", args.save_interval)
        args.viewer = sim.get("viewer", getattr(args, "viewer", None))
        args.out_dir = paths.get("out_dir", args.out_dir)
        args.cloth_lib = paths.get("cloth_library_dir", args.cloth_lib)
        # Cloth source and mesh-file (config-driven unless CLI provided)
        cfg_source = cloth_cfg.get("source", None)  # 'library' | 'dir'
        if args.cloth_source is None and cfg_source is not None:
            args.cloth_source = "lib" if cfg_source == "library" else ("dir" if cfg_source == "dir" else args.cloth_source)
        args.mesh_file = paths.get("mesh_file", args.mesh_file)
        
        # Build TrajectoryConfig from config
        timing_cfg = traj_cfg_dict.get("timing", {})
        _traj_cfg = TrajectoryConfig(
            position_noise_std=traj_cfg_dict.get("position_noise_std", 0.02),
            timing_noise_std=traj_cfg_dict.get("timing_noise_std", 0.1),
            grasp_height=traj_cfg_dict.get("grasp_height", 0.21),
            lift_height=traj_cfg_dict.get("lift_height", 0.35),
            drop_offset_range=tuple(traj_cfg_dict.get("drop_offset_range", [-0.15, 0.15])),
            grasp_dwell=timing_cfg.get("grasp_dwell", 0.5),
            drop_dwell=timing_cfg.get("drop_dwell", 0.5),
        )
        
        # Build DataCollectionConfig from config
        camera_cfg = dc_cfg_dict.get("camera", {})
        dc_cfg = DataCollectionConfig(
            grasp_offset_steps=dc_cfg_dict.get("grasp_offset_steps", 5),
            release_offset_steps=dc_cfg_dict.get("release_offset_steps", 5),
            point_sample_count=dc_cfg_dict.get("point_sample_rate", 1000),
            camera_position=np.array(camera_cfg.get("position", [0.5, 0.0, 2.0]), dtype=np.float32),
            camera_direction=np.array(camera_cfg.get("direction", [0.0, 0.0, -1.0]), dtype=np.float32),
            grid=camera_cfg.get("grid", 256),
            save_interval=args.save_interval
        )
        
        # Extract contact parameters
        table_top_z = contact_cfg.get("table_top_z", 0.20)
        ee_radius = contact_cfg.get("ee_radius", 0.035)
        table_margin = contact_cfg.get("table_margin", 0.003)
        
        # Extract cloth parameters
        cloth_base = np.array(cloth_cfg.get("base_position", [0.5, 0.25, 0.25]), dtype=np.float32)
        cloth_extent = traj_cfg_dict.get("cloth_extent_m", 0.4)
    else:
        # Use defaults
        _traj_cfg = TrajectoryConfig()
        dc_cfg = DataCollectionConfig(save_interval=args.save_interval)

    # Determine cloth source
    # Priority: CLI --cloth-source > config cloth.source > default fallback
    if args.cloth_source is None:
        # default heuristic: use lib if cloth_lib is provided, else dir
        args.cloth_source = "lib" if args.cloth_lib else "dir"

    # Initialize cloth library only if we use it
    sim_env = None
    if args.cloth_source == "lib" and args.cloth_lib:
        # Orientation config for library-driven sampling
        use_default_orientation = False
        rot_sample_deg_min = 0.0
        rot_sample_deg_max = 360.0
        if config:
            cloth_cfg = config.get("cloth", {})
            orient_cfg = cloth_cfg.get("orientation", {})
            use_default_orientation = bool(orient_cfg.get("use_default", False))
            if "sampling_degrees" in orient_cfg and isinstance(orient_cfg["sampling_degrees"], (list, tuple)) and len(orient_cfg["sampling_degrees"]) == 2:
                rot_sample_deg_min = float(orient_cfg["sampling_degrees"][0])
                rot_sample_deg_max = float(orient_cfg["sampling_degrees"][1])

        base_dir = Path(__file__).resolve().parent
        sim_env = SimulationEnv(SimulationConfig(
            env_count=args.n_env,
            cloth_library_dir=(base_dir / args.cloth_lib).resolve(),
            use_default_orientation=use_default_orientation,
            rot_sample_deg_min=rot_sample_deg_min,
            rot_sample_deg_max=rot_sample_deg_max,
        ))

    # Initialize trajectory generator and data collector
    _traj_gen = TrajectoryGenerator(_traj_cfg, seed=args.seed)
    collector = DataCollector(dc_cfg, out_dir=args.out_dir, seed=args.seed)

    steps_per_iter = args.num_frames
    datapoint_counter = 0

    for it in range(args.iterations):
        # Select mesh per source
        mesh_file = None
        z_rot = None
        if args.cloth_source == "lib" and sim_env is not None:
            spec = sim_env.pick_cloth_for_iteration()
            z_rot = sim_env.random_cloth_orientation()  # Optional[float]; None means use metadata default
            mesh_file = str(spec.mesh_path) if spec is not None else None
        else:
            # DIR mode: respect same orientation config if provided
            mesh_file = args.mesh_file
            z_rot = 0.0
            if config:
                cloth_cfg = config.get("cloth", {})
                orient_cfg = cloth_cfg.get("orientation", {})
                use_default = bool(orient_cfg.get("use_default", False))
                if use_default:
                    z_rot = None  # defer to metadata default
                else:
                    if "sampling_degrees" in orient_cfg and isinstance(orient_cfg["sampling_degrees"], (list, tuple)) and len(orient_cfg["sampling_degrees"]) == 2:
                        deg_min = float(orient_cfg["sampling_degrees"][0])
                        deg_max = float(orient_cfg["sampling_degrees"][1])
                        # Sample deterministically per-iteration for reproducibility
                        rng = np.random.RandomState(args.seed + it)
                        deg = float(rng.uniform(deg_min, deg_max))
                        z_rot = math.radians(deg)

        # Wire environment to use config and optional base position override from config
        example = Example(
            viewer,
            n_env=args.n_env,
            mesh_file=mesh_file,
            z_rotation=z_rot,
            base_position=tuple(cloth_base.tolist()) if isinstance(cloth_base, np.ndarray) else tuple(cloth_base),
            config=config,
        )
        
        # Quadrant indices are now available on example.quadrant_indices if metadata was loaded
        # Can be used for trajectory seeding or grasp point selection
        if example.quadrant_indices:
            print(f"[iteration {it}] Loaded quadrant indices with {sum(len(v) for v in example.quadrant_indices.values())} total vertices")
        
        # Generate per-env trajectories centered on each env's actual cloth base position from metadata
        base = np.array([
            float(example.cloth_base_pos[0]),
            float(example.cloth_base_pos[1]),
            float(example.cloth_base_pos[2])
        ], dtype=np.float32)
        cloth_centers = [tuple((base + example.env_offsets[i]).tolist()) for i in range(example.n_env)]
        trajs = _traj_gen.generate_batch(example.n_env, cloth_centers, cloth_extent_m=cloth_extent)
        example.set_generated_trajectories(trajs)
        per_env_particles = get_env_partition_counts(example.model, example.n_env)
        # For now, collect full range; if you have grasp/release steps, pass them to should_collect
        grasp_step = 0
        release_step = steps_per_iter - 1

        collector.reset()

        for step_idx in range(steps_per_iter):
            if not example.viewer.is_paused():
                example.step()
            example.render()

            if step_idx % dc_cfg.save_interval != 0:
                continue

            if not collector.should_collect(step_idx, grasp_step, release_step):
                continue

            # Extract per-env cloth meshes and end-effector positions
            meshes_world = extract_meshes_by_env(example.state_0, per_env_particles, example.n_env)
            ee_positions = extract_ee_positions_by_env(example, example.state_0)
            # Heuristic contact vertices: EE proximity or near table top
            contact_indices = compute_contact_indices(meshes_world, ee_positions)

            collector.add_step(step_idx, meshes_world, ee_positions, contact_indices)

        # Save each TIME STEP and ENVIRONMENT as its own datapoint folder
        # Folder: out_dir/datapoint_{i}/ with separate .npy arrays per datapoint
        if collector.buffer:
            env_count = len(collector.buffer[0]["envs"]) if collector.buffer else 0
            for t_idx, entry in enumerate(collector.buffer):
                step_val = int(entry["step"])
                for env_id in range(env_count):
                    env_entry = entry["envs"][env_id]
                    dp_dir = Path(args.out_dir) / f"datapoint_{datapoint_counter}"
                    dp_dir.mkdir(parents=True, exist_ok=True)

                    # Save each array separately
                    np.save(dp_dir / "step.npy", np.array(step_val, dtype=np.int32))
                    np.save(dp_dir / "mesh_vertices.npy", env_entry["mesh_vertices"])  # (V,3)
                    np.save(dp_dir / "points2d.npy", env_entry["points_2d"])         # (K,2)
                    np.save(dp_dir / "depth.npy", env_entry["depth"])                 # (K,)
                    np.save(dp_dir / "ee.npy", env_entry["ee_position"])              # (3,)
                    np.save(dp_dir / "contact.npy", env_entry["contact_vertices"])    # (Kc,)

                    datapoint_counter += 1
            print(f"Saved {datapoint_counter} total datapoints under {args.out_dir} so far")

        # Optional USD export per iteration (only if viewer == 'usd')
        if getattr(args, "viewer", None) == "usd" and collector.buffer:
            try:
                from pxr import Usd, UsdGeom, Gf
                usd_path = Path(args.out_dir) / f"iteration_{it:04d}.usd"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                world = UsdGeom.Xform.Define(stage, "/World")
                # Use first saved timestep as a static snapshot
                snapshot = collector.buffer[0]
                for env_id in range(len(snapshot["envs"])):
                    env_entry = snapshot["envs"][env_id]
                    pts = env_entry["mesh_vertices"].astype(np.float32)
                    prim_path = f"/World/ClothEnv_{env_id:02d}"
                    points = UsdGeom.Points.Define(stage, prim_path)
                    points.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in pts])
                    points.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.9)])
                stage.GetRootLayer().Save()
                print(f"USD snapshot saved: {usd_path}")
            except Exception as e:
                print(f"USD export skipped (error: {e})")

        # Reset/close between iterations
        # Recreate example next iteration when using cloth library
        if sim_env is None and it < args.iterations - 1:
            example.reset()

    example.viewer.close()


if __name__ == "__main__":
    main()
