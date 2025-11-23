# Cloth Manipulation Dataset Collection

Complete system for collecting robotic cloth manipulation datasets using Newton physics simulation with per-environment trajectory generation and randomized cloth library.

## ✅ Trajectory System Status

**VERIFIED ACTIVE**: Per-environment generated trajectories are fully integrated and running.

### Current Trajectory Flow

Each environment executes an independent sequence with 5 distinct phases:

1. **Phase 1: Approach / Descend** (0 → `grasp_time`)
   - Move from `lift_pos` → `grasp_pos` (or `pre_grasp` → `grasp`)
   - Gripper: **OPEN**
   - Interpolation: Linear

2. **Phase 2: Grasp Dwell** (`grasp_time` → `grasp_time + grasp_dwell`)
   - Hold at `post_grasp_pos`
   - Gripper: **CLOSING** (Transitions to closed)
   - Allows physics to stabilize grasp

3. **Phase 3: Lift / Ascend** (`...` → `... + lift_time`)
   - Move from `post_grasp_pos` → `lift_pos`
   - Gripper: **CLOSED**
   - Lifts cloth from table

4. **Phase 4: Transport** (`...` → `... + drop_time`)
   - Move from `lift_pos` → `drop_pos` (or `pre_drop`)
   - Gripper: **CLOSED**
   - Carries cloth to release zone

5. **Phase 5: Drop Dwell** (`...` → `total_time`)
   - Hold at `post_drop_pos`
   - Gripper: **OPEN**
   - Releases cloth

### Per-Environment Randomization

Each environment gets **unique** trajectory parameters:
- **Grasp position**: Random point on cloth surface ± position noise
- **Drop position**: Offset from lift by random amount in configurable range
- **Timing**: Each phase duration has ± timing noise
- **Cloth center**: Automatically offset based on environment grid position

---

## 📁 Configuration System

### JSON Config File

Control all simulation parameters via `config.json`. The structure supports nested configuration for robot, control, and cloth properties.

```json
{
  "simulation": {
    "n_env": 4,
    "iterations": 10,
    "num_frames": 1000,
    "seed": 42,
    "save_interval": 1,
    "viewer": "usd",
    "device": "cuda:0",
    "add_cloth": true,
    "add_robot": true,
    "fps": 60,
    "sim_substeps": 15
  },
  "paths": {
    "out_dir": "output",
    "cloth_library_dir": "assets",
    "mesh_file": null
  },
  "cloth": {
    "source": "library",
    "orientation": {
      "use_default": false,
      "sampling_degrees": [-180, 180]
    },
    "keep_flat": true,
    "use_same_mesh_per_env": true,
    "base_position": [0.5, 0.25, 0.25],
    "solver_iterations": 5
  },
  "trajectory": {
    "position_noise_std": 0.02,
    "timing_noise_std": 0.1,
    "grasp_height": 0.20,
    "lift_height": 0.5,
    "drop_offset_range": [-0.15, 0.15],
    "cloth_extent_m": 0.4,
    "timing": {
      "grasp_dwell": 0.5,
      "drop_dwell": 0.5
    },
    "orientations": {
      "down_quat": [1.0, 0.0, 0.0, 0.0]
    }
  },
  "data_collection": {
    "grasp_offset_steps": 5,
    "release_offset_steps": 5,
    "point_sample_rate": 1000,
    "camera": {
      "position": [0.5, 0.0, 2.0],
      "direction": [0.0, 0.0, -1.0],
      "grid": 256
    }
  },
  "contact": {
    "table_top_z": 0.20,
    "ee_radius": 0.035,
    "table_margin": 0.003,
    "robot_friction": 1.0,
    "table_friction": 0.5
  },
  "control": {
    "gripper_open": 0.8,
    "gripper_closed": 0.03,
    "k_null": 1.0,
    "use_nullspace": true,
    "max_joint_qd": 2.0
  },
  "robot": {
    "flip_about_xy": false,
    "base_position": [-0.25, 0.0, 0.0]
  }
}
```

### Parameter Reference

#### `simulation`
- `n_env`: Number of parallel environments.
- `iterations`: Number of reset-and-run cycles.
- `num_frames`: Steps per iteration.
- `sim_substeps`: Physics substeps per frame (default: 15).

#### `cloth`
- `source`: "library" (random selection) or "dir" (single file).
- `orientation.sampling_degrees`: [min, max] range for random Z-rotation.
- `base_position`: World-space cloth placement [x, y, z].
- `solver_iterations`: Solver iterations for cloth physics (default: 5).

#### `trajectory`
- `position_noise_std`: Gaussian noise on grasp/drop positions.
- `timing.grasp_dwell`: Time to hold grasp before lifting.
- `orientations.down_quat`: Fixed end-effector orientation quaternion.

#### `control`
- `k_null`: Null-space stiffness for IK.
- `max_joint_qd`: Velocity clamp for safety.
- `gripper_open/closed`: Width parameters.

#### `robot`
- `base_position`: Robot base location [x, y, z].
- `flip_about_xy`: Rotate robot 180 deg (if needed for coordinate conventions).

---

## 🚀 Usage

### Run with Config File

```powershell
# Use config file
python -m newton.examples.cloth.data_collection.example_collect_dataset --config config.json

# Override specific parameters via CLI (takes precedence)
python -m newton.examples.cloth.data_collection.example_collect_dataset --config config.json --n-env 8 --iterations 20

# Run without config (uses defaults)
python -m newton.examples.cloth.data_collection.example_collect_dataset --n-env 4 --iterations 5 --cloth-lib assets
```

### CLI Arguments

All config.json parameters can be overridden via command-line:

```powershell
--config PATH          # Path to JSON config file
--n-env N              # Number of parallel environments
--iterations N         # Number of simulation iterations
--num-frames N         # Steps per iteration
--seed N               # Random seed
--save-interval N      # Save every N steps
--out-dir PATH         # Output directory
--cloth-lib PATH       # Cloth library directory
--cloth-source {lib,dir} # Source mode
--mesh-file PATH       # Specific mesh file (if source=dir)
--viewer {usd,gl,none} # Viewer type
```

---

## 📦 Output Format

### Per-Datapoint Folders

Each timestep × environment produces a separate folder:

```
output/
├── datapoint_0/
│   ├── step.npy          # int32: timestep index
│   ├── mesh_vertices.npy # float32 (V, 3): cloth vertex positions
│   ├── points2d.npy      # float32 (K, 2): projected 2D points (normalized)
│   ├── depth.npy         # float32 (K,): depth values
│   ├── ee.npy            # float32 (3,): end-effector position
│   └── contact.npy       # int32 (C,): contact vertex indices
├── datapoint_1/
│   └── ...
```

### Optional USD Snapshots

When `viewer == "usd"`, saves a USD file per iteration:

```
output/
└── iteration_0000.usd   # First saved timestep, all envs as Points
```

---

## 🧪 Cloth Library Setup

The cloth preprocessing and dataset preparation docs have moved.

- For metadata population and preprocessing (center + remesh + quadrants), see:
  `newton/examples/cloth/data_collection/cloth_preprocessing/README.md`

Runtime note: the environment assumes meshes are preprocessed and reads cloth parameters and orientation only from `metadata.json` or the run config.

---

## 🔧 Architecture

### Module Organization

```
data_collection/
├── cloth_franka_env.py            # Main simulation environment (import only)
├── example_collect_dataset.py     # Dataset collection driver
├── kernels.py                     # Centralized Warp kernels
├── data_collector.py              # Projection & buffering
├── trajectory_generator.py        # Per-env trajectory generation
├── simulation_env.py              # Cloth library management
├── config.json                    # Configuration template
└── README.md                      # This file
```

### Kernel Consolidation

All Warp kernels live in `kernels.py`:
- `compute_ee_delta`: End-effector pose error
- `make_compute_body_out_kernel`: Jacobian body-out kernel factory
- `extract_ee_positions_kernel`: Per-env EE position extraction

---

## 🎯 Key Features

### ✅ Per-Environment Trajectories
- Each env has independent grasp/lift/drop positions
- Reproducible via seeded RNG
- Position and timing noise for diversity

### ✅ Robust Multi-Env Control
- **Corrected Jacobian**: Computes Jacobian per-environment using correct body IDs.
- **Null-space Control**: Maintains "elbow-up" posture via null-space projection.
- **Velocity Clamping**: Prevents instability via `max_joint_qd`.

### ✅ Cloth Library Integration
- Random mesh selection per iteration
- Random Z-rotation per iteration
- Consistent mesh across envs within iteration

### ✅ Occlusion-Aware Projection
- Orthographic-like camera projection
- Z-buffer occlusion via grid binning
- Per-frame normalization of 2D coordinates

---

## 🐛 Troubleshooting

### "Incomplete argument annotations on function..."

**Cause**: Warp kernel parameters missing type annotations.

**Fix**: Ensure all kernel parameters have `wp.array(dtype=...)` annotations (already fixed in `kernels.py`).

### "Variable not allowed in type expression"

**Status**: This is a static checker warning about Warp's annotation style. Safe to ignore—runtime works correctly.

### Trajectories Not Diverging Between Envs

**Check**: Verify `example.set_generated_trajectories(trajs)` is called before simulation starts.

### Cloth Library Not Loading

**Check**: Ensure `cloth_library_dir` points to a directory containing subfolders with `.obj` or `.stl` files.

---

## 📊 Performance Tips

### CUDA Graph Capture

The simulation automatically captures CUDA graphs when using GPU. If you see repeated "Module load" messages, ensure kernels are precompiled (already done via `set_up_control`).

### Large-Scale Dataset Collection

For maximum throughput:
1. Increase `n_env` (e.g., 16-32 environments)
2. Set `save_interval > 1` to reduce I/O
3. Use `viewer: "none"` to disable rendering
4. Consider batching multiple iterations per process

---

## 📚 References

- Newton Physics: https://github.com/NVIDIA/newton
- Warp: https://github.com/NVIDIA/warp
- Trajectory Generator: `trajectory_generator.py`
- Data Collector: `data_collector.py`
