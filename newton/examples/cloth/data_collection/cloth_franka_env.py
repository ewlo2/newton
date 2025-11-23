# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Cloth + Franka Simulation Environment
#
# This module builds a reusable simulation environment with a Franka robot
# and a deformable cloth, intended to be imported by dataset collection
# drivers and tools.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.utils import transform_twist
from newton.tests.unittest_utils import find_nan_members
from newton.examples.cloth.data_collection import kernels as dc_kernels

import os 
import json
import pymeshlab
import trimesh
from typing import Dict, List, Tuple, Optional
import math
import time


"""
Note: kernels were moved to newton.examples.cloth.data_collection.kernels
to consolidate Warp kernels in a single place.
"""


class Example:
    def __init__(self, viewer, n_env=4, mesh_file: Optional[str] = None, z_rotation: Optional[float] = None, base_position: Optional[Tuple[float, float, float]] = None, config: Optional[Dict] = None):
        # parameters
        #   simulation
        self.config = config

        def _cfg(path: str, default):
            cur = self.config if isinstance(self.config, dict) else None
            if cur is None:
                return default
            for part in path.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur

        self.add_cloth = _cfg("simulation.add_cloth", True)
        self.add_robot = _cfg("simulation.add_robot", True)
        self.sim_substeps = int(_cfg("simulation.sim_substeps", 15))
        # Cloth solver iterations (not dataset iterations)
        self.iterations = int(_cfg("cloth.solver_iterations", 5))
        self.fps = float(_cfg("simulation.fps", 60))
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.n_env = n_env
        # Optional per-environment generated trajectories (set later)
        self.generated_trajectories = None
        # Quadrant indices loaded from metadata (if available)
        self.quadrant_indices = None
        # Raw metadata for the selected cloth (set during mesh load)
        self.mesh_metadata: Optional[dict] = None
        # Optional override for cloth base position from config
        self.base_position_override = base_position

        #   contact (configurable)
        #       body-cloth contact
        self.cloth_body_contact_margin = float(_cfg("contact.cloth_body_contact_margin", 0.01))
        #       self-contact
        self.self_contact_radius = float(_cfg("contact.self_contact_radius", 0.002))
        self.self_contact_margin = float(_cfg("contact.self_contact_margin", 0.003))

        self.soft_contact_ke = float(_cfg("contact.soft_contact_ke", 100.0))
        self.soft_contact_kd = float(_cfg("contact.soft_contact_kd", 2e-3))

        self.robot_friction = float(_cfg("contact.robot_friction", 1.0))
        self.table_friction = float(_cfg("contact.table_friction", 0.5))
        self.self_contact_friction = float(_cfg("contact.self_contact_friction", 0.25))

        self.scene = ModelBuilder()
        self.soft_contact_max = 1000000

        self.viewer = viewer
        # Optional overrides for cloth library integration
        self.mesh_file_override = mesh_file
        self.z_rotation_override = z_rotation

        # Control-related configurable parameters
        self.gripper_open = float(_cfg("control.gripper_open", 0.8))
        self.gripper_closed = float(_cfg("control.gripper_closed", 0.03))
        offset_cfg = _cfg("control.endeffector_offset", [0.0, 0.0, 0.22])
        if isinstance(offset_cfg, (list, tuple)) and len(offset_cfg) == 3:
            self.endeff_vec = (float(offset_cfg[0]), float(offset_cfg[1]), float(offset_cfg[2]))
        else:
            self.endeff_vec = (0.0, 0.0, 0.22)
        self.k_null = float(_cfg("control.k_null", 1.0))
        self.gripper_scale = float(_cfg("control.gripper_scale", 0.04))
        self.use_nullspace = bool(_cfg("control.use_nullspace", True))
        self.include_rotation = bool(_cfg("control.include_rotation", True))
        _max_qd = _cfg("control.max_joint_qd", None)
        self.max_joint_qd = float(_max_qd) if _max_qd is not None else None
        # Trajectory orientation configuration
        dq_cfg = _cfg("trajectory.orientations.down_quat", [1.0, 0.0, 0.0, 0.0])
        if isinstance(dq_cfg, (list, tuple)) and len(dq_cfg) == 4:
            self.down_quat = (float(dq_cfg[0]), float(dq_cfg[1]), float(dq_cfg[2]), float(dq_cfg[3]))
        else:
            self.down_quat = (1.0, 0.0, 0.0, 0.0)
        # Robot base transform configuration
        self.robot_flip_about_xy = bool(_cfg("robot.flip_about_xy", True))
        base_pos_cfg = _cfg("robot.base_position", [0.0, 0.0, 0.9])
        if isinstance(base_pos_cfg, (list, tuple)) and len(base_pos_cfg) == 3:
            self.robot_base_position = (float(base_pos_cfg[0]), float(base_pos_cfg[1]), float(base_pos_cfg[2]))
        else:
            self.robot_base_position = (0.0, 0.0, 0.9)

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_builder(franka)
            self.bodies_per_env = franka.body_count
            self.dof_q_per_env = franka.joint_coord_count
            self.dof_qd_per_env = franka.joint_dof_count

        # add a table (positioned in front of the robot at origin)
        # Original: robot at (-0.5, -0.5), table at (0.0, -0.5) → 0.5m in front (X direction)
        # Now: robot at (0, 0). Position and size are configurable via config.contact.*
        table_hx = float(_cfg("contact.table_half_extent_x", 0.75))
        table_hy = float(_cfg("contact.table_half_extent_y", 0.75))
        table_hz = float(_cfg("contact.table_half_height", 0.1))
        table_top_z = _cfg("contact.table_top_z", None)
        center_z = (float(table_top_z) - table_hz) if (table_top_z is not None) else table_hz
        table_x = float(_cfg("contact.table_pos_x", 0.5))
        table_y = float(_cfg("contact.table_pos_y", 0.0))
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(table_x, table_y, center_z),
                wp.quat_identity(),
            ),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
        )

    # add the cloth mesh
        def load_and_process_mesh() -> Tuple[np.ndarray, np.ndarray]:
            """
            Load and process mesh file.
            
            Args:
                mesh_file: Path to mesh file
                input_scale_factor: Scale factor for mesh coordinates
                
            Returns:
                Tuple of (vertices, indices) arrays
            """
            # Determine mesh file path: we require an explicit path from config/library (no fallback)
            if self.mesh_file_override is not None:
                mesh_file = self.mesh_file_override
            else:
                raise ValueError("mesh_file must be provided via config/library; no internal fallback is allowed")
            if not os.path.exists(mesh_file):
                raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

            # Attempt to read per-cloth metadata.json (orientation, center, downsampling)
            meta = None
            quadrant_indices = None
            meta_path = os.path.join(os.path.dirname(mesh_file), "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    # keep around for placement/rotation usage
                    self.mesh_metadata = meta
                    # Load quadrant indices from separate file if referenced
                    if meta and "quadrant_indices_file" in meta:
                        indices_file = os.path.join(os.path.dirname(mesh_file), meta["quadrant_indices_file"])
                        if os.path.isfile(indices_file):
                            with open(indices_file, "r") as f:
                                quadrant_indices = json.load(f)
                except Exception:
                    meta = None
                    quadrant_indices = None

            # Load mesh
            mesh_data = trimesh.load_mesh(mesh_file)
            mesh_points_np = mesh_data.vertices
            faces_np = mesh_data.faces

            # Swap axes from xzy to xyz if mesh is STL
            if mesh_file.lower().endswith('.stl'):
                mesh_points_np = mesh_points_np[:, [0, 2, 1]]  # swap y and z

            # Runtime remeshing is disabled for data collection. Assume meshes are preprocessed.
            # If metadata includes 'downsampling', it is informational only here.
            
            # Store quadrant indices for later use (trajectory seeding)
            self.quadrant_indices = quadrant_indices
            
            # Scale mesh points
            mesh_points = [v * 1.0 for v in mesh_points_np]
            vertices_np = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in mesh_points]
            indices_np = np.array(faces_np, dtype=np.int32).flatten()
            
            return vertices_np, indices_np

    

        vertices, mesh_indices = load_and_process_mesh()
        
        if self.add_cloth:
            # Position cloth on the table in front of the robot
            # Original: robot at (-0.5, -0.5), cloth at (0.0, -0.25) → 0.5m in front (X), 0.25m to side (Y)
            # Now: robot at (0, 0), so cloth at (0.5, 0.25) to maintain same relative position
            # Compose rotation exclusively from metadata/config
            if self.mesh_metadata is None:
                raise ValueError("Missing cloth metadata; cannot determine orientation")
            orient = self.mesh_metadata.get("orientation", {})
            base_rot_meta = orient.get("base_rotation", None)
            if not base_rot_meta or "axis" not in base_rot_meta or "degrees" not in base_rot_meta:
                raise ValueError("Metadata.orientation.base_rotation must provide 'axis' and 'degrees'")
            axis = base_rot_meta["axis"]
            deg = float(base_rot_meta["degrees"])
            base_rot = wp.quat_from_axis_angle(wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), math.radians(deg))
            print(f"Base rotation {base_rot}")

            if self.z_rotation_override is not None:
                z_rot_rad = float(self.z_rotation_override)
            else:
                if "default_z_rotation_degrees" not in orient:
                    raise ValueError("No z-rotation override provided and metadata.orientation.default_z_rotation_degrees missing")
                z_rot_rad = math.radians(float(orient["default_z_rotation_degrees"]))
                print("Default rotation deg ")
            print(f"Z rotation (rad) {z_rot_rad}")
            if abs(z_rot_rad) > 1e-12:
                rot_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), z_rot_rad)
                # apply Z in world/table frame before base tilt
                final_rot = rot_z * base_rot
            else:
                final_rot = base_rot

            # Determine cloth placement exclusively from metadata/config
            if self.base_position_override is not None:
                cloth_pos = tuple(self.base_position_override)
            else:
                if "center_offset" not in self.mesh_metadata:
                    raise ValueError("No base_position override provided and metadata.center_offset is missing")
                co = self.mesh_metadata["center_offset"]
                if not (isinstance(co, (list, tuple)) and len(co) == 3):
                    raise ValueError("metadata.center_offset must be a length-3 list")
                cloth_pos = (float(co[0]), float(co[1]), float(co[2]))

            # expose base position for external consumers (e.g., trajectory generator)
            self.cloth_base_pos = wp.vec3(*cloth_pos)
            print(f"Final rotation {final_rot}")
            print(f"Cloth base position {self.cloth_base_pos}")

            # Load cloth parameters from metadata (required)
            cp = self.mesh_metadata.get("cloth_parameters", None)
            if not isinstance(cp, dict):
                raise ValueError("metadata.cloth_parameters missing; add_cloth_mesh parameters must come from metadata")
            required_keys = [
                "density","scale","tri_ke","tri_ka","tri_kd","edge_ke","edge_kd","particle_radius"
            ]
            for k in required_keys:
                if k not in cp:
                    raise ValueError(f"metadata.cloth_parameters.{k} missing")

            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=final_rot,
                pos=self.cloth_base_pos,
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=float(cp["density"]),
                scale=float(cp["scale"]),
                tri_ke=float(cp["tri_ke"]),
                tri_ka=float(cp["tri_ka"]),
                tri_kd=float(cp["tri_kd"]),
                edge_ke=float(cp["edge_ke"]),
                edge_kd=float(cp["edge_kd"]),
                particle_radius=float(cp["particle_radius"]),
            )

            self.scene.color()


        '''
        Added parallel builder
        '''
        builder = newton.ModelBuilder()
        
        # Compute environment offsets (same logic as replicate uses internally)
        spacing_cfg = _cfg("simulation.replicate_spacing", (3, 3, 0))
        if isinstance(spacing_cfg, (list, tuple)) and len(spacing_cfg) == 3:
            spacing = tuple(spacing_cfg)
        else:
            spacing = (3, 3, 0)

        # Use the public newton.utils.compute_world_offsets function.
        # This is the correct, portable way to get the offsets.
        self.env_offsets = newton.utils.compute_world_offsets(self.n_env, spacing, builder.up_axis)
        
        # Manually add each environment using the calculated offsets
        for i in range(self.n_env):
            offset_vec = self.env_offsets[i]
            xform = wp.transform(offset_vec, wp.quat_identity())
            builder.add_builder(self.scene, xform=xform)
        
        builder.add_ground_plane()

        self.model = builder.finalize(requires_grad=False)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver = None
        if self.add_cloth:
            # initialize cloth solver
            #   set edge rest angle to zero to disable bending, this is currently a workaround to make SolverVBD stable
            #   TODO: fix SolverVBD's bending issue
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                self_contact_radius=self.self_contact_radius,
                self_contact_margin=self.self_contact_margin,
                handle_self_contact=True,
                vertex_collision_buffer_pre_alloc=32,
                edge_collision_buffer_pre_alloc=64,
                integrate_with_external_rigid_solver=True,
                collision_detection_interval=-1,
            )

        self.viewer.set_model(self.model)

        # create Warp arrays for gravity so we can swap Model.gravity during
        # a simulation running under CUDA graph capture
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)  # used for the robot solver
        # gravity in m/s^2 (vector configurable)
        g_cfg = _cfg("simulation.gravity", None)
        if isinstance(g_cfg, (list, tuple)) and len(g_cfg) == 3:
            g_vec = wp.vec3(float(g_cfg[0]), float(g_cfg[1]), float(g_cfg[2]))
        else:
            g_vec = wp.vec3(0.0, 0.0, -9.81)
        self.gravity_earth = wp.array(g_vec, dtype=wp.vec3)  # used for the cloth solver

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_generated_trajectories(self, traj_list):
        """Enable per-environment generated trajectories.
        traj_list: List of dicts with keys grasp_pos, lift_pos, drop_pos, grasp_time, lift_time, drop_time, total_time
        Index in list must correspond to env_id.
        """
        self.generated_trajectories = traj_list

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        # for robot control
        self.delta_q = wp.empty(self.model.joint_count, dtype=float)
        self.joint_q_des = wp.array(self.model.joint_q.numpy(), dtype=float)

        # Pre-compile kernels for all end-effector body IDs to avoid repeated kernel compilation
        self.compute_body_out_kernels = {}
        for env_id in range(self.n_env):
            ee_body_id = self.endeffector_id + env_id * self.bodies_per_env
            
            # Create a kernel for this specific end-effector using centralized factory
            self.compute_body_out_kernels[ee_body_id] = dc_kernels.make_compute_body_out_kernel(
                self.endeffector_offset, ee_body_id
            )

        # Keep the original kernel for backward compatibility (uses first end-effector)
        self.compute_body_out_kernel = self.compute_body_out_kernels[self.endeffector_id]
        
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()
        # Buffer for current end-effector rotations per environment
        self.ee_rotations = wp.empty(self.n_env, dtype=wp.quat)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        # Define robot at origin (0, 0) so that all poses are naturally relative
        # When replicate() is called, each environment gets its own offset automatically
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                self.robot_base_position,
                (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi) if self.robot_flip_about_xy else wp.quat_identity()),
            ),
            floating=False,
            scale=1,  # unit: cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = self.gripper_closed
        clamp_open_activation_val = self.gripper_open

        # Define poses relative to robot origin (0, 0, 0)
        # With robot at (0,0) and table/cloth at (0.5, 0.0), adjust X by +0.5
        # These poses are now relative offsets that will work for all environments
        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position, 4D quaternion), gripper open (1) or closed (0)
                # top right
                [2, 0.64, -0.075, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2,   0.64, -0.075, 0.21, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2,   0.64, -0.075, 0.21, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2,   0.57, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3,  0.48, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1,  0.48, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [2,  0.59, 0.1275, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3,  0.59, 0.1275, 0.21, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3,  0.59, 0.1275, 0.21, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2,  0.59, 0.1275, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 0.48, 0.1275, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.48, 0.1275, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # top left
                [2, 0.36, -0.075, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.36, -0.075, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.36, -0.075, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.43, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3,  0.52, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1,  0.52, -0.075, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom left
                [3, 0.41, 0.15, 0.205, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 0.41, 0.15, 0.205, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.49, 0.15, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 0.49, 0.15, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.49, 0.15, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2,   0.5, 0.21, 0.30, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2,   0.5, 0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2,   0.5, 0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2,   0.5, 0.21, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1,   0.5, 0.12, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.5, 0.12, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.5, 0.05, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1,   0.5, 0.05, 0.35, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(self.endeff_vec, wp.quat_identity())

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q
        Uses the fixed self.endeffector_id (for backward compatibility)
        """
        return self.compute_body_jacobian_for_env(
            model, joint_q, joint_qd, self.endeffector_id, include_rotation
        )

    def compute_body_jacobian_for_env(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        body_id: int,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of a specific body's velocity related to joint_q
        This version accepts body_id as a parameter for multi-environment support
        Uses pre-compiled kernels for efficiency
        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        # Use the pre-compiled kernel for this body_id
        compute_body_out = self.compute_body_out_kernels[body_id]

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                compute_body_out, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        t_mod = (
            self.sim_time
            if self.sim_time < self.robot_key_poses_time[-1]
            else self.sim_time % self.robot_key_poses_time[-1]
        )
        include_rotation = self.include_rotation
        current_interval = np.searchsorted(self.robot_key_poses_time, t_mod)

        # Get all joint states as numpy array once
        all_joint_q = state_in.joint_q.numpy()
        all_joint_qd = state_in.joint_qd.numpy()
        
        # CRITICAL FIX #1: Read target_qd array ONCE before the loop to avoid overwriting
        target_qd_np = self.target_joint_qd.numpy()
        
        # Orientation policy: keep a fixed orientation for the entire trajectory (from config)
        # We still compute position targets per phase below.

        # Compute control for each environment separately
        for env_id in range(self.n_env):
            # Get the offset for this environment
            env_offset = self.env_offsets[env_id]
            
            # Determine per-env target (position, orientation, gripper)
            if self.generated_trajectories is not None:
                traj = self.generated_trajectories[env_id]
                # Segment durations
                g = float(traj["grasp_time"])      # move: lift -> grasp (descent)
                gd = float(traj.get("grasp_dwell", 0.0))  # dwell at grasp
                l = float(traj["lift_time"])       # move: grasp -> lift (ascend)
                d = float(traj["drop_time"])       # move: lift -> drop (transport)
                dd = float(traj.get("drop_dwell", 0.0))   # dwell at drop
                # Total time
                if "total_time" in traj:
                    T = float(traj["total_time"])
                else:
                    T = float(g + gd + l + d + dd)
                tloc = (self.sim_time % T) if T > 1e-6 else 0.0
                # Waypoints (support explicit pre/post if provided)
                grasp = np.asarray(traj.get("grasp_pos", traj.get("post_grasp_pos")), dtype=np.float32)
                pre_grasp = np.asarray(traj.get("pre_grasp_pos", grasp), dtype=np.float32)
                post_grasp = np.asarray(traj.get("post_grasp_pos", grasp), dtype=np.float32)
                lift = np.asarray(traj["lift_pos"], dtype=np.float32)
                drop = np.asarray(traj.get("drop_pos", traj.get("pre_drop_pos")), dtype=np.float32)
                pre_drop = np.asarray(traj.get("pre_drop_pos", drop), dtype=np.float32)
                post_drop = np.asarray(traj.get("post_drop_pos", drop), dtype=np.float32)
                # Phase boundaries: [0, g) descend; [g, g+gd) grasp dwell; [g+gd, g+gd+l) ascend;
                # [g+gd+l, g+gd+l+d) transport; [g+gd+l+d, T) drop dwell
                t1 = g
                t2 = g + gd
                t3 = t2 + l
                t4 = t3 + d
                if tloc < t1:
                    # Descend: move from lift to pre-grasp, gripper open
                    a = (tloc / g) if g > 1e-6 else 1.0
                    p = lift * (1.0 - a) + pre_grasp * a
                    grip = 0.8  # open
                elif tloc < t2:
                    # Grasp dwell: at post-grasp, transition to closed
                    p = post_grasp
                    grip = 0.03  # closed
                elif tloc < t3:
                    # Ascend: move from grasp back to lift, gripper closed
                    a = ((tloc - t2) / l) if l > 1e-6 else 1.0
                    p = post_grasp * (1.0 - a) + lift * a
                    grip = 0.03  # closed
                elif tloc < t4:
                    # Transport: move from lift to drop, gripper closed
                    a = ((tloc - t3) / d) if d > 1e-6 else 1.0
                    p = lift * (1.0 - a) + pre_drop * a
                    grip = 0.03  # closed
                else:
                    # Drop dwell: at post-drop, gripper open
                    p = post_drop
                    grip = 0.8  # open
                # positions from generator are absolute world; no extra env offset
                target_pos = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
                # Keep the same orientation for the entire trajectory: use the configured "down" quaternion
                dq = self.down_quat
                target_quat = wp.quat(float(dq[0]), float(dq[1]), float(dq[2]), float(dq[3]))
                target_grip = grip
            else:
                # Create target transform with environment offset applied using built-in key poses
                target_pos = wp.vec3(
                    self.targets[current_interval][0] + env_offset[0],
                    self.targets[current_interval][1] + env_offset[1],
                    self.targets[current_interval][2] + env_offset[2],
                )
                target_quat = wp.quat(*self.targets[current_interval][3:7])
                target_grip = float(self.targets[current_interval][-1])

            target_transform = wp.transform(target_pos, target_quat)
            
            # Extract this environment's joint states
            q_start = env_id * self.dof_q_per_env
            q_end = q_start + self.dof_q_per_env
            qd_start = env_id * self.dof_qd_per_env
            qd_end = qd_start + self.dof_qd_per_env
            
            q = all_joint_q[q_start:q_end]
            qd = all_joint_qd[qd_start:qd_end]
            
            # CRITICAL FIX #2: Compute end-effector error for THIS environment's end-effector
            # The end-effector body ID needs to account for environment offset
            ee_body_id = self.endeffector_id + env_id * self.bodies_per_env
            
            wp.launch(
                dc_kernels.compute_ee_delta,
                dim=1,
                inputs=[
                    state_in.body_q,
                    self.endeffector_offset,
                    ee_body_id,  # Use the correct end-effector for this environment
                    target_transform,
                ],
                outputs=[self.ee_delta],
            )
            
            delta_target = self.ee_delta.numpy()[0]

            # CRITICAL FIX #3: Compute Jacobian for THIS environment's end-effector
            # We need to pass the correct body_id for this environment
            self.compute_body_jacobian_for_env(
                self.model,
                state_in.joint_q,
                state_in.joint_qd,
                ee_body_id,
                include_rotation=include_rotation,
            )
            
            # Extract Jacobian columns for this environment only
            J_full = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
            J = J_full[:, qd_start:qd_end]  # Get only this environment's DOF columns
            
            J_inv = np.linalg.pinv(J)

            if self.use_nullspace:
                # 2. Compute null-space projector
                #    I is size [num_joints x num_joints] for THIS environment
                I = np.eye(J.shape[1], dtype=np.float32)
                N = I - J_inv @ J

                # 3. Define a desired "elbow-up" reference posture for this environment
                q_des = q.copy()
                # Extract this environment's portion of the initial pose
                initial_pose_for_env = self.initial_pose[q_start:q_end]
                q_des[1:] = initial_pose_for_env[1:]  # keep joints near initial safe pose

                # 4. Define a null-space velocity term pulling joints toward q_des
                delta_q_null = self.k_null * (q_des - q)

                # 5. Combine primary task and null-space controller
                delta_q = J_inv @ delta_target + N @ delta_q_null
            else:
                delta_q = J_inv @ delta_target

            # Apply gripper finger control
            delta_q[-2] = target_grip * self.gripper_scale - q[-2]
            delta_q[-1] = target_grip * self.gripper_scale - q[-1]

            # Optional clamp on joint velocities for stability/safety
            if self.max_joint_qd is not None and np.isfinite(self.max_joint_qd):
                delta_q = np.clip(delta_q, -self.max_joint_qd, self.max_joint_qd)

            # CRITICAL FIX #1 (continued): Update the shared array for this environment
            target_qd_np[qd_start:qd_end] = delta_q
        
        # CRITICAL FIX #1 (final): Assign the full array ONCE after all environments are computed
        self.target_joint_qd.assign(target_qd_np)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        use_rerun = False

        if use_rerun:
            self.viewer = newton.viewer.ViewerRerun(
                server=True,                    # Start in server mode
                address="127.0.0.1:9876",      # Server address
                launch_viewer=True,            # Auto-launch web viewer
                app_id="newton-simulation"     # Application identifier
            )

            self.viewer.set_model(self.model)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()



    def test(self):
        p_lower = wp.vec3(-0.34, -0.9, 0.0)
        p_upper = wp.vec3(0.34, 0.0, 0.51)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )

    def reset(self):
        """Reset the simulation to its initial state."""
        self.sim_time = 0.0
        
        # Reset states to initial configuration
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        
        # Reset control and contacts
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        # Reset robot to initial joint configuration
        if self.add_robot:
            # Re-evaluate forward kinematics with initial joint positions
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Recapture CUDA graph if needed
        if self.add_cloth:
            self.capture()

    # removed run_iteration and run methods; this example is imported and stepped externally
                
if __name__ == "__main__":
    # Intentionally left empty: this module is now imported and driven externally.
    pass
