# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
from newton.utils import transform_twist


# Note: Warp requires concrete type annotations on kernel parameters.
# We use wp.array(dtype=...) directly in annotations for runtime correctness.


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    """
    Compute the 6D pose delta (translation xyz, rotation as quaternion components wxyz)
    between the current end-effector (body_id with offset applied) and a target transform.

    Notes:
    - body_id should already include any environment-specific offset.
    - The result is written to ee_delta[0].
    """
    tf = body_q[body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[0] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


def make_compute_body_out_kernel(offset: wp.transform, body_id: int):
    """
    Factory that returns a Warp kernel specialized to a particular body_id and offset.

    The kernel maps body spatial velocity at the given body_id through the offset using
    transform_twist, writing a 6D spatial velocity into body_out.
    """

    @wp.kernel
    def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
        mv = transform_twist(wp.static(offset), body_qd[wp.static(body_id)])
        for i in range(6):
            body_out[i] = mv[i]

    return compute_body_out


@wp.kernel
def extract_ee_positions_kernel(
    body_q: wp.array(dtype=wp.transform),
    ee_offset: wp.transform,
    ee_body_id: int,
    bodies_per_env: int,
    out_pos: wp.array(dtype=wp.vec3),
):
    """
    Per-env kernel: for thread env_id, reads body transform at ee_body_id + env_id * bodies_per_env,
    applies ee_offset, and writes the translation to out_pos[env_id].
    """
    env_id = wp.tid()
    idx = ee_body_id + env_id * bodies_per_env
    tf = body_q[idx] * ee_offset
    out_pos[env_id] = wp.transform_get_translation(tf)


@wp.kernel
def extract_ee_rotations_kernel(
    body_q: wp.array(dtype=wp.transform),
    ee_offset: wp.transform,
    ee_body_id: int,
    bodies_per_env: int,
    out_rot: wp.array(dtype=wp.quat),
):
    """
    Per-env kernel: for thread env_id, reads body transform at ee_body_id + env_id * bodies_per_env,
    applies ee_offset, and writes the rotation quaternion to out_rot[env_id] (wxyz order).
    """
    env_id = wp.tid()
    idx = ee_body_id + env_id * bodies_per_env
    tf = body_q[idx] * ee_offset
    out_rot[env_id] = wp.transform_get_rotation(tf)
