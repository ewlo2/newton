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
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
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
import time


"""
Note: kernels were moved to newton.examples.cloth.data_collection.kernels
to consolidate Warp kernels in a single place.
"""


class Example:
    def __init__(self, viewer, n_env=4, mesh_file: Optional[str] = None, z_rotation: Optional[float] = None):
        # parameters
        #   simulation
        self.add_cloth = True
        self.add_robot = True
        """
        Deprecated shim module.

        This file has been renamed to `cloth_franka_env.py` for clarity.
        Please import `Example` from `newton.examples.cloth.data_collection.cloth_franka_env`.
        """

        from .cloth_franka_env import Example  # re-export for backward compatibility

        __all__ = ["Example"]
        #   contact

        #       body-cloth contact
