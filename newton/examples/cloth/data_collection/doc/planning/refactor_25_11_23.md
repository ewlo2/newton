This is a very pragmatic approach. You are right—you don't need to inherit from `gym.vector.VectorEnv` if it adds unnecessary boilerplate or dependencies. You just need the **architectural clarity** that Gym provides.

To achieve modularity (swapping robots) and LeRobot-style clarity (separation of concerns) while maintaining Newton's GPU performance, I recommend a **"Component-Based Vectorized Architecture."**

Here is the blueprint for refactoring your codebase.

### 1. The Directory Structure

This structure tells a new developer exactly where to look for specific logic.

```text
newton_cloth/
├── core/
│   ├── base_env.py          # Abstract base class (defines the interface)
│   └── base_robot.py        # Abstract robot class (defines how robots load/act)
├── envs/
│   └── cloth_manipulation.py # The specific task (Table, Cloth, Rewards)
├── robots/
│   ├── franka.py            # Franka specifics (URDF, Home Pose, IK Kernels)
│   └── ur5.py               # (Future) Easy to swap in
├── controllers/
│   ├── diff_ik.py           # Generic Differential IK logic
│   └── osc.py               # (Future) Operational Space Control
├── policies/
│   └── scripted_trajectory.py # High-level logic (Grasp -> Lift -> Drop)
└── collect_data.py          # The main entry point
```

---

### 2. The Robot Abstraction (Solving the "Hardcoded" Problem)

Currently, your IK kernels and joint indices are hardcoded in the environment. We move this into a Robot class.

**File:** `core/base_robot.py`
```python
class BaseRobot:
    def __init__(self, name, base_pose):
        self.name = name
        self.base_pose = base_pose
        self.dof = 0
        self.body_ids = []

    def load_into_builder(self, builder):
        """Adds URDF to the Newton ModelBuilder."""
        raise NotImplementedError

    def get_end_effector_pose(self, state, env_ids):
        """Returns batched EE pose (Warp array)."""
        raise NotImplementedError

    def compute_jacobian(self, state, env_ids):
        """Returns batched Jacobian."""
        raise NotImplementedError
```

**File:** `robots/franka.py`
```python
from newton_cloth.core.base_robot import BaseRobot
import newton
import warp as wp
# Import your specific kernels here

class Franka(BaseRobot):
    def __init__(self, ...):
        super().__init__(...)
        self.ee_link_name = "panda_hand"
        # Pre-compile the specific kernels for Franka here if needed

    def load_into_builder(self, builder):
        # Your existing URDF loading logic
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(...)
        self.dof = 7 # + gripper

    def get_end_effector_pose(self, state):
        # Wraps your `extract_ee_positions_kernel`
        # Returns a Warp array of transforms
        pass
```

---

### 3. The Controller (The Link between Policy and Robot)

The Policy says "Go Here." The Controller says "Move Joints Like This." This allows you to swap robots without rewriting the policy.

**File:** `controllers/diff_ik.py`
```python
class BatchDiffIKController:
    def __init__(self, robot, config):
        self.robot = robot
        self.k_null = config.k_null

    def compute_controls(self, state, target_ee_poses):
        """
        Args:
            state: The full Newton state
            target_ee_poses: (N_envs, 7) Warp array [x,y,z, qx,qy,qz,qw]
        Returns:
            joint_velocities: (N_envs, DOF) Warp array
        """
        # 1. Get current EE pose (Delegate to Robot)
        current_ee = self.robot.get_end_effector_pose(state)
        
        # 2. Compute Jacobian (Delegate to Robot)
        # This calls `compute_body_jacobian_for_env` internally
        J = self.robot.compute_jacobian(state)
        
        # 3. Solve J * qd = V_err (Generic Math)
        # This logic is robot-agnostic! It just does the math.
        joint_vels = self._solve_damped_least_squares(J, current_ee, target_ee_poses)
        
        return joint_vels
```

---

### 4. The Environment (The World Manager)

The Environment brings it all together. It is **Vectorized by default**.

**File:** `envs/cloth_manipulation.py`
```python
class VectorizedClothEnv:
    def __init__(self, config, robot_cls=Franka):
        self.n_env = config.n_env
        self.builder = newton.ModelBuilder()
        
        # 1. Init Robot (Modular!)
        self.robot = robot_cls(base_position=config.robot_pos)
        
        # 2. Build Scene (Batched)
        self._build_scene()
        
        # 3. Init Controller
        self.controller = BatchDiffIKController(self.robot, config.control)

    def _build_scene(self):
        # Calculate offsets
        # Call self.robot.load_into_builder() N times with offsets
        # Load cloth N times
        self.model = self.builder.finalize()

    def step(self, target_poses, gripper_actions):
        """
        Input: High-level Cartesian Targets (from Policy)
        """
        # 1. Use Controller to get low-level joint actions
        # Note: We pass the WHOLE state to the controller
        joint_actions = self.controller.compute_controls(self.state_0, target_poses)
        
        # 2. Apply to Simulation
        self.state_0.joint_qd.assign(joint_actions)
        # Apply gripper actions...

        # 3. Step Physics (Graph Capture)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.sim_step()

        # 4. Return Observations
        return self._get_obs()

    def _get_obs(self):
        # Return dict of Warp arrays: {'img': ..., 'q': ...}
        pass
```

---

### 5. The Main Loop (Clean & Readable)

This is what new developers will see. It reads almost like English.

**File:** `collect_data.py`
```python
from envs.cloth_manipulation import VectorizedClothEnv
from robots.franka import Franka
from robots.ur5 import UR5  # Look how easy it is to import a different robot
from policies.scripted_trajectory import BatchScriptedPolicy

def main():
    # 1. Setup
    # Easy to swap robot class here
    env = VectorizedClothEnv(config, robot_cls=Franka) 
    policy = BatchScriptedPolicy(env.n_env, config.traj)
    
    # 2. Reset
    obs = env.reset()
    policy.reset() # Generates N random trajectories on GPU
    
    # 3. Loop
    for step in range(config.max_steps):
        # A. Policy: "Where should the gripper be at time t?"
        # Output: (N, 7) Warp Array
        target_pose = policy.get_target_pose(step, obs)
        
        # B. Gripper Logic
        gripper_action = policy.get_gripper_state(step)

        # C. Environment: Handles IK, Physics, Rendering
        next_obs = env.step(target_pose, gripper_action)
        
        # D. Data Collection (Batch Save)
        # save_to_disk(obs, target_pose)
        
        obs = next_obs

if __name__ == "__main__":
    main()
```

### Why this works for you

1.  **Feasible in Newton:** We are never breaking the GPU data stream. The "Policy" and "Controller" just output Warp arrays that flow into `env.step`.
2.  **Modular Robots:** The `env` doesn't know it's controlling a Franka. It just asks `self.robot` for the Jacobian and sends it to the generic `DiffIKController`. If you add a UR5, you just write `robots/ur5.py` and implement the abstract methods.
3.  **Clean "LeRobot" Feel:** The main loop separates the *intention* (Policy) from the *execution* (Env/Controller).
4.  **No `gym` overhead:** You aren't forced to return `(obs, reward, done, info)` tuples if you don't want to. You define the API that makes sense for your data collector.

### Migration Strategy

1.  **Extract `Franka` class:** Take the `create_articulation` and `compute_body_jacobian` logic from `cloth_franka_env.py` and put it into `robots/franka.py`.
2.  **Create `BatchDiffIKController`:** Extract the solver logic (pseudo-inverse, nullspace projection) from `generate_control_joint_qd` into `controllers/diff_ik.py`.
3.  **Simplify Env:** Rewrite `Example` to instantiate the Robot and Controller, removing the hardcoded logic.
4.  **Update Main:** Update `example_collect_dataset.py` to drive the new API.