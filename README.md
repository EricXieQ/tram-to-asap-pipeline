# TRAM-to-ASAP Pipeline

End-to-end pipeline for training a Unitree G1 humanoid robot to imitate human motions captured from video.

**Film a video of a person doing a motion** -> **Extract 3D human pose (TRAM)** -> **Convert to robot format (ASAP)** -> **Train RL policy** -> **Deploy to real G1 robot**

Built on top of [TRAM](https://github.com/yufu-wang/TRAM) and [ASAP](https://github.com/LeCAR-Lab/ASAP).

---

## Pipeline Overview

```
Video (mp4)
  |
  v  Step 1: TRAM (4D human capture)
hps_track_0.npy  -- rotation matrices (N, 24, 3, 3), camera-space translation
  |
  v  Step 2: convert_tram_to_asap.py
*_amass.npz  -- axis-angle poses (N, 72), world-space translation, AMASS-compatible
  |
  v  Step 3: fit_smpl_motion.py
*.pkl  -- retargeted to G1 robot skeleton (23 DOF)
  |
  v  Step 4: ASAP training (train_agent.py)
model_XXXX.pt / model_XXXX.onnx  -- trained RL policy
  |
  v  Step 5: Deploy to real G1 (sim2real/)
Unitree G1 robot performs the motion
```

---

## Prerequisites

- **Hardware:** NVIDIA GPU (tested on RTX 4090), Unitree G1 robot (for deployment)
- **Software:**
  - Conda/Mamba
  - TRAM environment (Python 3.10)
  - ASAP/HumanoidVerse environment (Python 3.8)
  - IsaacGym
  - Blender (optional, for visualization)

## Environment Setup

```bash
# TRAM environment
conda create -n tram python=3.10
conda activate tram
cd tram && pip install -e .

# ASAP environment
conda create -n hvgym python=3.8
conda activate hvgym
cd ASAP && pip install -e . && pip install -r requirements.txt
pip install -e isaac_utils
```

---

## Step 1: Extract 3D Human Pose from Video (TRAM)

TRAM takes a video and outputs SMPL body parameters for each tracked person.

```bash
conda activate tram
cd tram

# Estimate camera motion (use --static_camera for tripod shots)
python scripts/estimate_camera.py --video /path/to/video.mp4 --static_camera

# Estimate human pose
python scripts/estimate_humans.py --video /path/to/video.mp4

# (Optional) Visualize results
python scripts/visualize_tram.py --video /path/to/video.mp4
```

**Output:** `tram/results/<video_name>/hps/hps_track_0.npy`

Contains:
| Key | Shape | Description |
|-----|-------|-------------|
| `pred_rotmat` | (N, 24, 3, 3) | Rotation matrices for 24 SMPL joints |
| `pred_shape` | (N, 10) | Body shape parameters (betas) per frame |
| `pred_trans` | (N, 1, 3) | Translation in camera space |
| `pred_cam` | (N, 3) | Weak-perspective camera parameters |
| `frame` | (N,) | Frame indices |

**Tips for filming:**
- Use a tripod (static camera) when possible and use the `--static_camera` flag
- Ensure good background texture (brick walls, bookshelves -- not blank white walls)
- Don't let the person fill the entire frame -- leave background visible

---

## Step 2: Convert TRAM Output to AMASS Format

Converts TRAM's rotation matrices + camera coordinates to ASAP-compatible axis-angle poses + world coordinates.

```bash
conda activate tram
cd tram
python convert_tram_to_asap.py <video_name> --fps 30 --smooth 5
```

**What this script does:**

1. **Rotation format conversion:** Rotation matrices (3x3) -> axis-angle (3 values) using:
   - Angle: `cos(theta) = (trace(R) - 1) / 2`
   - Axis: `k = 1/(2*sin(theta)) * [R32-R23, R13-R31, R21-R12]`
   - Result: `axis_angle = axis * theta`

2. **Coordinate system fix:** Camera coords (Y-down) -> world coords (Z-up) via 90-degree rotation around X-axis

3. **Translation:** X and Y are zeroed out to prevent drift (keeps Z height). For locomotion motions, you may want to preserve X/Y -- see [Known Issues](#known-issues).

4. **Temporal smoothing:** Applies `uniform_filter1d` to reduce TRAM's frame-to-frame jitter

5. **Shape averaging:** Per-frame betas (N, 10) -> single averaged vector (16,)

**Output:** `ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/<name>_amass.npz`

Contains:
| Key | Shape | Description |
|-----|-------|-------------|
| `poses` | (N, 72) | Axis-angle, 24 joints x 3 |
| `betas` | (16,) | Averaged body shape |
| `trans` | (N, 3) | World-space translation |
| `gender` | string | "neutral" |
| `mocap_framerate` | int | 30 |

---

## Step 3: Retarget to G1 Robot Skeleton

Maps the 24 SMPL joints to the G1 robot's 23 actuated degrees of freedom via IK optimization.

```bash
conda activate hvgym
cd ASAP
python scripts/data_process/fit_smpl_motion.py +robot=g1/g1_29dof_anneal_23dof
```

This processes all `.npz` files in `raw_tairantestbed_smpl/` and runs 1000 iterations of optimization per motion to match SMPL joint positions to the robot's kinematic structure.

**Prerequisites:**
- Shape prior must exist: `humanoidverse/data/shape/g1_29dof_anneal_23dof/shape_optimized_v1.pkl`
- If missing, generate it: `python scripts/data_process/fit_smpl_shape.py +robot=g1/g1_29dof_anneal_23dof`

**Output:** `ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/<name>.pkl`

Contains:
| Key | Shape | Description |
|-----|-------|-------------|
| `root_trans_offset` | (T, 3) | Root position trajectory |
| `pose_aa` | (T, 27, 3) | Axis-angle for all bodies |
| `dof` | (T, 23) | Actuated joint positions |
| `root_rot` | (T, 4) | Root rotation quaternion |
| `smpl_joints` | (T, 24, 3) | Reference SMPL joint positions |
| `fps` | int | 30 |

---

## Step 4: Train RL Policy

Train the G1 robot to track the retargeted motion in IsaacGym.

```bash
conda activate hvgym
cd ASAP

python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=4096 \
  headless=True \
  project_name=MotionTracking \
  experiment_name=<your_experiment_name> \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/<your_motion>.pkl" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_penalty_degree=0.00001 \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=True \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  robot.asset.self_collisions=0
```

**Training tips:**
- Use `num_envs=4096` or higher -- more environments = better gradient estimates = faster learning
- Use `headless=True` for faster training, `headless=False` to watch
- RTX 4090 (24GB) comfortably handles 4096 envs (~5.8GB VRAM)
- Monitor with TensorBoard: `tensorboard --logdir=logs/<project_name>/` (in a separate terminal)

**Key metrics to monitor:**
- `Train/mean_reward` -- should increase over time
- `Train/mean_episode_length` -- should increase until it matches the motion clip length
- `Env/upper_body_diff_norm` -- upper body tracking error (lower = better)
- `Env/joint_pos_diff_norm` -- joint position error (lower = better)
- `rew_termination` -- how often the robot falls (closer to 0 = better)

Training typically converges when the reward plateaus and the episode length matches the motion clip duration.

**Evaluate the trained policy:**
```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/hvgym/lib:$LD_LIBRARY_PATH
python humanoidverse/eval_agent.py +checkpoint=logs/<path_to>/model_XXXX.pt
```

---

## Step 5: Deploy to Real G1 Robot

> **Warning:** Always test with the robot suspended in a harness/crane first.

The trained policy is automatically exported as ONNX (`exported/model_XXXX.onnx`). The deployment pipeline uses ROS2 + Unitree SDK2 to send joint commands to the real robot.

**Requirements:**
- ROS2 Humble
- Unitree SDK2
- ONNX Runtime

**Deployment code:** `ASAP/sim2real/`

**Safety procedure:**
1. Hang the G1 from a crane/harness
2. Power on the robot
3. Run the policy with the robot suspended -- verify joints move correctly
4. If it looks good, slowly lower until feet touch the ground
5. Gradually let it bear its own weight with the harness as backup
6. Keep a hand on the emergency stop at all times

---

## Optional: Visualize Motions in Blender

You can visualize the AMASS `.npz` files in Blender at any point to verify the motion looks correct.

```bash
# Step 1: Generate mesh sequences
conda activate tram
cd mosh_work
python convert_amass_to_meshes.py

# Step 2: Create Blender files
~/blender-5.0.1-linux-x64/blender --background --python make_blend.py -- <motion_name>
```

Output: `mosh_work/blender_files/<motion_name>.blend`

---

## Optional: Visualize in MuJoCo

```bash
conda activate hvgym
cd ASAP
python scripts/vis/vis_q_mj.py \
  +robot=g1/g1_29dof_anneal_23dof \
  visualize_motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/<your_motion>.pkl"
```

---

## Data Format Comparison

| | TRAM output (.npy) | AMASS intermediate (.npz) | ASAP input (.pkl) |
|---|---|---|---|
| **Rotations** | Rotation matrices (N, 24, 3, 3) | Axis-angle (N, 72) | Axis-angle (T, J, 3) |
| **Shape** | Per-frame betas (N, 10) | Single beta vector (16,) | Single beta vector |
| **Translation** | Camera-space (N, 1, 3) | World-space (N, 3) | World-space root_trans_offset (T, 3) |
| **Coordinate system** | Camera (Y-down) | World (Z-up) | World (Z-up) |
| **Skeleton** | SMPL (24 joints) | SMPL (24 joints) | G1 robot (23 DOF) |

---

## Known Issues

- **Translation drift:** TRAM's translation estimate can drift, especially with handheld cameras. Use `--static_camera` for tripod shots. For standing motions, X/Y are zeroed out in the conversion. For locomotion, this zeroing needs to be disabled, and physics-based motion cleaning (e.g., MaskedMimic/ProtoMotions) is recommended instead.

- **VIMO forward shift:** When a person extends arms toward the camera (e.g., during a dab), TRAM may interpret this as the person moving forward. This is a limitation of TRAM's weak-perspective to full-perspective conversion.

- **Short clips:** Very short clips (< 2 seconds) may not have enough frames for effective training.

---

## References

- [ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills](https://github.com/LeCAR-Lab/ASAP) (RSS 2025)
- [TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](https://github.com/yufu-wang/TRAM) (arXiv 2024)
- [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/)
- [MaskedMimic / ProtoMotions](https://github.com/NVlabs/ProtoMotions) (for physics-based motion cleaning)
