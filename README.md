# PI-R2-Flow

Code release for **PI-R2: Reactive Real-time Flow Policies** (https://arxiv.org/abs/2607.26055).

This repo contains two components:
- `deployment/` — real-time policy deployment stack for xArm6 + XHand with GR00T
- `learning/` — GR00T-N1.7 training code for all three variants (fork of NVIDIA Isaac-GR00T; see [Training](#training-learning))

---

## Repo Structure

```
pi-r2-flow/
├── deployment/
│   ├── apps/
│   │   ├── run_policy.py         # main control loop
│   │   ├── _policy_args.py       # all CLI flags
│   │   └── run_camera_server.py  # RealSense ZMQ server
│   ├── mindex/
│   │   ├── robots/
│   │   │   ├── xarm_sdk.py       # xArm6 driver
│   │   │   └── xhand_robot.py    # XHand driver
│   │   ├── cameras/realsense.py  # camera server/client
│   │   ├── policy/
│   │   │   ├── groot_client.py   # GR00T ZMQ client
│   │   │   └── control_utils.py  # interpolation, ensembling
│   │   └── recording/dataset.py  # eval log writer (HDF5)
│   └── scripts/
│       ├── render_episode_modalities.py
│       └── render_finger_focus.py
└── learning/
    └── Isaac-GR00T/              # GR00T-N1.7 training (submodule: fork of NVIDIA Isaac-GR00T @ pir2)
```

---

## Overview

We deploy a GR00T Vision-Language-Action (VLA) policy on a dexterous manipulation system (6-DOF arm + 12-DOF dexterous hand). 

We support three policy variants:

| Variant | Checkpoint type | Query mode |
|---|---|---|
| **PI-R2 (Ours)** | `pir2` | continuous or pipelined |
| **Train-time RTC** | `rtc` | pipelined |
| **Standard flow** | `plain_flow` | sync, or continuous + temporal ensembling |

**Hardware:**
- Arm: UFactory xArm6
- Hand: XHand (12-DOF, RS485)
- Camera: Intel RealSense (overhead)
- Inference: remote GPU running GR00T

**Action space:** 18-dim = xArm6 joints `[0:6]` + XHand joints `[6:18]`

---

## Installation

Clone with `--recursive` to get the Isaac-GR00T submodule:

```bash
git clone --recursive https://github.com/pi-r2-flow/pi-r2-flow.git
# or, if already cloned:
git submodule update --init --recursive
```

**Deployment stack** (robot machine):

```bash
cd deployment
pip install -e .
pip install -e ".[camera]"   # RealSense support
```

---

## Training (`learning/`)

`learning/Isaac-GR00T` is a fork of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
(pinned to the `pir2` branch) that adds PI-R2 to the GR00T-N1.7 flow-matching action head.

Install per the Isaac-GR00T README and download the `GR00T-N1.7-3B` base checkpoint. Training data
uses the standard GR00T (LeRobot) format. All three variants share one entrypoint and are selected
by flags:

```bash
cd learning/Isaac-GR00T
torchrun --nproc_per_node=8 gr00t/experiment/launch_finetune.py \
    --base-model-path      <path/to/GR00T-N1.7-3B> \
    --dataset-path         <path/to/lerobot_dataset> \
    --modality-config-path <path/to/modality_config.py> \
    --embodiment-tag NEW_EMBODIMENT --num-gpus 8 \
    --global-batch-size 512 --max-steps 40000 \
    --output-dir <path/to/output> \
    <VARIANT FLAGS>       # see table
```

| Variant | `<VARIANT FLAGS>` |
|---|---|
| **Standard flow** | *(none)* |
| **Train-Time RTC** | `--streaming --streaming-constant-weight 0.0 --streaming-chunk-wise-weight 0.0 --streaming-rtc-weight 1.0 --streaming-rtc-d-max 10 --streaming-mask-clean-end` |
| **PI-R2 (Ours)** | `--streaming --streaming-constant-weight 0.2 --streaming-chunk-wise-weight 0.8 --streaming-schedule-mode pir2 --streaming-chunk-size-max 5 --streaming-mask-clean-end --image-delay-max 5 --image-delay-embed-dim 64` |

For PI-R2, also set `export GR00T_IMAGE_DELAY_MAX=5` in the environment.

---

## Deployment

Start the GR00T inference server on your GPU machine ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)):

```bash
# Single GPU (baselines)
python gr00t/eval/run_gr00t_server.py --model-path <checkpoint> --port 5555 --host 0.0.0.0

# 2-GPU split (VLM on port 5555, DiT on port 5556) — used with --async-vlm for PI-R2
MODEL_PATH=<checkpoint> bash scripts/gr00t_inference_2gpu.sh
```

If the inference server is on a remote cluster, forward the ports via SSH before running:

```bash
ssh -N -L 5555:<compute-node>:5555 <username>@<cluster-login>
ssh -N -L 5556:<compute-node>:5556 <username>@<cluster-login>
```

Start the camera server on the robot machine:

```bash
python deployment/apps/run_camera_server.py --port 5000
```

### Key Parameters

**`--query-mode {sync,pipelined,continuous}`**: How the policy queries the GR00T server.

- **`sync`** — Simplest. The robot freezes (holds last command) while waiting for each inference result. Inference latency shows up as a visible pause between executed chunks. Good for debugging.
- **`pipelined`** — Starts the next inference partway through executing the current chunk, timed so the result arrives just as the current sub-chunk (length specified with `--chunk-len`) runs out. The robot keeps moving during inference — no freeze. Requires `--chunk-len` ≥ expected inference latency in steps.
- **`continuous`** — A worker thread queries back-to-back as fast as inference allows. The freshest chunk is swapped in as soon as it lands and the current chunk is consumed. Lowest effective latency; highest query rate. Used for PI-R2 (Ours).

**`--chunk-len N`**: How many actions to execute from each predicted chunk before re-querying. Lower = more frequent re-queries (more responsive to new observations) but higher sensitivity to inference latency.

**`--nfe N`**: Number of denoising steps (function evaluations) in the flow.

### Running the Policy

#### Ours (PI-R2-Flow) — `pir2`, continuous or pipelined

```bash
# Continuous (re-queries as fast as inference allows; recommended)
python deployment/apps/run_policy.py \
    --task "catch the book" \
    --arm-mode absolute --hand-mode absolute \
    --query-mode continuous \
    --host localhost --port 5556 \
    --chunk-len 2 --nfe 24 --ckpt-type pir2 \
    --vlm-host localhost --vlm-port 5555 --async-vlm \
    --eval-log eval_logs/ours

# Pipelined (kicks off next query when current chunk is ~consumed)
python deployment/apps/run_policy.py \
    --task "catch the book" \
    --arm-mode absolute --hand-mode absolute \
    --query-mode pipelined \
    --host localhost --port 5556 \
    --chunk-len 2 --nfe 24 --ckpt-type pir2 \
    --vlm-host localhost --vlm-port 5555 --async-vlm \
    --eval-log eval_logs/ours_pipelined
```

#### RTC — `rtc`, pipelined

RTC checkpoints are trained with clean-action inpainting: the front `d` slots of the denoising buffer are overwritten with the actions the robot actually executed during inference. Use `--inpaint --force-nonstreaming`. Set `--chunk-len` ≥ measured latency (e.g., 5) to give the model time to complete inference before the next swap.

```bash
python deployment/apps/run_policy.py \
    --task "put the box in the basket" \
    --arm-mode absolute --hand-mode absolute \
    --query-mode pipelined \
    --host localhost --port 5556 \
    --chunk-len 5 --nfe 4 --ckpt-type rtc \
    --vlm-host localhost --vlm-port 5555 \
    --inpaint --force-nonstreaming \
    --eval-log eval_logs/rtc
```

#### Standard Flow — `plain_flow`, sync or continuous + temporal ensembling

```bash
# Sync (simplest; robot freezes during each query)
python deployment/apps/run_policy.py \
    --task "put the box in the basket" \
    --arm-mode absolute --hand-mode absolute \
    --query-mode sync \
    --host localhost --port 5556 \
    --chunk-len 10 --nfe 4 --ckpt-type plain_flow \
    --force-nonstreaming \
    --eval-log eval_logs/flow_sync

# Continuous + temporal ensembling (ACT-style weighted averaging over chunks)
python deployment/apps/run_policy.py \
    --task "put the box in the basket" \
    --arm-mode absolute --hand-mode absolute \
    --query-mode continuous \
    --host localhost --port 5556 \
    --chunk-len 25 --nfe 4 --ckpt-type plain_flow \
    --ensemble \
    --eval-log eval_logs/flow_async
```

### Visualizing Eval Logs

```bash
# Full modality panel (joint torque, fingertip force, proprio, action)
python deployment/scripts/render_episode_modalities.py \
    ~/eval_logs/catch_the_book/episode_0000.hdf5

# Per-finger focus (camera + fingertip force + key DOF)
python deployment/scripts/render_finger_focus.py \
    ~/eval_logs/catch_the_book/episode_0000.hdf5 --fingers thumb index
```

---

## Extending to a Different Robot

The deployment stack is robot-agnostic. Each driver implements a minimal interface:

```python
class YourRobot:
    num_dofs: int

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def get_observation(self, fresh: bool = False) -> dict | None:
        # Returns at minimum:
        # {"joint_positions": np.ndarray (num_dofs,)}
        # Optionally: {"joint_torque": np.ndarray (num_dofs,)}
        ...

    def send_action(self, action: np.ndarray) -> None:
        # action: (num_dofs,) joint angles in radians
        ...
```

See `deployment/mindex/robots/xarm_sdk.py` and `xhand_robot.py` for reference implementations.

**Steps:**

1. **Write a robot interface** following the interface above.

2. **Swap it in `run_policy.py`** (section 1, CONNECT):
```python
# Replace the xArm/XHand imports with your driver:
from your_package import YourRobot
arm = YourRobot(...)
```

3. **Adjust the action space.** The policy outputs 18-dim by default (`ARM_DOFS=6` + 12 hand DOF). Update `ARM_DOFS` in `run_policy.py` to match your robot, and retrain (or fine-tune) the GR00T checkpoint on data from your setup.

4. **Match the checkpoint.** Ensure the checkpoint was trained with the same action mode (`absolute`/`delta`), DOF count, and observation modalities as your deployment config.

---

## Citation

```bibtex
@misc{park2026pir2,
  title  = {{\pi}R^2: Reactive Real-time Flow Policies},
  author = {Park, Sungjae and Tulsiani, Shubham},
  year   = {2026}
}
```
