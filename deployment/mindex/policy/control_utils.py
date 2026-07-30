"""Pure helpers for the policy deployment loop in `apps/run_policy.py`.

Extracted here so the main loop file stays focused on orchestration. These
functions don't touch loop-internal state (locks, buffers, args) — everything
they need is passed in. The thread-coupled bits (workers, locks, event handlers)
stay in `run_policy.py` because they need closures over shared state.

Public API:
    interpolate_to(arm, hand, target, n_steps, hz, dry_run, arm_dofs)
    wait_for_key(prompt) -> bool
    to_absolute(policy_out, state_now, arm_mode, hand_mode, arm_dofs) -> ndarray
    ensemble_action(buf, t_now, period, m) -> ndarray | None
    apply_ema(action, state, alpha_arm, alpha_hand, arm_dofs) -> (action, state)
    clip_to_state(action, state_now, max_arm, max_hand, arm_dofs) -> ndarray
    get_obs_retry(robot, retries, delay) -> dict
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import numpy as np


def interpolate_to(arm, hand, target: np.ndarray, n_steps: int, hz: float,
                   dry_run: bool, arm_dofs: int) -> None:
    """Linearly interpolate from current measured pose to `target` over n_steps.

    Sends each waypoint as joint commands. Used for homing and for the
    ramp-to-first-action step. With `--no-arm`, pass arm=None and arm_dofs=0;
    the target vector is then hand-only.
    """
    arm_obs  = get_obs_retry(arm) if arm is not None else None
    hand_obs = get_obs_retry(hand)
    current_hand = hand_obs["joint_positions"]
    if arm is not None:
        current = np.concatenate([arm_obs["joint_positions"], current_hand])
    else:
        current = current_hand

    period = 1.0 / hz
    for waypoint in np.linspace(current, target, n_steps):
        t0 = time.perf_counter()
        if not dry_run:
            if arm is not None:
                arm.send_action(waypoint[:arm_dofs].astype(np.float32))
                hand.send_action(waypoint[arm_dofs:].astype(np.float32))
            else:
                hand.send_action(waypoint.astype(np.float32))
        rem = period - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)


def wait_for_key(prompt: str = "Press SPACE/Enter to continue, q to quit: ") -> bool:
    """Block in raw-tty mode until the user presses SPACE/Enter (return True)
    or q / Ctrl-C / Ctrl-D (return False). Restores tty settings on exit."""
    print(prompt, flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch in ("q", "\x03", "\x04"):
                    return False
                if ch in (" ", "\r", "\n"):
                    return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clip_to_state(action: np.ndarray, state_now: np.ndarray,
                  max_delta_arm: float, max_delta_hand: float,
                  arm_dofs: int) -> np.ndarray:
    """Clip |action - state_now| per limb. Same semantic as gello's internal
    max_delta during teleop, applied client-side here so both teleop and policy
    inference share one code path for the safety clip.

    action, state_now: (D,) arrays where D = arm_dofs + hand_dofs.
    max_delta_arm, max_delta_hand: clip thresholds in rad. 0 = disabled (that
        slice passes through unclipped).
    arm_dofs: slice boundary. Set to 0 for hand-only setups.

    Returns a new (D,) float32 array. Does not modify inputs.
    """
    out = action.astype(np.float32).copy()
    if arm_dofs > 0 and max_delta_arm > 0:
        d = np.clip(out[:arm_dofs] - state_now[:arm_dofs],
                    -max_delta_arm, max_delta_arm)
        out[:arm_dofs] = state_now[:arm_dofs] + d
    if max_delta_hand > 0:
        d = np.clip(out[arm_dofs:] - state_now[arm_dofs:],
                    -max_delta_hand, max_delta_hand)
        out[arm_dofs:] = state_now[arm_dofs:] + d
    return out


def to_absolute(policy_out: np.ndarray, state_now: np.ndarray,
                arm_mode: str, hand_mode: str, arm_dofs: int) -> np.ndarray:
    """Convert raw policy output to absolute joint targets, per limb.
    arm dims are [0:arm_dofs], hand dims are [arm_dofs:]. With --no-arm,
    arm_dofs == 0 and only hand_mode matters."""
    out = policy_out.astype(np.float32).copy()
    if arm_dofs > 0 and arm_mode == "delta":
        out[:arm_dofs] = state_now[:arm_dofs] + policy_out[:arm_dofs]
    if hand_mode == "delta":
        out[arm_dofs:] = state_now[arm_dofs:] + policy_out[arm_dofs:]
    return out


def ensemble_action(buf, t_now: float, period: float, m: float) -> np.ndarray | None:
    """ACT-style temporal ensembling, newer-prioritized convention.

    buf: list of (chunk: ndarray[T,D], t_obs: float) entries.
    Returns (D,) weighted-avg prediction for wall-clock t_now, or None if no
    chunk's window covers t_now.

    For each chunk c, idx = round((t_now - t_obs_c) / period) is BOTH:
      - the position within c whose prediction is for t_now
      - the age of c in ticks since obs was sampled
    Weight = exp(-m * idx) — newest chunks (idx≈0) get weight 1; older decays.

    Note: this is opposite to ACT's original w_i = exp(-m * i) with i=0 oldest.
    The newer-prioritized form is more robust at VLA-scale latencies (~130ms)
    where stale-chunk averaging causes oscillations (RTC paper, sec 4.1).
    """
    preds, weights = [], []
    for chunk, t_obs in buf:
        idx = int(round((t_now - t_obs) / period))
        if 0 <= idx < len(chunk):
            preds.append(chunk[idx])
            weights.append(np.exp(-m * idx))
    if not preds:
        return None
    p = np.stack(preds, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32); w /= w.sum()
    return (w[:, None] * p).sum(axis=0).astype(np.float32)


def apply_ema(action: np.ndarray,
              ema_state: np.ndarray | None,
              alpha_arm: float,
              alpha_hand: float,
              arm_dofs: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-limb EMA low-pass on the action.

    out[i] = a * prev[i] + (1-a) * new[i]
    Higher alpha = smoother + laggier. 0 = pass-through on that slice.

    State is lazily initialized to the first action so the first tick passes
    through. Returns (filtered_action, updated_state). Caller persists state
    across ticks.
    """
    if alpha_arm <= 0.0 and alpha_hand <= 0.0:
        return action, ema_state    # both off, pass-through

    if ema_state is None:
        ema_state = action.copy()
    out = action.copy()
    if arm_dofs > 0 and alpha_arm > 0.0:
        a = alpha_arm
        ema_state[:arm_dofs] = a * ema_state[:arm_dofs] + (1.0 - a) * action[:arm_dofs]
        out[:arm_dofs] = ema_state[:arm_dofs]
    else:
        ema_state[:arm_dofs] = action[:arm_dofs]
    if alpha_hand > 0.0:
        a = alpha_hand
        ema_state[arm_dofs:] = a * ema_state[arm_dofs:] + (1.0 - a) * action[arm_dofs:]
        out[arm_dofs:] = ema_state[arm_dofs:]
    else:
        ema_state[arm_dofs:] = action[arm_dofs:]
    return out.astype(np.float32), ema_state


def get_obs_retry(robot, retries: int = 10, delay: float = 0.1):
    """Retry-wrapper around robot.get_observation(). Used during homing where
    a transient None on the first poll shouldn't abort startup. Raises if all
    retries exhausted."""
    for _ in range(retries):
        obs = robot.get_observation()
        if obs is not None:
            return obs
        time.sleep(delay)
    raise RuntimeError(
        f"Cannot read state from {robot.__class__.__name__} after {retries} attempts."
    )
