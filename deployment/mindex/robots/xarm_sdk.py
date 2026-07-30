"""Bare xArm SDK wrapper for policy inference.

No gello, no background servo thread, no internal max_delta clamping.
Just the minimal lerobot-shaped Robot interface
(`connect`, `disconnect`, `get_observation`, `send_action`, `is_connected`)
talking to xArm-Python-SDK directly.

For teleop use mindex.robots.xarm_robot.XArmRobot (gello-based).
"""

from __future__ import annotations

import time

import numpy as np

NUM_JOINTS = 6


class XArmSDK:
    """Position-controlled xArm6 via streaming servo mode (mode=1).

    Action and state are 6-dim joint angles in radians.
    """

    def __init__(self, ip: str = "192.168.1.244", collision_sensitivity: int = 4,
                 joint_jerk: float | None = None, joint_maxacc: float | None = None):
        """
        collision_sensitivity: 0 = off, 1 = low, ..., 5 = highest.
            Default 4 = strong protection. Lower if false-positive trips.
        joint_jerk: cap on joint jerk in rad/s^3. None = leave xArm default.
            Smaller = smoother but laggier. Try 20-40 to reduce streaming jitter.
        joint_maxacc: cap on joint accel in rad/s^2. None = leave xArm default.
            Smaller = softer reactions at chunk boundaries. Try 10-20.
        """
        self._ip = ip
        self._collision_sensitivity = int(collision_sensitivity)
        self._joint_jerk = joint_jerk
        self._joint_maxacc = joint_maxacc
        self._arm = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def num_dofs(self) -> int:
        return NUM_JOINTS

    def connect(self) -> None:
        if self._connected:
            return
        from xarm.wrapper import XArmAPI
        self._arm = XArmAPI(self._ip, is_radian=True, report_type="real")

        # Force full recovery from any latched fault/estop state.
        # The plain (clean_error → motion_enable → set_state) triplet handles most
        # soft errors, but some sticky states (post-collision, servo-off after
        # e-stop, mode-lock) need a more aggressive sequence.
        for attempt in range(3):
            self._arm.clean_error()
            self._arm.clean_warn()
            self._arm.motion_enable(True)     # enable all joints
            time.sleep(0.3)
            self._arm.set_mode(0)             # position mode first (some transitions
            time.sleep(0.1)                    # require going through mode 0 briefly)
            self._arm.set_state(0)            # sport state
            time.sleep(0.3)
            err = getattr(self._arm, "error_code", 0) or 0
            warn = getattr(self._arm, "warn_code", 0) or 0
            state = getattr(self._arm, "state", -1)
            if err == 0 and state == 2:       # 2 = ready-to-move sport state
                break
            print(f"xArm recovery attempt {attempt+1}/3: "
                  f"err={err} warn={warn} state={state}, retrying...")
            time.sleep(0.5)
        else:
            print(f"WARN: xArm still in error state after 3 attempts "
                  f"(err={self._arm.error_code}, state={self._arm.state}). "
                  f"You may need to release the physical E-stop.")

        self._arm.set_mode(1)              # servo mode: streaming joint targets
        time.sleep(0.5)
        # 0 = off, 1-5 = increasing sensitivity. Higher trips faster on contact.
        self._arm.set_collision_sensitivity(self._collision_sensitivity)
        time.sleep(0.2)
        # Motion-shaping caps (smoothing knobs). Both no-op if None.
        if self._joint_jerk is not None:
            self._arm.set_joint_jerk(float(self._joint_jerk), is_radian=True)
            time.sleep(0.1)
        if self._joint_maxacc is not None:
            self._arm.set_joint_maxacc(float(self._joint_maxacc), is_radian=True)
            time.sleep(0.1)
        self._arm.set_state(0)             # ready to move
        time.sleep(0.5)
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            self._arm.set_state(3)  # stop
        except Exception:
            pass
        try:
            self._arm.disconnect()
        except Exception:
            pass

    def get_observation(self, fresh: bool = False) -> dict | None:
        """`fresh` is accepted for API parity with XArmRobot/XHandRobot —
        this wrapper always polls fresh via get_servo_angle.

        Returns: joint_positions (rad) + joint_torque (Nm, controller-estimated).
        joint_torque missing if firmware doesn't support get_joints_torque."""
        if not self._connected:
            return None
        try:
            code, angles = self._arm.get_servo_angle(is_radian=True)
            if code != 0:
                return None
            out = {
                "joint_positions": np.asarray(angles[:NUM_JOINTS], dtype=np.float32),
            }
            try:
                tcode, torques = self._arm.get_joints_torque()
                if tcode == 0 and torques is not None:
                    out["joint_torque"] = np.asarray(torques[:NUM_JOINTS], dtype=np.float32)
            except Exception:
                pass  # firmware may not expose get_joints_torque; non-fatal
            return out
        except Exception as e:
            print(f"warning: xarm get_observation failed: {e}")
            return None

    def send_action(self, action: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("XArmSDK not connected.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (NUM_JOINTS,):
            raise ValueError(f"action must be shape ({NUM_JOINTS},), got {action.shape}")
        # set_servo_angle_j is the low-latency streaming command for mode=1.
        # wait=False returns immediately; the arm interpolates internally.
        self._arm.set_servo_angle_j(action.tolist(), wait=False, is_radian=True)
