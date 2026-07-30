"""XHand 12-DOF robotic hand driver.

Wraps the vendor `xhand_controller` SDK with a lerobot-shaped `Robot` interface
(`connect`, `disconnect`, `get_observation`, `send_action`, `is_connected`).
Joint positions in radians.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np

NUM_JOINTS = 12

# Per vendor SDK: 0=passive (limp), 3=position control, 5=force control.
MODE_PASSIVE = 0
MODE_POSITION = 3
MODE_FORCE = 5

# Finger ids that carry fingertip force/temperature sensors (per
# xhand_control_example.py:147). 5 sensors, indexed 0..4 in state.sensor_data.
SENSORED_FINGER_IDS = (2, 5, 7, 9, 11)


class XHandRobot:
    """Position-controlled XHand. 12 joints, units in radians.

    Connection: RS485 over USB-serial by default (this rig). EtherCAT is also
    supported by the SDK; if you switch to it, grant the python interpreter raw
    socket capability:
        sudo setcap cap_net_raw+ep $(readlink -f $(which python3))
    """

    def __init__(
        self,
        protocol: str = "RS485",
        serial_port: str = "/dev/serial/by-id/usb-FTDI_USB-RS485-WE_FTAAUYBU-if00-port0",
        baud_rate: int = 3_000_000,
        kp: int = 100,
        ki: int = 0,
        kd: int = 0,
        tor_max: int = 300,
        mode: int = MODE_POSITION,
    ):
        # SDK pybind binding requires int for kp/ki/kd/tor_max/mode (FingerCommand_t).
        self._protocol = protocol
        self._serial_port = serial_port
        self._baud_rate = baud_rate
        self._kp = int(kp)
        self._ki = int(ki)
        self._kd = int(kd)
        self._tor_max = int(tor_max)
        self._mode = int(mode)

        self._device = None  # vendor XHandControl instance, set in connect()
        self._command = None  # pre-allocated HandCommand_t, set in connect()
        self._hand_id: Optional[int] = None
        self._connected = False
        self._lock = threading.Lock()  # serialises RS485 bus between threads
        self._cached_obs: dict | None = None  # state cached from last send_command response
        # Software bias subtracted in _parse_state. Set by reset_sensors() after
        # the vendor SDK reset_sensor call — vendor reset alone leaves residual
        # offsets (~5-30 N), so we snapshot a fresh frame and subtract.
        self._bias_ft: np.ndarray  = np.zeros((len(SENSORED_FINGER_IDS), 3), dtype=np.float32)
        self._bias_raw: np.ndarray = np.zeros((len(SENSORED_FINGER_IDS), 120, 3), dtype=np.float32)

    # ---- lerobot-shaped surface ----------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        # Import here so the module can be imported in envs without the wheel
        # (e.g. for type-checking or docs).
        from xhand_controller import xhand_control

        self._device = xhand_control.XHandControl()

        if self._protocol == "EtherCAT":
            ports = self._device.enumerate_devices("EtherCAT")
            if not ports:
                raise RuntimeError(
                    "No EtherCAT devices found. Check cable and that the Python "
                    "interpreter has cap_net_raw+ep (see XHand SDK README)."
                )
            rsp = self._device.open_ethercat(ports[0])
            if rsp.error_code != 0:
                raise RuntimeError(f"Failed to open XHand via EtherCAT: {rsp.error_message}")
            ids = self._device.list_hands_id()
            if not ids:
                raise RuntimeError("Device opened but no hand IDs reported.")
            self._hand_id = ids[0]

        elif self._protocol == "RS485":
            # Vendor example calls enumerate first; some SDK builds need it as init.
            enum_ports = self._device.enumerate_devices("RS485") or []
            # Try the requested port first, then fall back to other /dev/ttyUSB*.
            candidates = [self._serial_port] + [
                p for p in enum_ports
                if p.startswith("/dev/ttyUSB") and p != self._serial_port
            ]
            last_err = None
            self._hand_id = None
            for port in candidates:
                rsp = self._device.open_serial(port, self._baud_rate)
                if rsp.error_code != 0:
                    last_err = f"open_serial({port}) failed: {rsp.error_message}"
                    continue
                ids = self._device.list_hands_id()
                if ids:
                    self._serial_port = port
                    self._hand_id = ids[0]
                    break
                last_err = f"opened {port} but list_hands_id() empty"
                # SDK has no explicit close; we accept potential port leak here
                # since we'll re-open the right one on the next iteration.
            if self._hand_id is None:
                raise RuntimeError(
                    f"Failed to open XHand via RS485 on any of {candidates}. "
                    f"Last: {last_err}"
                )

        else:
            raise ValueError(f"Unknown protocol: {self._protocol!r} (expected EtherCAT or RS485)")

        self._command = xhand_control.HandCommand_t()
        for i in range(NUM_JOINTS):
            fc = self._command.finger_command[i]
            fc.id = i
            fc.kp = self._kp
            fc.ki = self._ki
            fc.kd = self._kd
            fc.tor_max = self._tor_max
            fc.mode = self._mode
            fc.position = 0.0

        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        # Best-effort: drop the hand into passive mode so it doesn't hold a pose
        # after the controller exits.
        try:
            self._set_mode(MODE_PASSIVE)
        except Exception:
            pass
        self._connected = False
        # Vendor SDK doesn't expose an explicit close in the example; relying
        # on object teardown.

    def get_observation(self, fresh: bool = False) -> dict | None:
        """Return hand state.

        Args:
            fresh: If True, force a fresh RS485 read_state (~10-15ms round-trip).
                   If False (default), return the cached state populated by the
                   most recent send_action — instant, but may be tick-misaligned.

        Use fresh=True for delta-action recording and inference, where the
        anchor state must be a true tick-synchronous measurement (the cache's
        refresh moment depends on when send_action was last called, which
        varies between recording and replay → bias → drift).

        On success, dict keys:
          - joint_positions: (12,) radians
          - joint_torques:   (12,) vendor units
          - joint_temperatures: (12,) °C   (low byte of vendor field; high bits encode status)
          - fingertip_force: (5, 3) N, ordered by SENSORED_FINGER_IDS
          - fingertip_temperatures: (5,) °C
        """
        self._require_connected()
        if not fresh and self._cached_obs is not None:
            return self._cached_obs
        with self._lock:
            # force_update=True does a true RS485 read, not the cache.
            err, state = self._device.read_state(self._hand_id, True)
        if err.error_code != 0:
            print(f"warning: read_state failed: {err.error_message}", file=sys.stderr)
            return None
        self._cached_obs = self._parse_state(state)
        return self._cached_obs

    def send_action(self, action: np.ndarray) -> None:
        """Command 12-vector of target joint positions and cache the returned state."""
        self._require_connected()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (NUM_JOINTS,):
            raise ValueError(
                f"action must have shape ({NUM_JOINTS},), got {action.shape}"
            )
        for i in range(NUM_JOINTS):
            self._command.finger_command[i].position = float(action[i])
        with self._lock:
            err = self._device.send_command(self._hand_id, self._command)
            if err.error_code != 0:
                print(f"warning: send_command: {err.error_message}", file=sys.stderr)
                return
            # read_state(False) returns the state cached by send_command — no extra RS485 trip.
            err2, state = self._device.read_state(self._hand_id, False)
        if err2.error_code == 0:
            self._cached_obs = self._parse_state(state)

    def _parse_state(self, state) -> dict:
        pos = np.empty(NUM_JOINTS, dtype=np.float32)
        tor = np.empty(NUM_JOINTS, dtype=np.float32)
        temp = np.empty(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            fs = state.finger_state[i]
            pos[i] = fs.position
            tor[i] = fs.torque
            temp[i] = float(fs.temperature & 0xFF)  # low byte = °C
        ft_force = np.empty((len(SENSORED_FINGER_IDS), 3), dtype=np.float32)
        ft_temp = np.empty(len(SENSORED_FINGER_IDS), dtype=np.float32)
        ft_raw_force = []
        for k in range(len(SENSORED_FINGER_IDS)):
            sd = state.sensor_data[k]
            ft_force[k, 0] = sd.calc_force.fx
            ft_force[k, 1] = sd.calc_force.fy
            ft_force[k, 2] = sd.calc_force.fz
            ft_temp[k] = sd.calc_temperature
            ft_raw_force.append(
                np.array([[f.fx, f.fy, f.fz] for f in sd.raw_force], dtype=np.float32)
            )
        return {
            "joint_positions": pos,
            "joint_torques": tor,
            "joint_temperatures": temp,
            "fingertip_force": ft_force - self._bias_ft,
            "xhand_tactile": np.stack(ft_raw_force, axis=0) - self._bias_raw,
            "fingertip_temperatures": ft_temp,
        }

    # ---- helpers --------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        return NUM_JOINTS

    @property
    def hand_id(self) -> int:
        self._require_connected()
        return self._hand_id  # type: ignore[return-value]

    def reset_sensors(self, sensor_ids: tuple = SENSORED_FINGER_IDS,
                      max_retries: int = 3, retry_wait_sec: float = 0.2,
                      verify: bool = True, verify_thresh_n: float = 2.0) -> None:
        """Zero fingertip sensor offsets via SDK reset_sensor call.

        Matches vendor xhand_control_example.py: one reset_sensor call per sensor,
        retry only on error_code != 0.

        Args:
            sensor_ids: which fingertip sensors to reset (default: all 5).
            max_retries: max retry attempts per sensor on failure.
            retry_wait_sec: wait between retries.
            verify: if True, after resetting do a fresh read and warn on any
                    fingertip whose |F| exceeds verify_thresh_n. Catches the
                    common mistake of resetting while the hand is in contact.
            verify_thresh_n: max acceptable |F| post-reset in N (default 2.0).
        """
        MAX_OUTER_ITERS = 5
        ids_to_reset = list(sensor_ids)
        for outer in range(MAX_OUTER_ITERS):
            self._require_connected()
            for sid in ids_to_reset:
                for attempt in range(max_retries):
                    err = self._device.reset_sensor(self._hand_id, sid)
                    if err.error_code == 0:
                        break
                    print(f"warning: reset_sensor(finger {sid}) attempt {attempt+1} "
                        f"failed: {err.error_message}", file=sys.stderr)
                    time.sleep(retry_wait_sec)

            # Vendor reset_sensor() alone leaves ~5-30 N residual offsets.
            # Snapshot a software bias from a few fresh reads and subtract going
            # forward — matches vendor's monitor_tactile.py example.
            self._bias_ft[:]  = 0.0
            self._bias_raw[:] = 0.0
            ft_acc  = np.zeros_like(self._bias_ft)
            raw_acc = np.zeros_like(self._bias_raw)
            n = 0
            for _ in range(5):
                time.sleep(0.02)
                o = self.get_observation(fresh=True)
                if o is None: continue
                ft_acc  += o["fingertip_force"]
                raw_acc += o["xhand_tactile"]
                n += 1
            if n > 0:
                self._bias_ft  = (ft_acc  / n).astype(np.float32)
                self._bias_raw = (raw_acc / n).astype(np.float32)

            if not verify:
                return

            obs = self.get_observation(fresh=True)
            if obs is None:
                print("warning: post-reset verify read failed", file=sys.stderr)
                return
            mags = np.linalg.norm(obs["fingertip_force"], axis=1)  # (5,)
            bad_idx = np.where(mags > verify_thresh_n)[0]
            if bad_idx.size == 0:
                return    # clean

            # Retry only the still-bad sensors next iteration.
            ids_to_reset = [sensor_ids[i] for i in bad_idx]
            if outer < MAX_OUTER_ITERS - 1:
                labels = ["thumb", "index", "middle", "ring", "pinky"]
                msgs = [f"{labels[i]}={mags[i]:.1f}N" for i in bad_idx]
                print(f"  retry reset for {', '.join(msgs)}  "
                      f"(iter {outer + 1}/{MAX_OUTER_ITERS})", file=sys.stderr)

        # Exhausted outer retries — print final warning.
        labels = ["thumb", "index", "middle", "ring", "pinky"]
        msgs = [f"{labels[i]}={mags[i]:.1f}N" for i in bad_idx]
        print(f"warning: post-bias fingertip forces still non-zero "
              f"({', '.join(msgs)}) after {MAX_OUTER_ITERS} retries — "
              f"sensor noisy or hand contacting something", file=sys.stderr)

    def _set_mode(self, mode: int) -> None:
        self._require_connected()
        for i in range(NUM_JOINTS):
            self._command.finger_command[i].mode = mode
        err = self._device.send_command(self._hand_id, self._command)
        if err.error_code != 0:
            raise RuntimeError(f"set_mode({mode}) failed: {err.error_message}")
        self._mode = mode

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("XHandRobot not connected. Call connect() first.")
