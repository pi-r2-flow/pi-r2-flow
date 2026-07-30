"""LeRobot-compatible HDF5 episode writer.

Usage:
    writer = EpisodeWriter("/path/to/demos", task="pick up cup")
    # in control loop:
    writer.add_step(obs, action, time.time())
    # on save keystroke:
    writer.save(episode_idx)   # → episode_0000.hdf5
    # on discard keystroke:
    writer.discard()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("h5py required: pip install h5py")


class EpisodeWriter:
    """Buffers steps in memory; flushes to HDF5 on save().

    HDF5 schema:
        observation/images/overhead  (T, H, W, 3)     uint8   — if rgb in obs
        observation/state            (T, D)            float32 — joint_positions
        observation/tactile          (T, 5, 120, 3)   float32 — if xhand_tactile in obs
        action                       (T, D)            float32 — absolute commanded targets
        action_delta                 (T, D)            float32 — action − state (computed on save)
        timestamps                   (T,)              float64
        attrs: task, n_steps
    """

    def __init__(self, output_dir: str | Path, task: str):
        slug = task.lower().replace(" ", "_")[:40] or "untitled"
        self._output_dir = Path(output_dir) / slug
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task = task
        self._buffer: list[dict] = []
        self._extra_attrs: dict = {}
        # start numbering after any existing episodes
        existing = sorted(self._output_dir.glob("episode_*.hdf5"))
        self._next_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    def set_extra_attrs(self, attrs: dict) -> None:
        """Stash extra HDF5 attrs (e.g. tag poses) to be written on save().
        Reset on every discard() / save()."""
        self._extra_attrs = {k: v for k, v in attrs.items() if v is not None}

    def add_step(self, obs: dict, action: np.ndarray, timestamp: float,
                 img_timestamp: float | None = None) -> None:
        self._buffer.append({"obs": obs, "action": np.asarray(action, dtype=np.float32),
                              "timestamp": float(timestamp),
                              "img_timestamp": float(img_timestamp) if img_timestamp is not None else float(timestamp)})

    def save(self) -> Path:
        if not self._buffer:
            raise RuntimeError("No steps buffered — nothing to save.")
        path = self._output_dir / f"episode_{self._next_idx:04d}.hdf5"
        self._next_idx += 1
        with h5py.File(path, "w") as f:
            f.attrs["task"] = self._task
            f.attrs["n_steps"] = len(self._buffer)
            for k, v in self._extra_attrs.items():
                f.attrs[k] = v

            f.create_dataset(
                "timestamps",
                data=np.array([s["timestamp"] for s in self._buffer], dtype=np.float64),
            )
            actions = np.stack([s["action"] for s in self._buffer], axis=0).astype(np.float32)
            f.create_dataset("action", data=actions)

            obs_grp = f.create_group("observation")

            states = None
            if "joint_positions" in self._buffer[0]["obs"]:
                states = np.stack(
                    [s["obs"]["joint_positions"] for s in self._buffer]
                ).astype(np.float32)
                obs_grp.create_dataset("state", data=states)

            # Also save delta = action − state for delta-action training.
            # Same info as (action, state) but pre-computed for convenience.
            if states is not None and actions.shape == states.shape:
                f.create_dataset("action_delta", data=(actions - states).astype(np.float32))

            if "joint_torque" in self._buffer[0]["obs"]:
                obs_grp.create_dataset(
                    "joint_torque",
                    data=np.stack([s["obs"]["joint_torque"] for s in self._buffer]).astype(np.float32),
                )

            if "fingertip_force" in self._buffer[0]["obs"]:
                obs_grp.create_dataset(
                    "fingertip_force",
                    data=np.stack([s["obs"]["fingertip_force"] for s in self._buffer]).astype(np.float32),
                )

            if "xhand_tactile" in self._buffer[0]["obs"]:
                obs_grp.create_dataset(
                    "tactile",
                    data=np.stack([s["obs"]["xhand_tactile"] for s in self._buffer]).astype(np.float32),
                )

            if "rgb" in self._buffer[0]["obs"]:
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset(
                    "overhead",
                    data=np.stack([s["obs"]["rgb"] for s in self._buffer]).astype(np.uint8),
                )
                img_grp.create_dataset(
                    "timestamps",
                    data=np.array([s["img_timestamp"] for s in self._buffer], dtype=np.float64),
                )
                if "rgb_cam2" in self._buffer[0]["obs"]:
                    img_grp.create_dataset(
                        "cam2",
                        data=np.stack([s["obs"]["rgb_cam2"] for s in self._buffer]).astype(np.uint8),
                    )

        n = len(self._buffer)
        self._buffer.clear()
        self._extra_attrs = {}
        print(f"Saved {n} steps → {path}")
        return path

    def discard(self) -> None:
        n = len(self._buffer)
        self._buffer.clear()
        self._extra_attrs = {}
        print(f"Discarded {n} steps.")

    def undo_last(self) -> Path | None:
        """Delete the last saved episode and decrement counter."""
        prev = self._output_dir / f"episode_{self._next_idx - 1:04d}.hdf5"
        if prev.exists():
            prev.unlink()
            self._next_idx -= 1
            return prev
        return None

    @property
    def episode_idx(self) -> int:
        return self._next_idx

    @property
    def n_steps(self) -> int:
        return len(self._buffer)
