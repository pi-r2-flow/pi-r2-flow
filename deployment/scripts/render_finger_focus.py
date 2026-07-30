"""Render a focused eval-log video: camera + per-finger force & key action DOF.

    python scripts/render_finger_focus.py episode_0000.hdf5 --fingers thumb index

Layout:
  left (full height):   overhead camera
  top-right:            fingertip force |F| for selected fingers only
  bottom-right:         predicted action — one key DOF per selected finger

Output: <episode>.<finger1>_<finger2>.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
import numpy as np

# (force_idx, dof_slice_in_hand, key_dof_offset_in_slice, key_label)
FINGER_META = {
    "thumb":  (0, slice(0, 3),   0, "thumb_bend"),
    "index":  (1, slice(3, 6),   1, "index_j1"),
    "middle": (2, slice(6, 8),   0, "mid_j1"),
    "ring":   (3, slice(8, 10),  0, "ring_j1"),
    "pinky":  (4, slice(10, 12), 0, "pinky_j1"),
}

COLORS = {
    "thumb": "tab:blue", "index": "tab:orange",
    "middle": "tab:green", "ring": "tab:red", "pinky": "tab:purple",
}
ARM_DOFS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--fingers", nargs="+", default=["thumb", "index"],
                    choices=list(FINGER_META), metavar="FINGER")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    fingers = args.fingers
    suffix = "_".join(fingers)
    out = args.out or args.episode.with_suffix(f".{suffix}.mp4")

    with h5py.File(args.episode) as f:
        task       = str(f.attrs.get("task", ""))
        timestamps = f["timestamps"][()]
        actions    = f["action"][()]                         # (T, 18)
        ft         = f["observation/fingertip_force"][()]    # (T, 5, 3)
        has_images = "observation/images/overhead" in f
        images     = f["observation/images/overhead"][()] if has_images else None

    T = len(timestamps)
    t_rel   = timestamps - timestamps[0]
    real_fps = 1.0 / float(np.diff(timestamps).mean())
    fps = args.fps or real_fps

    ft_mag   = np.linalg.norm(ft, axis=-1)   # (T, 5)
    hand_act = actions[:, ARM_DOFS:]          # (T, 12)

    # Resample to uniform time so cursor sweeps at constant speed.
    # For each output frame at t = n/fps, snap to nearest input frame index.
    dur = float(t_rel[-1])
    out_times = np.arange(0, dur, 1.0 / fps)
    frame_indices = np.searchsorted(t_rel, out_times).clip(0, T - 1)

    print(f"Episode: {args.episode.name}  fingers={fingers}  T={T}  "
          f"dur={dur:.1f}s  fps={real_fps:.1f}  out_frames={len(frame_indices)}  out={out}")

    # ---- layout: 2 rows × 2 cols ----
    fig = plt.figure(figsize=(19.6, 7.3))
    fig.suptitle(f"{args.episode.name} — '{task}'", fontsize=11)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3,
                           width_ratios=[1, 1.2])

    ax_img   = fig.add_subplot(gs[0:2, 0])
    ax_force = fig.add_subplot(gs[0, 1])
    ax_act   = fig.add_subplot(gs[1, 1])

    # camera
    if has_images:
        im_handle = ax_img.imshow(images[0])
    else:
        blank = np.full((480, 640, 3), 40, dtype=np.uint8)
        im_handle = ax_img.imshow(blank)
        ax_img.text(0.5, 0.5, "no camera", ha="center", va="center",
                    transform=ax_img.transAxes, color="gray", fontsize=14)
    ax_img.set_title("overhead camera")
    ax_img.axis("off")

    # force: selected fingers only
    for finger in fingers:
        fidx, _, _, _ = FINGER_META[finger]
        ax_force.plot(t_rel, ft_mag[:, fidx], color=COLORS[finger], lw=1.0, label=finger)
    ax_force.set_title("Fingertip force magnitude |F| (N)")
    ax_force.set_ylabel("|F| (N)")
    ax_force.legend(loc="upper right", fontsize=8)

    # action: one key DOF per finger
    for finger in fingers:
        _, dof_slice, key_off, key_label = FINGER_META[finger]
        dof_vals = hand_act[:, dof_slice][:, key_off]
        ax_act.plot(t_rel, dof_vals, color=COLORS[finger], lw=1.0, label=key_label)
    ax_act.set_title("Predicted action (rad)")
    ax_act.set_ylabel("rad")
    ax_act.set_xlabel("time (s)")
    ax_act.legend(loc="upper right", fontsize=8)

    cursors = [ax.axvline(0, color="k", lw=1.2) for ax in (ax_force, ax_act)]

    writer = FFMpegWriter(fps=fps, bitrate=4000)
    N = len(frame_indices)
    with writer.saving(fig, str(out), dpi=90):
        for n, idx in enumerate(frame_indices):
            if has_images:
                im_handle.set_data(images[idx])
            t_cur = out_times[n]
            for cur in cursors:
                cur.set_xdata([t_cur, t_cur])
            writer.grab_frame()
            if n % 50 == 0:
                print(f"  {n}/{N} frames", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
