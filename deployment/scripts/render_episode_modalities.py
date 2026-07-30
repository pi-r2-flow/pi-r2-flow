"""Render one episode to an MP4: camera + hand modalities.

    python scripts/render_episode_modalities.py demos/marker/.../episode_0000.hdf5 \
        --out /tmp/marker_ep0.mp4

Panels:
  - overhead camera frame
  - hand joint torque (12 dims, raw xhand sensor)
  - fingertip force magnitude per finger (5 fingers)
  - proprio state: thumb joints (idx 6-8) and ring joints (idx 14-15)
  - action: hand only (idx 6-17, 12 DOF)

A vertical cursor sweeps the time-series panels in sync with the video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")              # headless render → file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
import numpy as np

FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]
ARM_DOFS = 6

# Hand joint layout (indices into state/action[6:])
# thumb: 0,1,2 → bend, rota1, rota2
# index: 3,4,5 → bend, j1, j2
# mid:   6,7
# ring:  8,9
# pinky: 10,11
THUMB_IDXS = [6, 7, 8]    # absolute indices into 18-dim state/action
RING_IDXS  = [14, 15]
THUMB_LABELS = ["th_bend", "th_rota1", "th_rota2"]
RING_LABELS  = ["ring_j1", "ring_j2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output MP4 (default: <episode>.modalities.mp4)")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback fps (default: real-time from timestamps)")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth frame to speed up (default 1)")
    args = ap.parse_args()

    out = args.out or args.episode.with_suffix(".modalities.mp4")

    with h5py.File(args.episode) as f:
        task       = str(f.attrs.get("task", ""))
        timestamps = f["timestamps"][()]
        torque     = f["observation/joint_torque"][()]      # (T, 18)
        ft         = f["observation/fingertip_force"][()]   # (T, 5, 3)
        tactile    = f["observation/tactile"][()]           # (T, 5, 120, 3)
        state      = f["observation/state"][()]             # (T, 18)
        action     = f["action"][()]                        # (T, 18)
        has_images = "observation/images/overhead" in f
        images     = f["observation/images/overhead"][()] if has_images else None

    T = len(timestamps)
    t_rel   = timestamps - timestamps[0]
    hand_tq = torque[:, ARM_DOFS:]          # (T, 12)
    ft_mag  = np.linalg.norm(ft, axis=-1)  # (T, 5)

    real_fps = 1.0 / float(np.diff(timestamps).mean())
    fps = args.fps or real_fps

    # Resample to uniform time so cursor sweeps at constant speed.
    dur = float(t_rel[-1])
    out_times = np.arange(0, dur, 1.0 / fps)[::args.stride]
    frame_indices = np.searchsorted(t_rel, out_times).clip(0, T - 1)

    print(f"Episode: {args.episode.name}  task='{task}'  T={T}  dur={dur:.1f}s  "
          f"real_fps={real_fps:.1f}  render_fps={fps:.1f}  out_frames={len(frame_indices)}  out={out}")

    # ---- layout: 4 rows × 5 cols ----
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{args.episode.name} — '{task}'", fontsize=12)
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.55, wspace=0.4)

    ax_img     = fig.add_subplot(gs[0:2, 0:2])
    ax_handtq  = fig.add_subplot(gs[0, 2:5])
    ax_force   = fig.add_subplot(gs[1, 2:5])
    ax_proprio = fig.add_subplot(gs[2, 0:5])
    ax_act     = fig.add_subplot(gs[3, 0:5])

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for j in range(hand_tq.shape[1]):
        ax_handtq.plot(t_rel, hand_tq[:, j], lw=0.7, alpha=0.8)
    ax_handtq.set_title("Hand joint torque (12 dims, raw)"); ax_handtq.set_ylabel("raw")

    for k, (lbl, c) in enumerate(zip(FINGER_LABELS, colors)):
        ax_force.plot(t_rel, ft_mag[:, k], label=lbl, lw=1.0, color=c)
    ax_force.set_yscale("symlog", linthresh=0.1)
    ax_force.set_title("Fingertip force magnitude |F| (N, symlog)"); ax_force.set_ylabel("|F| (N)")
    ax_force.legend(loc="upper right", fontsize=8, ncol=5)

    # proprio: thumb (blue shades) + ring (red shades)
    thumb_colors = ["#1f77b4", "#6baed6", "#9ecae1"]
    ring_colors  = ["#d62728", "#fc8d59"]
    for idx, lbl, c in zip(THUMB_IDXS, THUMB_LABELS, thumb_colors):
        ax_proprio.plot(t_rel, state[:, idx], lw=1.0, color=c, label=lbl)
    for idx, lbl, c in zip(RING_IDXS, RING_LABELS, ring_colors):
        ax_proprio.plot(t_rel, state[:, idx], lw=1.0, color=c, label=lbl, linestyle="--")
    ax_proprio.set_title("Proprio state — thumb & ring (rad)"); ax_proprio.set_ylabel("rad")
    ax_proprio.legend(loc="upper right", fontsize=7, ncol=5)

    # action: hand only (12 DOF)
    hand_joint_labels = ["th_bend","th_r1","th_r2","idx_b","idx_j1","idx_j2",
                         "mid_j1","mid_j2","ring_j1","ring_j2","pky_j1","pky_j2"]
    for j in range(12):
        ax_act.plot(t_rel, action[:, ARM_DOFS + j], lw=0.7, alpha=0.8, label=hand_joint_labels[j])
    ax_act.set_title("Action — hand (12 DOF)"); ax_act.set_ylabel("rad")
    ax_act.set_xlabel("time (s)")
    ax_act.legend(loc="upper right", fontsize=6, ncol=6)

    cursors = [ax.axvline(0, color="k", lw=1.2)
               for ax in (ax_handtq, ax_force, ax_proprio, ax_act)]

    if has_images:
        im_handle = ax_img.imshow(images[0])
        ax_img.set_title("overhead camera")
    else:
        blank = np.full((480, 640, 3), 40, dtype=np.uint8)
        im_handle = ax_img.imshow(blank)
        ax_img.text(0.5, 0.5, "no camera", ha="center", va="center",
                    transform=ax_img.transAxes, color="gray", fontsize=14)
        ax_img.set_title("overhead camera")
    ax_img.axis("off")

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
