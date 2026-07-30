"""Argparse builder for run_policy.py.

Extracted to keep the main app file focused on control logic. The
build_argparser() function returns an ArgumentParser with every flag the
policy loop understands. The main app calls .parse_args() itself so it
can also do post-parse validation (e.g. rejecting --ensemble + sync).
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # Hand
    ap.add_argument("--protocol", default="RS485", choices=["RS485", "EtherCAT"])
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--baud-rate", type=int, default=3_000_000)
    ap.add_argument("--rate-hz", type=float, default=25.0,
                    help="control loop target. Default 25 matches teleop recording "
                         "(actual ~27 Hz). Don't run faster than training rate or "
                         "the policy sees a distribution-shifted command frequency.")
    # Arm
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--arm-ip", default="192.168.1.244")
    ap.add_argument("--collision-sensitivity", type=int, default=2,
                    help="0=off, 1-5 increasing. xArm trips into fault on contact.")
    ap.add_argument("--joint-jerk", type=float, default=100,
                    help="cap joint jerk (rad/s^3) at xArm controller. "
                         "Lower=smoother but laggier. None=xArm default. Try 20-40.")
    ap.add_argument("--joint-maxacc", type=float, default=20,
                    help="cap joint max acceleration (rad/s^2) at xArm controller. "
                         "Lower=softer reactions. None=xArm default. Try 10-20.")
    # Camera
    ap.add_argument("--cam-host", default="localhost")
    ap.add_argument("--cam-port", type=int, default=5000)
    # GR00T server
    ap.add_argument("--host", default="localhost", help="GR00T server host")
    ap.add_argument("--port", type=int, default=5555, help="GR00T server port")
    ap.add_argument("--task", default="")
    ap.add_argument("--chunk-len", type=int, default=None,
                    help="non-ensemble: actions to execute per chunk before "
                         "re-query (also truncates stored chunk to this length). "
                         "ensemble: only governs pipelined kickoff cadence; "
                         "stored chunks keep full server horizon (50 actions) "
                         "for the rolling ensemble buffer. None = full chunk.")
    ap.add_argument("--query-mode", default="sync",
                    choices=["sync", "continuous", "pipelined"],
                    help="how GR00T queries are scheduled. "
                         "sync = block on each query. "
                         "continuous = worker thread queries non-stop. "
                         "pipelined = kick off next query while executing current chunk.")
    ap.add_argument("--async-vlm", action="store_true",
                    help="Decouple VLM and DiT on the server side: spawn a "
                         "background thread that pushes images to the server's "
                         "VLM cache continuously, and have each policy query "
                         "run DiT-only against the cached VLM. Orthogonal to "
                         "--query-mode (combinable with sync, continuous, or "
                         "pipelined). Requires DECOUPLED=1 server.")
    ap.add_argument("--vlm-host", default=None,
                    help="(--async-vlm + 2-GPU only) separate VLM-server host. "
                         "Default = same as --host (single-server mode). "
                         "Set together with --vlm-port to route update_vlm_cache "
                         "to a dedicated GPU/process. Requires the server-side "
                         "return_features / vl_embeds support in decoupled_policy.py. "
                         "Launch the two servers via "
                         "scripts/gr00t_inference_2gpu.sh on grogu.")
    ap.add_argument("--vlm-port", type=int, default=None,
                    help="(--async-vlm + 2-GPU only) separate VLM-server port. "
                         "Default = same as --port. See --vlm-host.")
    ap.add_argument("--arm-mode", default="absolute",
                    choices=["absolute", "delta"],
                    help="how to interpret arm dims [0:arm_dofs] of policy output. "
                         "absolute = target joint angles. "
                         "delta = δ from measured arm state; target = arm_state + δ. "
                         "Must match the variant the checkpoint was trained on.")
    ap.add_argument("--hand-mode", default="absolute",
                    choices=["absolute", "delta"],
                    help="how to interpret hand dims [arm_dofs:] of policy output. "
                         "absolute = target joint angles. "
                         "delta = δ from measured hand state; target = hand_state + δ. "
                         "Must match the variant the checkpoint was trained on.")

    # Safety / debug
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full loop but do not send_action to robots")
    ap.add_argument("--debug-swap", action="store_true",
                    help="print per-swap diagnostics: stale, chunk_idx, swap_target. "
                         "Also prints pipelined kickoff events. Use to verify "
                         "stale ≈ need_steps and swap_target ≤ T after kickoff fix.")
    ap.add_argument("--no-home", action="store_true",
                    help="skip the initial homing step")

    # Temporal ensembling (ACT-style; client-side smoothing for async modes)
    ap.add_argument("--ensemble", action="store_true",
                    help="ACT-style temporal ensembling. Requires --query-mode "
                         "{continuous, pipelined}. Maintains a buffer of recent "
                         "finished chunks; per tick, weighted-average their "
                         "predictions for the current wall-clock tick. Replaces "
                         "single-chunk swap-and-execute.")
    ap.add_argument("--ensemble-m", type=float, default=0.01,
                    help="exp-decay rate per tick of chunk age. 0 = uniform avg; "
                         "higher = newer chunks dominate. ACT paper: 0.01 for "
                         "fast tasks. Try 0.05-0.1 for sharper weighting.")
    ap.add_argument("--ensemble-buffer-size", type=int, default=30,
                    help="max chunks kept for ensembling. Default 30 = action horizon.")

    # PI-R2 / streaming-flow inference knobs
    ap.add_argument("--nfe", type=int, default=None,
                    help="override num_inference_timesteps (bucket count N, "
                         "controls dt = noise_s/N). None = checkpoint default. "
                         "Common: 1, 2, 4, 7, 15, 50, 100.")
    # Latency-aware clean-action inpaint (pir2 streaming + train-time RTC).
    ap.add_argument("--inpaint", action="store_true",
                    help="Send d 'during-latency' clean actions (= chunk_prev[idx:idx+d]) "
                         "as inpaint to the server. d = auto-derived latency_ticks from "
                         "rolling query time. Client skips past inpaint window when "
                         "emitting (chunk_idx = d after swap). Required for both Path A "
                         "and Path B inpaint deployment.")
    ap.add_argument("--force-nonstreaming", action="store_true",
                    help="Path A: force server's non-streaming flow loop (nfe full Euler "
                         "integration) instead of streaming inference. Required for train-"
                         "time-RTC ckpts (Config 2). Inpaint key on wire becomes "
                         "options['inpaint'] (vs options['action'] for streaming Path B). "
                         "Implies and requires --inpaint.")
    ap.add_argument("--ckpt-type", default=None,
                    choices=["plain_flow", "pir2", "rtc"],
                    help="declare ckpt-type for sanity asserts. "
                         "plain_flow: rejects --inpaint (no train-time clean-clamp "
                         "exposure → OOD). rtc: requires --force-nonstreaming (flow-"
                         "loop route). pir2: all paths allowed. Read the ckpt's "
                         "config.json: streaming=False → plain_flow; "
                         "streaming_rtc_weight>0 → rtc; else pir2. "
                         "Omit to skip the check.")

    # Eval logging
    ap.add_argument("--eval-log", type=str, default=None,
                    help="dir to log per-tick obs/state/action as an HDF5 "
                         "episode (same schema as teleop recording). Also "
                         "stores wall-clock timestamps of every policy query "
                         "(get_action_chunk / cached / seed / update_vlm) as "
                         "the `query_events` HDF5 attr. Existing analysis "
                         "scripts (visualize_episode.py, "
                         "render_episode_modalities.py) work on the resulting "
                         "file without modification.")

    return ap
