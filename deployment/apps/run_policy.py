"""GR00T policy deployment loop @ ~30 Hz (matches teleop recording rate).

    python apps/run_policy.py --task "pick up cup" --host 192.168.1.100

Requires:
  - GR00T inference server: python gr00t/eval/run_gr00t_server.py --model-path <ckpt>
  - Camera server:          python apps/run_camera_server.py --port 5000

Action space: (18,) = xArm6[0:6] + XHand[6:18], or (12,) with --no-arm.

Arm and hand action modes are chosen independently with --arm-mode and
--hand-mode (each {absolute, delta}). Pick them to match the variant the
checkpoint was trained on (e.g. armA_handD ⇒ --arm-mode absolute --hand-mode delta).

Safety:
  - Smoothly ramps from current pose to first predicted action over ~3s
  - --dry-run: log actions but do not send to robot
"""

from __future__ import annotations

import argparse
import collections
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindex.robots.xhand_robot import XHandRobot  # noqa: E402
from mindex.policy.control_utils import (                          # noqa: E402
    interpolate_to, wait_for_key, to_absolute, ensemble_action, get_obs_retry,
)
from apps._policy_args import build_argparser                       # noqa: E402
from mindex.recording.dataset import EpisodeWriter                 # noqa: E402

INIT_STEPS = 150
INIT_HZ    = 25.0

# Same home pose used by run_teleop.py
ARM_HOME_RAD = np.deg2rad([0, -40, -40, 180, -20, 135]).astype(np.float32)

#  -45, -20, -20 , 150, -20+70, -90-45 (for book catch option 2)
# ARM_HOME_RAD = np.deg2rad([-60, -20, -20 , 150, -20+70, -90-45]).astype(np.float32) # for book catch option 2

HAND_HOME    = np.zeros(12, dtype=np.float32)


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.ensemble and args.query_mode == "sync":
        raise SystemExit("--ensemble requires --query-mode continuous or pipelined "
                         "(sync only has 1 chunk in flight at a time).")

    # ============================================================================
    # main() table of contents:
    #
    #   1. CONNECT         — hand, arm, camera, GR00T client; flag validation
    #   2. HOME + WARMUP   — home pose; fill rgb/state history buffers
    #   3. WARM-START      — first GR00T chunk; ramp from home to first action
    #   4. LOOP STATE      — chunk, locks, deques, shared bookkeeping
    #   5. HELPERS         — closures over loop state:
    #                          metrics (_hz, _query_ms, _latency_ticks)
    #                          query   (_snapshot_inpaint, _do_query, _store_result)
    #                          workers (_worker_continuous, _pipelined, _vlm_cache)
    #   6. SPAWN WORKERS   — start background threads per query_mode
    #   7. MAIN LOOP       — observe → kickoff → resolve → safety → send
    # ============================================================================

    # ============================================================================
    # 1. CONNECT — hand, arm, camera, GR00T client
    # ============================================================================
    if args.dry_run:
        # Dry-run: skip real XHand connection (no robot hardware needed).
        class _FakeHand:
            num_dofs = 12
            _last = HAND_HOME.copy()
            def connect(self): pass
            def reset_sensors(self): pass
            def get_observation(self, fresh: bool = False):
                # Stub all fields the 45-state config might ask for. Zeros for
                # torques + fingertip_force are fine for dry-run (no real sensors).
                return {
                    "joint_positions": self._last.copy(),
                    "joint_torques": np.zeros(12, dtype=np.float32),
                    "fingertip_force": np.zeros((5, 3), dtype=np.float32),
                }
            def send_action(self, a):
                self._last = np.asarray(a, dtype=np.float32)
            def disconnect(self): pass
        hand = _FakeHand()
        print("(dry-run) using fake hand — real XHand not connected.")
    else:
        print(f"Opening XHand via {args.protocol}...")
        hand = XHandRobot(protocol=args.protocol, serial_port=args.serial_port, baud_rate=args.baud_rate)
        hand.connect()
        hand.reset_sensors()

    arm = None
    if not args.no_arm and not args.dry_run:
        from mindex.robots.xarm_sdk import XArmSDK
        print(f"Connecting to xArm at {args.arm_ip} (collision_sensitivity={args.collision_sensitivity})...")
        arm = XArmSDK(ip=args.arm_ip,
                      collision_sensitivity=args.collision_sensitivity,
                      joint_jerk=args.joint_jerk,
                      joint_maxacc=args.joint_maxacc)
        arm.connect()
    elif args.dry_run and not args.no_arm:
        # In dry-run, no real xArm connection — use a stub.
        class _FakeArm:
            num_dofs = 6
            _last = ARM_HOME_RAD.copy()
            def get_observation(self, fresh: bool = False):
                return {"joint_positions": self._last.copy()}
            def send_action(self, a):
                self._last = np.asarray(a, dtype=np.float32)
            def disconnect(self):
                pass
        arm = _FakeArm()
        print("(dry-run) using fake arm — real xArm not connected.")
    arm_dofs = arm.num_dofs if arm else 0

    from mindex.cameras.realsense import CameraClient
    print(f"Camera client → {args.cam_host}:{args.cam_port}")
    cam = CameraClient(host=args.cam_host, port=args.cam_port)

    from mindex.policy.groot_client import GR00TClient
    print(f"Connecting to GR00T (DiT) at {args.host}:{args.port}...")
    if args.vlm_host is not None or args.vlm_port is not None:
        vh = args.vlm_host or args.host
        vp = args.vlm_port if args.vlm_port is not None else args.port
        print(f"  also connecting to GR00T (VLM) at {vh}:{vp}  [2-server mode]")
    client = GR00TClient(host=args.host, port=args.port, task=args.task,
                         vlm_host=args.vlm_host, vlm_port=args.vlm_port,
                         control_period_ms=1000.0 / args.rate_hz)

    # ---- eval logging (--eval-log): per-tick recording + query timestamps ----
    # Reuse EpisodeWriter so logs are HDF5 in the same schema as teleop
    # (visualize_episode.py + render_episode_modalities.py work unchanged).
    # Client-method monkey-patch keeps the many query call sites untouched.
    eval_writer = None
    eval_query_events: list[tuple[str, float]] = []
    if args.eval_log:
        eval_writer = EpisodeWriter(args.eval_log, task=args.task or "eval")
        print(f"  eval-log: writing to {eval_writer._output_dir}/episode_{eval_writer.episode_idx:04d}.hdf5")
        for _name in ("get_action_chunk", "get_action_chunk_cached",
                       "seed_streaming_from_obs", "update_vlm_cache"):
            _orig = getattr(client, _name)
            def _make(n, o):
                def _wrapped(*a, **k):
                    eval_query_events.append((n, time.time()))
                    return o(*a, **k)
                return _wrapped
            setattr(client, _name, _make(_name, _orig))

    print(f"  ping: {client.ping()}")
    print(f"  video_T={client.video_T}  state_T={client.state_T}")

    # Clear server-side streaming buffer at episode start. No-op for non-streaming
    # (flow / non-pir2) checkpoints, but required for pir2 correctness across
    # repeated launches against the same server. Silently ignored if server is
    # not DECOUPLED=1.
    server_decoupled = True
    assert server_decoupled, "Server is not decoupled - decoupled server is a superset of non-decoupled server"
    try:
        client.reset_streaming_buffer()
        print(f"  streaming buffer reset")
    except Exception as e:
        server_decoupled = False
        print(f"  streaming buffer reset skipped: {e}")

    # Fail fast: --async-vlm requires DECOUPLED=1 endpoints. If the server
    # rejects reset_streaming_buffer (no such endpoint), the cached-VLM
    # endpoints will also be missing → don't even start the loop.
    if args.async_vlm and not server_decoupled:
        raise SystemExit(
            "--async-vlm requires a DECOUPLED=1 server (update_vlm_cache + "
            "get_action_chunk_cached endpoints). The server above doesn't "
            "have them. Restart the server with DECOUPLED=1 or drop --async-vlm."
        )

    # --inpaint: send d "during-latency" clean actions to server. Inpaint sourced
    # from current chunk being consumed: chunk[chunk_idx : chunk_idx + d] = "actions
    # robot WILL execute during inference". After swap, chunk_idx jumps to d (skip
    # past inpaint window) so client emits chunk[d : d+chunk_len] = freshly predicted.
    # d auto-derived per query from rolling _query_ms.
    #
    # --force-nonstreaming: Path A (flow loop nfe=50). Required for train-time RTC.
    #                       Inpaint goes via options["inpaint"] key.
    # default streaming:    Path B (1 substep per call, nfe = T/N - 1). Inpaint
    #                       goes via options["action"] key.
    # Note: --force-nonstreaming alone (no --inpaint) is allowed — pir2 ckpts
    # are trained on a superset of flow-matching schedules, so the non-streaming
    # flow loop is a valid baseline / ablation.
    if args.ckpt_type == "plain_flow" and args.inpaint:
        raise SystemExit("--ckpt-type plain_flow + --inpaint is OOD: plain flow "
                         "ckpts never saw clean-clamp inputs during training. "
                         "Drop --inpaint, or set --ckpt-type {pir2,rtc} if the "
                         "ckpt was trained with streaming.")
    if args.ckpt_type == "rtc" and not args.force_nonstreaming:
        raise SystemExit("--ckpt-type rtc requires --force-nonstreaming: train-"
                         "time RTC was trained with the flow-loop schedule, not "
                         "with streaming inference.")
    if args.inpaint and args.query_mode == "sync":
        raise SystemExit("--inpaint + --query-mode sync is incoherent: sync "
                         "freezes the robot during the query, so there is no "
                         "during-latency window for the front-d clean clamp "
                         "to represent. Use --query-mode {continuous,pipelined}.")
    if args.inpaint:
        if not args.chunk_len:
            raise SystemExit("--inpaint requires --chunk-len to be set.")
        path = "A (force_nonstreaming flow loop)" if args.force_nonstreaming else "B (streaming, K=1)"
        print(f"  inpaint enabled: Path {path}, chunk_prev sourced, skip-emit on swap")
        # FIXME(Path B): the noise schedule we run is NOT the derived schedule.
        # The derived Path B schedule (docs/path_b_schedule_viz.html §1) requires
        # the server to init the streaming buffer as a chunk_size=d staircase
        # with NO +1 offset: τ[chunk j] = noise_s * (M-1-j)/(M-1), chunk 0
        # clamped clean, M = T/d. nfe must equal K*(T/d - 1).
        #
        # The current server `_streaming_inference` inits with chunk_size=1
        # (K=T) and a +1 offset: τ[i] = noise_s * (1 - (i+1)/T). This is
        # NOT a chunk_size=d staircase, and there is no client-side knob to
        # change it. So the per-position τ the model sees during Path B does
        # not match the derived schedule's steady state.
        #
        # Path A (force_nonstreaming) is unaffected — it uses the flow loop
        # with per-position τ encoded explicitly (front d at noise_s bucket).
        # Path B needs a server-side change to plumb init_chunk_size = d and
        # drop the +1, OR an inference-time refactor. Until then, Path B
        # deployment is approximate (works only insofar as symj training band
        # covers the mismatch).
        #
        # Secondary issue: chunk_len is manual (--chunk-len) while d is
        # auto-derived (_latency_ticks). For the schedule to close, slide_steps
        # must == d, so user has to keep --chunk-len ≈ expected latency_ticks
        # by hand.

    if args.dry_run:
        print("** DRY RUN ** — actions will be logged but not sent to robots")

    # ============================================================================
    # 2. HOME + WARMUP — home pose, fill rolling rgb/state history buffers
    # ============================================================================
    if not args.no_home:
        n_dof_home = arm_dofs + hand.num_dofs
        home = np.concatenate([ARM_HOME_RAD, HAND_HOME]) if arm is not None else HAND_HOME
        print(f"Moving to home pose over {INIT_STEPS/INIT_HZ:.1f}s …")
        interpolate_to(arm, hand, home.astype(np.float32),
                       INIT_STEPS, INIT_HZ, args.dry_run, arm_dofs)
        print("At home.")

    # ---- warm up rolling history buffers ----
    video_T = client.video_T
    state_T = client.state_T
    rgb_buf   = collections.deque(maxlen=video_T)
    state_buf = collections.deque(maxlen=state_T)
    state_keys = client.state_keys  # e.g. 18-dim ['arm_joint_position','hand_joint_position']
                                    # or 45-dim ['...','...','hand_joint_torque','fingertip_force']

    def _build_flat_state(hand_obs: dict | None, arm_obs: dict | None) -> np.ndarray | None:
        """Concat per-tick obs into a flat state in client.state_keys order so
        the same client supports both 18-dim and 45-dim ckpts. Maps each
        state_key to the right field of hand_obs / arm_obs; fingertip_force is
        flattened (5,3) -> (15,). Returns None if hand_obs is None (any partial
        read means skip this tick; downstream loop retries)."""
        if hand_obs is None:
            return None
        parts: list[np.ndarray] = []
        for k in state_keys:
            if k == "arm_joint_position":
                if arm_obs is None or "joint_positions" not in arm_obs:
                    return None
                parts.append(np.asarray(arm_obs["joint_positions"], dtype=np.float32))
            elif k == "hand_joint_position":
                parts.append(np.asarray(hand_obs["joint_positions"], dtype=np.float32))
            elif k == "arm_joint_torque":
                if arm_obs is None or "joint_torques" not in arm_obs:
                    return None
                parts.append(np.asarray(arm_obs["joint_torques"], dtype=np.float32))
            elif k == "hand_joint_torque":
                parts.append(np.asarray(hand_obs["joint_torques"], dtype=np.float32))
            elif k == "fingertip_force":
                parts.append(np.asarray(hand_obs["fingertip_force"], dtype=np.float32).reshape(-1))
            else:
                raise RuntimeError(f"Unknown state_key {k!r}; extend _build_flat_state.")
        return np.concatenate(parts, axis=0).astype(np.float32)

    print("Warming up buffers...")
    while len(rgb_buf) < video_T or len(state_buf) < state_T:
        # same hand→arm→cam order as the main loop / teleop recording
        obs     = hand.get_observation(fresh=True)
        arm_obs = arm.get_observation(fresh=True) if arm else None
        frame   = cam.get()
        if frame is not None:
            rgb_buf.append(frame["rgb"])
        flat = _build_flat_state(obs, arm_obs)
        if flat is not None:
            state_buf.append(flat)
        time.sleep(1.0 / 30)

    # ============================================================================
    # 3. WARM-START — first GR00T chunk, ramp robot from home to first action
    # ============================================================================
    print("Querying first action chunk for ramp-up …")
    rgb_hist   = np.stack(list(rgb_buf),   axis=0)
    state_hist = np.stack(list(state_buf), axis=0)
    # Warmup query — no prev_chunk yet, so RTC inactive on first call.
    # Pass slide_steps=chunk_len so the server seeds the buffer with the same
    # cycle close shape that subsequent streaming calls will use. Without this,
    # v2 ckpts seeded at d=1 default cause a train-test mismatch on first cycle.
    first_chunk, _ = client.seed_streaming_from_obs(
        rgb_hist, state_hist, t_image_capture=time.time(),
        slide_steps=args.chunk_len)
    print("  warm-start: clean chunk via flow + buffer seeded (pir2 only)")
        
    
    # Truncation is for non-ensemble only. In ensemble mode we keep full chunks
    # so each contributes its full validity window to the rolling buffer.
    # --chunk-len governs swap/firing cadence only, not storage; chunks are
    # always kept at full server horizon (typically H=50).
    # Home-ramp endpoint = the position the streaming loop will FIRST EMIT, not
    # chunk[0]. For pir2 streaming the first emit is chunk[_effective_d() ≈
    # chunk_len] (today's chunk_idx=_effective_d fix); ramping to chunk[0] would
    # leave a ~chunk_len-position gap between robot's actual pose at loop-start
    # and the first commanded action, causing initial jitter. For RTC / Path A
    # the first emit is chunk[0] (stale=0 at episode start), so use chunk[0].
    _is_pir2_stream = (args.ckpt_type == "pir2") and not args.force_nonstreaming
    _first_idx = max(1, int(args.chunk_len or 1)) if _is_pir2_stream else 0
    _first_idx = min(_first_idx, first_chunk.shape[0] - 1)
    first_action = to_absolute(first_chunk[_first_idx], state_hist[-1],
                                args.arm_mode, args.hand_mode, arm_dofs)

    print(f"  first action: arm={first_action[:6].round(3)}  hand={first_action[6:].round(3)}")
    if not wait_for_key("Ready to ramp to first predicted action. SPACE/Enter=go, q=quit  "):
        print("\rAborted before ramp.")
        return
    print(f"\rRamping from home → first action over {INIT_STEPS/INIT_HZ:.1f}s …")
    interpolate_to(arm, hand, first_action.astype(np.float32),
                   INIT_STEPS, INIT_HZ, args.dry_run, arm_dofs)

    if not wait_for_key("At first action. SPACE/Enter to start policy loop, q to quit  "):
        print("\rAborted before policy loop.")
        return

    # Re-seed streaming buffer with fresh obs at robot's CURRENT pose (= just-ramped
    # first_action position), not the home pose. The original warmup ran Path A on
    # an obs captured at home; that buf's "past d actions" memory now mismatches
    # the robot's actual pose at policy-loop start (which is first_action[d] from
    # the home-ramp), causing initial-cycle jitter from buf-state misalignment.
    # Re-seeding aligns the model's past context with the robot's current state.
    _is_pir2_stream = (args.ckpt_type == "pir2") and not args.force_nonstreaming
    if _is_pir2_stream and False:
        print("\rRe-seeding streaming buffer with post-home obs ...")
        # Refresh rgb/state buffers with current obs (robot at first_action now).
        rgb_buf.clear()
        state_buf.clear()
        while len(rgb_buf) < video_T or len(state_buf) < state_T:
            obs     = hand.get_observation(fresh=True)
            arm_obs = arm.get_observation(fresh=True) if arm else None
            frame   = cam.get()
            if frame is not None: rgb_buf.append(frame["rgb"])
            flat = _build_flat_state(obs, arm_obs)
            if flat is not None:
                state_buf.append(flat)
            time.sleep(1.0 / 30)
        rgb_hist_post   = np.stack(list(rgb_buf),   axis=0)
        state_hist_post = np.stack(list(state_buf), axis=0)
        first_chunk, _ = client.seed_streaming_from_obs(
            rgb_hist_post, state_hist_post, t_image_capture=time.time(),
            slide_steps=args.chunk_len)
        print("  re-seeded: buf aligned with current pose")

    # In 2-server async-vlm mode, the client maintains its OWN copy of vl_embeds
    # received from the VLM server. seed_streaming_from_obs (above) runs the VLM
    # forward on the DiT server's GPU internally — it primes the DiT server's
    # cache but doesn't populate the client's local _vl_embeds. Without a primed
    # client cache, the first get_action_chunk_cached call races the VLM cache
    # worker and may fire BEFORE the worker's first update_vlm_cache completes
    # → "vl_embeds not cached yet" RuntimeError. Block here for one explicit
    # update_vlm_cache to populate the client cache before the policy loop.
    if args.async_vlm and getattr(client, "_two_server", False):
        print("\rPriming client-side VLM cache (2-server mode) ...")
        try:
            rgb_hist_prime = np.stack(list(rgb_buf), axis=0)
            state_hist_prime = np.stack(list(state_buf), axis=0)
            client.update_vlm_cache(
                rgb_hist_prime, state_hist_prime, args.task,
                t_image_capture=time.time(),
            )
            print("  primed: client-side vl_embeds ready")
        except Exception as e:
            print(f"  WARNING: priming failed ({e}); first query may race the cache worker")

    print("\rStarting policy loop.")




    # ============================================================================
    # 4. LOOP STATE — chunk, locks, deques, shared bookkeeping
    # ============================================================================
    # Seed the ensemble buffer with the warmup chunk so the loop has something
    # before any worker query finishes.
    _first_t_obs = time.perf_counter()

    chunk: np.ndarray = first_chunk
    chunk_idx = 0
    _last_swap_chunk_idx = 0   # chunk_idx right after most recent swap; +chunk_len = next swap target
    period = 1.0 / args.rate_hz
    stop = False
    last_action: np.ndarray | None = first_action.copy()

    def _sigint(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    n_dof = arm_dofs + hand.num_dofs
    print(f"Running policy @ {args.rate_hz:.0f} Hz, {n_dof}-DOF. Ctrl-C to exit.")

    # ============================================================================
    # 5. HELPERS (closures over the state above)
    # ============================================================================

    # ---- 5a. metric helpers ----
    _times:       collections.deque = collections.deque(maxlen=100)
    _query_times: collections.deque = collections.deque(maxlen=20)
    # Image-staleness diagnostics from cached path: how many ms / ticks old is
    # the VLM-cached image vs the current state at each get_action_chunk_cached.
    # Populated only when --async-vlm (other paths re-encode image every call).
    _image_delay_ms: collections.deque = collections.deque(maxlen=20)
    # Rolling buffer of actions sent to the robot. Used by _snapshot_inpaint
    # so the server clamps front d to the actions the robot actually executed.
    _emit_history: collections.deque = collections.deque(maxlen=max(args.chunk_len or 50, 50))
    def _hz() -> float:
        if len(_times) < 2:
            return 0.0
        return (len(_times) - 1) / (_times[-1] - _times[0])
    def _query_ms() -> tuple[float, float]:
        if not _query_times:
            return (0.0, 0.0)
        arr = np.asarray(_query_times)
        return (float(arr.mean()), float(arr[-1]))

    # ---- 5b. query worker shared state (locks, queues, current chunk ref) ----
    # All three modes (sync / continuous / pipelined) share these structures.
    obs_lock     = threading.Lock()
    chunk_lock   = threading.Lock()
    latest_obs: dict = {"rgb": None, "state": None, "stamp": 0.0}
    next_chunk: list = [None]                          # used by non-ensemble path
    chunk_buffer = collections.deque(maxlen=args.ensemble_buffer_size)
    # Shared (chunk, chunk_idx) for inpaint sourcing. Worker reads to snapshot
    # the next d positions (= "during-latency" actions robot will execute).
    # Main loop updates each tick after determining chunk/chunk_idx.
    current_chunk_ref: list = [None, 0]                # [chunk_arr, chunk_idx]
    _last_emit_raw: list = [None]                      # debug: prev tick's raw emit (boundary Δ)
    worker_stop  = threading.Event()
    submit_event = threading.Event()    # used by pipelined mode to wake the worker

    # ---- 5c. query helpers (read obs, call GR00T, hand result back) ----
    def _latency_ticks() -> int:
        """Auto-derive d from rolling _query_ms. d = ceil(mean_query_ms / period_ms)."""
        if not _query_times:
            return max(1, int(args.chunk_len or 1))      # initial guess = chunk_len
        mean_ms = float(np.mean(_query_times))
        return max(1, int(np.round(mean_ms / (period * 1000.0))))

    def _effective_d() -> int:
        """The d we feed to the server on the next call (both as slide_steps AND
        as inpaint count). Behavior depends on ckpt type AND query mode:

          - **RTC (Path A, force_nonstreaming or --ckpt-type rtc)**: returns the
            MEASURED latency only. RTC was trained with front-d clean = "actions
            the robot actually executed during this call's inference latency",
            so inpaint count must match measured_d. chunk_len for RTC only
            controls query cadence (when to swap chunks), not the model's
            expected past length.

          - **Streaming pir2 in continuous mode**: returns measured. Worker
            queries back-to-back, eager swap fires per measured ticks, so
            consumption rate = measured. Using max(measured, chunk_len) would
            slide server > client consumes → positions lost per cycle.

          - **Streaming pir2 in sync/pipelined**: returns max(measured,
            chunk_len). chunk_len is the schedule's clean-prefix length;
            measured floor catches latency spikes so schedule adapts upward.
            On first call _latency_ticks() falls back to chunk_len, so
            _effective_d() = chunk_len. Bootstrapped value matches the seed
            in seed_streaming_from_obs(slide_steps=chunk_len).
        """
        measured = _latency_ticks()
        # measured = min(measured, int(args.chunk_len))
        is_rtc = bool(args.force_nonstreaming) or (args.ckpt_type == "rtc")
        if is_rtc or args.query_mode == "continuous":
            return measured
        return max(measured, max(1, int(args.chunk_len or 1))) # this is for pir2 pipelined mode

    def _emit_width(fallback_len: int = 0) -> int:
        """Width of the emit slice per swap (= swap_target - chunk_idx).

        Returns _effective_d() ONLY for pir2 streaming, where the schedule's
        freshly-cleaned region is exactly [d : 2d] (positions beyond 2d are
        under-denoised) so emit width MUST equal slide_steps.

        For RTC and plain_flow, the model produces a WIDE freshly-denoised
        region ([d:T], all clean), so the robot can emit anywhere in it. Use
        args.chunk_len (or len(chunk) fallback) — the scheduled-swap then acts
        as a far safety net and eager-swap drives cadence in continuous mode.
        Forcing _effective_d() here makes scheduled-swap fire every ~measured
        ticks, racing the (often higher) RTC query latency -> spurious
        swap-fallback stalls. So RTC/plain_flow keep chunko_len."""
        is_pir2_stream = (args.ckpt_type == "pir2") and not args.force_nonstreaming
        if is_pir2_stream or args.query_mode == "continuous":  #?? why is this here? ??? tomorrow  try 
            return _effective_d()
        return args.chunk_len if args.chunk_len else fallback_len

    def _snapshot_inpaint() -> np.ndarray | None:
        """Inpaint front [0:d] for the next chunk. The invariant:

            inpaint[d-1] must equal the robot's pose at swap = ck[ci + lat - 1].

        So emit[d] continues forward from where the robot actually is. Anchoring
        the front at the kickoff pose (history[-d:]) puts the continuation point
        `lat` steps BEHIND the robot -> backward kick -> visible oscillation at
        each swap (the symptom we chased for RTC and pir2).

        Slice formula: inpaint = ck[ci + lat - d : ci + lat], a d-wide window
        of the currently-executing chunk that ENDS at the swap index.

        Special cases:
          - RTC (d == lat): start = ci, end = ci+d -> ck[ci:ci+d]. The earlier
            RTC-only fix is exactly this special case where "start at ci" and
            "end at ci+lat" coincide because d == lat.
          - pir2 (d = chunk_len > lat): start = ci + lat - d < ci. In steady
            state ci is near 2d (just before swap), so start = lat + (ci - d)
            >= lat >= 0 -- the entire slice sits inside the chunk's freshly-
            cleaned [d:2d] window. _effective_d's guarantee d >= lat is what
            makes start <= ci hold. -> not used tho 

        Fallbacks:
          - start < 0 (warmup, ci small): prepend (-start) past actions from
            _emit_history to cover the pre-chunk portion.
          - No current chunk yet (cold start): fall back to history[-d:].
        """
        if not args.inpaint:
            return None
        d = _effective_d()
        if d <= 0:
            return None
        lat = _latency_ticks() ## ?? why is this here? for pir2 pipelined, it should just be d, but it is fine as inpaint not supported for pir2.
        with chunk_lock:
            ck = current_chunk_ref[0]
            ci = int(current_chunk_ref[1])
        if ck is not None:
            end = min(ci + lat, ck.shape[0])
            start = end - d # slightly wrong ?? , more like start ~ start + d and if start + d > end, then repeat the last action
            if start >= 0:
                return ck[start:end].astype(np.float32)
            # start < 0: prepend (-start) past actions from history.
            need_past = -start
            if len(_emit_history) >= need_past and end > 0:
                past = np.stack(list(_emit_history)[-need_past:], axis=0).astype(np.float32)
                future = ck[:end].astype(np.float32)
                return np.concatenate([past, future], axis=0)
        # Cold start (no current chunk yet): fall back to history[-d:].
        if len(_emit_history) >= d:
            return np.stack(list(_emit_history)[-d:], axis=0).astype(np.float32)
        return None

    def _do_query():
        """Snapshot obs, call GR00T, record latency.
        Returns (chunk, obs_time) or None if obs not ready."""
        with obs_lock:
            rgb  = latest_obs["rgb"]
            st   = latest_obs["state"]
            t_obs = latest_obs["stamp"]
            t_wall = latest_obs.get("t_wall")    # CLIENT wall-clock capture time
        # rgb is needed even in async_vlm: the server uses cached VLM features
        # but its processor still expects video+language in the obs dict to
        # build the input template (~5-15 ms CPU image transform on server).
        if rgb is None or st is None:
            return None
        # Inpaint: snapshot d during-latency actions from chunk being consumed.
        # Use the SAME d for both slide_steps (schedule shape) and inpaint
        # count (front-clamp count) so the schedule's clean prefix matches
        # the number of past actions we're pinning.
        d_eff = _effective_d()
        inp = _snapshot_inpaint()
        if args.debug_swap and inp is not None:
            _ia, _ih = inp[:, :arm_dofs], inp[:, arm_dofs:]
            print(f"  [inpaint] d={inp.shape[0]} "
                  f"arm[{_ia.min():.3f},{_ia.max():.3f}] "
                  f"hand[{_ih.min():.3f},{_ih.max():.3f}]", flush=True)
        _q0 = time.perf_counter()
        try:
            if args.async_vlm:
                # Server uses cached VLM features but its processor pipeline
                # still expects video + state + language in the obs dict to
                # build the input template. Pass the same rgb as the
                # full-query path.
                c, _info = client.get_action_chunk_cached(
                    rgb, st,
                    num_inference_timesteps=args.nfe,
                    slide_steps=d_eff,
                    inpaint_actions=inp,
                    force_nonstreaming=args.force_nonstreaming,
                    t_state_capture=t_wall,    # state captured at obs publish time
                )
                if _info.get("image_delay_ms") is not None:
                    _image_delay_ms.append(_info["image_delay_ms"])
            else:
                c = client.get_action_chunk(
                    rgb, st,
                    num_inference_timesteps=args.nfe,
                    slide_steps=d_eff,
                    inpaint_actions=inp,
                    force_nonstreaming=args.force_nonstreaming,
                )
        except Exception as e:
            print(f"\nquery failed: {e}", file=sys.stderr)
            return None
        _query_times.append((time.perf_counter() - _q0) * 1000)
        return (c, t_obs)

    def _store_result(c):
        """Worker → main handoff. Ensemble: append to ring buffer.
        Non-ensemble: overwrite single slot.
        For pipelined, clear pending guard on LAND (not on swap) so the
        next kickoff can fire pre-emptively to align with swap_target."""
        with chunk_lock:
            if args.ensemble:
                chunk_buffer.append(c)              # (chunk, t_obs); deque auto-evicts
            else:
                next_chunk[0] = c
            if args.query_mode == "pipelined":
                pipelined_kickoff_pending[0] = False

    # ---- 5d. swap helpers (shared across the 3 swap sites) ----
    def _pop_ready_chunk():
        """Pop the worker-delivered (chunk, t_obs) tuple if any; else None."""
        with chunk_lock:
            ready = next_chunk[0]
            next_chunk[0] = None
        return ready

    def _post_swap(c_arr, t_obs, label):
        """Common post-swap logic: compute initial chunk_idx for a new chunk
        (latency-skip + skip-emit), update _last_swap_chunk_idx, debug print.
        Returns chunk_idx. Pass t_obs=None for sync (no latency skip)."""
        nonlocal _last_swap_chunk_idx
        if t_obs is None:
            stale = 0
        else:
            stale = int(round((time.perf_counter() - t_obs) / period))

        # For pir2 streaming (Path B), the schedule produces d freshly-cleaned
        # positions per call at indices [d, 2d). Always emit from there, NOT from
        # the time-indexed stale offset. Sync mode's stale=0 would otherwise emit
        # chunk[0:d] = prior cycle's frozen front (1-cycle delay). Pipelined
        # with stale > d would otherwise emit from the ramp region (partial
        # denoise). Both pathologies avoided by anchoring at d = _effective_d().
        #
        # For RTC / Path A: still use time-indexed stale skip (no cycle close
        # structure; each call independent; emit must match temporal order).
        is_pir2_stream = (args.ckpt_type == "pir2") and not args.force_nonstreaming
        if is_pir2_stream:
            new_idx = min(_effective_d(), len(c_arr) - 1)
        else:
            new_idx = min(max(stale, 0), len(c_arr) - 1)
            # new_idx = min(_effective_d(), len(c_arr) - 1)
            # TODO DEBUG  ?? why do we need this? ??????
            if args.inpaint:
                new_idx = min(max(new_idx, _latency_ticks()), len(c_arr) - 1)
                # new_idx = min(max(new_idx, _latency_ticks()), len(c_arr) - 1)
        _last_swap_chunk_idx = new_idx
        if args.debug_swap:
            # Emit width = slide_steps for pir2 streaming (so [d_eff : 2*d_eff]
            # semantic holds and there's no server-vs-client drift); else chunk_len.
            new_target = new_idx + _emit_width(fallback_len=len(c_arr))
            print(f"  [{label}] step={step_idx} stale={stale} chunk_idx={new_idx} "
                  f"swap_target={new_target} T={len(c_arr)}", flush=True)
            # Separate intra-window coarseness from at-boundary jump.
            lo, hi = new_idx, min(new_target, len(c_arr))
            if hi - lo >= 2:
                seg = np.asarray(c_arr[lo:hi], dtype=np.float32)
                dseg = np.abs(np.diff(seg, axis=0))
                am, hm = float(dseg[:, :arm_dofs].max()), float(dseg[:, arm_dofs:].max())
                bnd = ""
                if _last_emit_raw[0] is not None:
                    bd = np.abs(np.asarray(c_arr[lo], dtype=np.float32) - _last_emit_raw[0])
                    bnd = (f" boundary_maxd arm={float(bd[:arm_dofs].max()):.4f} "
                           f"hand={float(bd[arm_dofs:].max()):.4f}")
                print(f"    [chunk-diag] emit[{lo}:{hi}] within_chunk_maxd "
                      f"arm={am:.4f} hand={hm:.4f}{bnd}", flush=True)
        return new_idx

    def _do_sync_query():
        """Sync mode's blocking query: build obs history → call GR00T → record
        latency. Returns the new chunk. Mirrors _do_query but builds obs from
        the per-tick rgb_buf/state_buf (not the latest_obs snapshot) since the
        main loop is the only caller here.

        No inpaint_actions: sync mode has no chunk overlap (the previous chunk
        is fully sent before this query fires). The buffer's post-slide state
        already matches what the robot received, so inpaint would just rewrite
        buf[:d] with the same chunk[:d] from the previous cycle. Async mode
        needs inpaint because new chunks are queried mid-execution of the
        previous chunk and the front d positions must be re-pinned to the
        actually-emitted (post-EMA) actions.
        """
        state_hist = np.stack(list(state_buf), axis=0)
        rgb_hist   = np.stack(list(rgb_buf),   axis=0)
        t_wall     = time.time()
        kw = dict(
            num_inference_timesteps=args.nfe,
            slide_steps=args.chunk_len,
            force_nonstreaming=args.force_nonstreaming,
        )
        _q0 = time.perf_counter()
        if args.async_vlm:
            chunk, _info = client.get_action_chunk_cached(
                rgb_hist, state_hist, t_state_capture=t_wall, **kw)
            if _info.get("image_delay_ms") is not None:
                _image_delay_ms.append(_info["image_delay_ms"])
        else:
            chunk = client.get_action_chunk(rgb_hist, state_hist, **kw)
        _query_times.append((time.perf_counter() - _q0) * 1000)
        return chunk

    # ---- 5e. worker thread entry points (one of these runs in background) ----
    def _worker_continuous():
        """Mode C: query back-to-back on freshest obs."""
        last_stamp = -1.0
        while not worker_stop.is_set():
            with obs_lock:
                stmp = latest_obs["stamp"]
            if stmp == last_stamp:
                time.sleep(0.005)
                continue
            # min_interval = _effective_d() * period
            # if time.perf_counter() - last_stamp < min_interval:
            #     time.sleep(0.005)
            #     continue
            c = _do_query()
            if c is not None:
                _store_result(c)
            target = _effective_d() * period    # still = measured, no floor — this is the whole point
            sleep = target - (time.perf_counter() - last_stamp)
            if sleep > 0:
                time.sleep(sleep)

    def _worker_pipelined():
        """Mode B: query only when main loop signals (kicked off mid-chunk)."""
        while not worker_stop.is_set():
            if not submit_event.wait(timeout=0.05):
                continue
            submit_event.clear()
            c = _do_query()
            if c is not None:
                _store_result(c)

    def _worker_vlm_cache():
        """async-vlm: continuously refresh server-side VLM cache with latest
        published rgb. Free-running: fires as fast as VLM forward completes
        (~10 Hz at ~100 ms per forward). Decoupled from main loop tick rate.

        Sends state alongside rgb+language: server only uses video+language
        for the VLM forward, but its _collate_observation pipeline iterates
        over all three modalities and KeyError's without a 'state' entry."""
        while not worker_stop.is_set():
            with obs_lock:
                rgb = latest_obs["rgb"]
                st  = latest_obs["state"]
                t_wall = latest_obs.get("t_wall")   # CLIENT image-capture wall time
            if rgb is None or st is None:
                time.sleep(0.005)
                continue
            try:
                client.update_vlm_cache(rgb, st, args.task, t_image_capture=t_wall)
            except Exception as e:
                print(f"\n[vlm_worker] update_vlm_cache failed: {e}", file=sys.stderr)
                time.sleep(0.1)

    # ============================================================================
    # 6. SPAWN WORKERS — start background threads per query_mode (+ async-vlm)
    # ============================================================================
    # Query-mode worker (one of three). DiT-only vs full VLM+DiT is decided
    # inside _do_query() based on args.async_vlm.
    if args.query_mode == "continuous":
        threading.Thread(target=_worker_continuous, daemon=True).start()
        print("GR00T mode: continuous (background queries non-stop)")
    elif args.query_mode == "pipelined":
        threading.Thread(target=_worker_pipelined, daemon=True).start()
        print("GR00T mode: pipelined (one query per chunk, kickoff timed from measured latency)")
    else:
        print("GR00T mode: sync (blocking)")

    # async-vlm is orthogonal: spawn the VLM-cache thread under ANY query mode.
    if args.async_vlm:
        threading.Thread(target=_worker_vlm_cache, daemon=True, name="vlm-cache").start()
        print("Async VLM: background thread refreshing server-side VLM cache (free-running)")

    # Use the first-chunk query latency as the initial estimate; it'll be
    # refined by the rolling average once the loop is running.
    pipelined_kickoff_pending: list = [False]   # one-shot guard per chunk

    # Seed the ensemble buffer with the warmup chunk so we have something to
    # serve in the first ticks before any worker query has landed.
    if args.ensemble:
        chunk_buffer.append((first_chunk.astype(np.float32), _first_t_obs))
        print(f"Ensemble: buffer maxlen={args.ensemble_buffer_size}, "
              f"m={args.ensemble_m}, "
              f"first_chunk seeded with len={len(first_chunk)}")

    # ============================================================================
    # 7. MAIN LOOP — observe → kickoff → resolve → safety → send → sleep
    # ============================================================================
    try:
        step_idx = 0
        while not stop:
            t0 = time.perf_counter()

            # ============================================================
            # OBSERVE (fresh tick-synchronous reads — order matches teleop)
            # ============================================================
            # Teleop reads: hand(fresh) → arm(fresh) → cam(cached). Keep same
            # order at inference so train/test state-vs-image alignment
            # matches (image sampled AFTER joint state, ~13 ms newer).
            obs     = hand.get_observation(fresh=True)
            arm_obs = arm.get_observation(fresh=True) if arm else None
            frame   = cam.get()

            if frame is not None:
                rgb_buf.append(frame["rgb"])
            flat = _build_flat_state(obs, arm_obs)
            if flat is not None:
                state_buf.append(flat)


            # ============================================================
            # PUBLISH latest_obs for background workers
            # ============================================================
            # Consumers: continuous/pipelined query workers + async-vlm worker.
            # Gate fires under any async-style scheduling.
            if ((args.query_mode != "sync" or args.async_vlm)
                    and len(rgb_buf) == video_T and len(state_buf) == state_T):
                with obs_lock:
                    latest_obs["rgb"]   = np.stack(list(rgb_buf),   axis=0)
                    latest_obs["state"] = np.stack(list(state_buf), axis=0)
                    latest_obs["stamp"] = time.perf_counter()
                    # Wall-clock capture timestamp passed to the server so
                    # downstream staleness is computed in the CLIENT's clock
                    # domain (perf_counter doesn't cross-machine).
                    latest_obs["t_wall"] = time.time()

            # ============================================================
            # PIPELINED: kickoff next query
            # ============================================================
            # Non-ensemble: event-based — fire when chunk_idx is need_steps
            # ticks away from swap_target so the new chunk lands right at swap.
            # Ensemble: time-based — chunk_idx isn't advanced (ensemble uses
            # rolling buffer, not chunk_idx indexing), so use period-based fire.
            # Unified kickoff: same event-based trigger for both ensemble and
            # non-ensemble. Fire when chunk_idx is need_steps away from the next
            # swap_target so the new chunk lands JUST in time for the swap.
            # (In ensemble mode chunk_idx + swap_target are virtual — chunk_idx
            # advances per tick like a counter, and the virtual swap below
            # resets _last_swap_chunk_idx to keep the rhythm. They aren't used
            # for emit in ensemble, only for kickoff timing.)
            if (args.query_mode == "pipelined"
                    and not pipelined_kickoff_pending[0]
                    and next_chunk[0] is None):

                q_mean_ms      = _query_ms()[0] if _query_times else 100.0
                need_steps     = int(np.ceil(q_mean_ms / (period * 1000)))
                eff_len        = _emit_width(fallback_len=len(first_chunk))
                kickoff_target = _last_swap_chunk_idx + eff_len
                remaining      = kickoff_target - chunk_idx

                if remaining <= need_steps:
                    submit_event.set()
                    pipelined_kickoff_pending[0] = True
                    if args.debug_swap:
                        print(f"  [kickoff] step={step_idx} chunk_idx={chunk_idx} "
                              f"swap_target={kickoff_target} remaining={remaining} "
                              f"need_steps={need_steps}", flush=True)


            # ============================================================
            # RESOLVE policy_out (single dispatch on ensemble vs not)
            # ============================================================
            if args.ensemble:
                # ----- ENSEMBLE: weighted avg over alive chunks in buffer -----
                with chunk_lock:
                    buf = list(chunk_buffer) 

                t_now = time.perf_counter()
                policy_out = ensemble_action(buf, t_now, period, args.ensemble_m)

                if policy_out is None:
                    # warmup: no chunks cover t_now yet. Fall back to first_chunk[0]
                    # so the loop has something to send until queries land.
                    policy_out = first_chunk[0]

                policy_out = policy_out.astype(np.float32)

                # Advance the virtual swap pointer so the unified pipelined
                # kickoff (above) sees a consistent chunk_idx → swap_target
                # rhythm. chunk_idx isn't used for emit in ensemble, just for
                # kickoff timing. On virtual swap, reset _last_swap_chunk_idx
                # so the next swap_target is eff_len ticks further out.
                chunk_idx += 1
                eff_len = args.chunk_len if args.chunk_len else len(first_chunk)
                if chunk_idx >= _last_swap_chunk_idx + eff_len:
                    _last_swap_chunk_idx = chunk_idx
                    if args.debug_swap:
                        print(f"  [virtual-swap] step={step_idx} chunk_idx={chunk_idx} "
                              f"next swap_target={chunk_idx + eff_len}", flush=True)

            else:
                # ----- NON-ENSEMBLE: (maybe swap chunk) → emit chunk[chunk_idx] -----

                # Compute swap target once so both the continuous eager-swap and
                # the scheduled-swap block below share the same gate.
                _emit_w = _emit_width()
                swap_target = (_last_swap_chunk_idx + _emit_w
                               if _emit_w else len(chunk))

                # 1. CONTINUOUS: eager swap — take the freshest worker chunk,
                # but ONLY once the current chunk's d positions have been fully
                # consumed (chunk_idx >= swap_target). Swapping earlier violates
                # the pir2 contract (server slid schedule by d assuming client
                # consumed d) and causes chunk-space motion to run ahead of
                # wall-clock rate.
                if args.query_mode == "continuous" and chunk_idx >= swap_target: # ?? why is this here? -> might be correct for ours, no need for RTC
                    
                    
                    
                    ready = _pop_ready_chunk()
                    if ready is not None:
                        chunk = ready[0]
                        chunk_idx = _post_swap(chunk, ready[1], "eager-swap")

                # 2. SCHEDULED swap when chunk_idx hits swap_target.
                if chunk_idx >= swap_target:
                    if args.query_mode == "sync":
                        # Blocking query (robot frozen during it).
                        try:
                            chunk = _do_sync_query()
                        except Exception as e:
                            print(f"\n[sync swap] query failed: {e}", file=sys.stderr)
                            chunk_idx = len(chunk) - 1
                            continue
                        chunk_idx = _post_swap(chunk, None, "sync-swap")
                    else:
                        # Pipelined: pop worker result if landed.
                        ready = _pop_ready_chunk()
                        if ready is not None:
                            chunk = ready[0]
                            chunk_idx = _post_swap(chunk, ready[1], "swap")
                        else:
                            # No new chunk yet -- hold last EMITTED action (not
                            # chunk[T-1], which is the un-denoised tail). Repeat
                            # last freshly-cleaned position one tick until next
                            # chunk arrives. chunk_idx was advanced past the
                            # freshly-cleaned region; rewind to last freshly
                            # cleaned position = chunk_idx - 1.
                            chunk_idx = max(0, chunk_idx - 1)
                            if args.debug_swap:
                                print(f"  [swap-fallback] step={step_idx} no chunk ready, "
                                      f"hold chunk[{chunk_idx}] (last emitted)", flush=True)

                # 3. Emit + advance. Defensive clamp guards latency spikes.
                chunk_idx = min(chunk_idx, len(chunk) - 1)
                policy_out = chunk[chunk_idx].astype(np.float32)
                chunk_idx += 1
                _last_emit_raw[0] = policy_out

            # ============================================================
            # δ → ABSOLUTE (per limb)
            # ============================================================
            # Re-anchor δ-dims to live measured state every step. Open-loop
            # δ-accumulation drifts catastrophically — verified on recorded
            # data. Latency compensation keeps executed δ aligned with "now".
            if (args.arm_mode == "delta" or args.hand_mode == "delta") and not state_buf:
                raise RuntimeError("delta mode needs at least one observed state.")

            state_now  = state_buf[-1] if state_buf else np.zeros_like(policy_out)
            raw_action = to_absolute(policy_out, state_now,
                                     args.arm_mode, args.hand_mode, arm_dofs)


            action = raw_action
            last_action = action

            _emit_history.append(action.copy() if isinstance(action, np.ndarray) else np.asarray(action))


            # ============================================================
            # SEND TO ROBOT
            # ============================================================
            if not args.dry_run:
                if arm is not None:
                    arm.send_action(action[:arm_dofs])
                    hand.send_action(action[arm_dofs:])
                else:
                    hand.send_action(action)

            # ============================================================
            # EVAL LOG — per-tick obs/state/action (schema matches teleop)
            # ============================================================
            if eval_writer is not None and obs is not None:
                combined_obs: dict = {}
                if arm_obs is not None:
                    combined_obs["joint_positions"] = np.concatenate([
                        arm_obs["joint_positions"], obs["joint_positions"]])
                else:
                    combined_obs["joint_positions"] = obs["joint_positions"]
                if (arm_obs is not None
                        and "joint_torque" in arm_obs
                        and "joint_torques" in obs):
                    combined_obs["joint_torque"] = np.concatenate([
                        arm_obs["joint_torque"], obs["joint_torques"]]).astype(np.float32)
                if "fingertip_force" in obs:
                    combined_obs["fingertip_force"] = obs["fingertip_force"].astype(np.float32)
                if "xhand_tactile" in obs:
                    combined_obs["xhand_tactile"] = obs["xhand_tactile"]
                if frame is not None:
                    combined_obs["rgb"] = frame["rgb"]
                img_ts = frame["timestamp"] if frame is not None else None
                eval_writer.add_step(combined_obs, action, time.time(), img_timestamp=img_ts)

            # Inpaint shared state: publish (chunk, chunk_idx) so worker can
            # snapshot chunk_prev[chunk_idx : chunk_idx + d] as during-latency
            # inpaint at next query fire. No deque — inpaint sources from the
            # chunk being consumed, not past emits.
            if args.inpaint:
                with chunk_lock:
                    current_chunk_ref[0] = chunk
                    current_chunk_ref[1] = chunk_idx


            # ============================================================
            # STATUS + SLEEP
            # ============================================================
            _times.append(time.perf_counter())
            step_idx += 1

            if step_idx % 25 == 0:
                tag             = "DRY" if args.dry_run else "EXE"
                q_mean, q_last  = _query_ms()
                q_hz            = 1000.0 / q_mean if q_mean > 0 else 0.0
                stale_steps_est = int(round(q_mean / (period * 1000))) if q_mean > 0 else 0
                # Async-vlm only: how stale is the cached image vs current state?
                img_stale_str = ""
                if _image_delay_ms:
                    img_mean = sum(_image_delay_ms) / len(_image_delay_ms)
                    img_ticks = img_mean / (period * 1000)
                    img_stale_str = f"img_stale {img_mean:.0f}ms (~{img_ticks:.1f}tk)  "
                print(f"\r[{tag}/{args.query_mode}] step {step_idx}  "
                      f"loop {_hz():.1f}Hz  "
                      f"gr00t {q_last:.0f}ms ({q_hz:.1f}Hz, ~{stale_steps_est}-step skip)  "
                      f"{img_stale_str}"
                      f"action[0..2]={action[:3].round(3)}    ",
                      end="", flush=True)

            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                print(f"Sleeping for {sleep} seconds")
                time.sleep(sleep)

    finally:
        worker_stop.set()
        print("\nDisconnecting...")
        # Save eval log (if enabled). Stash query timestamps as HDF5 attrs
        # via set_extra_attrs — numpy arrays are valid attr values.
        if eval_writer is not None and eval_writer._buffer:
            if eval_query_events:
                kinds = np.array([e[0] for e in eval_query_events], dtype="S32")
                times = np.array([e[1] for e in eval_query_events], dtype=np.float64)
                eval_writer.set_extra_attrs({
                    "query_kinds":       kinds,
                    "query_wall_times":  times,
                })
            saved = eval_writer.save()
            print(f"eval-log saved: {saved}  "
                  f"({len(eval_writer._buffer)} ticks, {len(eval_query_events)} queries)")
        hand.disconnect()
        if arm:
            arm.disconnect()
        client.close()


if __name__ == "__main__":
    main()
