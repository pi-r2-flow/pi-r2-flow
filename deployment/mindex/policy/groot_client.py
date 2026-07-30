"""GR00T VLA policy client.

Matches NVIDIA Isaac-GR00T PolicyServer wire format (this fork —
gr00t/policy/server_client.py uses np.save/.npy for arrays and custom class
markers, NOT msgpack_numpy):

  numpy ndarray  → {"__ndarray_class__": True,  "as_npy":  <np.save bytes>}
  ModalityConfig → {"__ModalityConfig_class__": True, "as_json": {...}}

Start the server separately:
    python gr00t/eval/run_gr00t_server.py \\
        --model-path <ckpt> --port 5555 --host 0.0.0.0
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import msgpack
import numpy as np
import zmq


# ---- match server's MsgSerializer (this fork) -------------------------------

def _encode_custom_classes(obj: Any) -> Any:
    """Mirror of server's encode_custom_classes."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom_classes(obj: Any) -> Any:
    """Mirror of server's decode_custom_classes — recovers ndarrays and
    unwraps ModalityConfig to a plain dict (we don't import ModalityConfig)."""
    if not isinstance(obj, dict):
        return obj
    if "__ModalityConfig_class__" in obj:
        return obj.get("as_json", {})
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode_custom_classes)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_custom_classes)


# ---- client ------------------------------------------------------------------

class GR00TClient:
    """ZMQ REQ client for a running GR00T inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555,
                 task: str = "", timeout_ms: int = 60000,
                 vlm_host: str | None = None, vlm_port: int | None = None,
                 control_period_ms: float | None = None):
        """
        Args:
            host, port: address of the main (DiT) server.
            vlm_host, vlm_port: optional address of a second server dedicated
                to VLM updates (2-GPU deployment). When either is set, the
                client opens a second REQ socket; update_vlm_cache is routed
                to it with return_features=True, and get_action_chunk_cached
                attaches the returned vl_embeds when calling the main server.
                Defaults None → single-server mode (legacy).
            control_period_ms: control-loop tick period in ms (e.g. 1000/27).
                Sent to server alongside t_state_capture so it can derive
                image_delay = (t_state - t_image_cached) / period in ticks.
                Server uses it for the delay_embedding (when the ckpt has one)
                AND always reports it back in info["image_delay_ticks"] as a
                staleness diagnostic — useful for tuning vlm-refresh rate even
                on models without delay_embedding. None = skip the calculation.
        """
        self._task = task
        self._control_period_ms = control_period_ms
        ctx = zmq.Context()
        # Main socket — handles get_action, get_action_chunk_cached, reset, etc.
        self._socket = ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")
        # ZMQ REQ enforces strict send→recv→send→recv. With --async-vlm, two
        # threads (VLM worker + query worker) share this single client and
        # would race on the socket without serialization. The server is
        # single-REP anyway, so this lock matches its contract.
        self._lock = threading.Lock()

        # Optional 2-GPU / 2-server mode: separate VLM socket + local feature cache.
        self._two_server = (vlm_host is not None or vlm_port is not None)
        if self._two_server:
            vh = vlm_host or host
            vp = vlm_port if vlm_port is not None else port
            self._vlm_socket = ctx.socket(zmq.REQ)
            self._vlm_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self._vlm_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self._vlm_socket.connect(f"tcp://{vh}:{vp}")
            self._vlm_lock = threading.Lock()
        else:
            self._vlm_socket = self._socket
            self._vlm_lock = self._lock
        # Client-side cache of VLM features (used only when self._two_server).
        # Populated by update_vlm_cache, consumed by get_action_chunk_cached.
        self._vl_embeds: dict | None = None
        self._vl_cache_id: int = 0
        # Wall-clock timestamp of when the cached vl_embeds' image was captured
        # (= what the client passed as t_image_capture to update_vlm_cache, echoed
        # back by the VLM server). Forwarded to the DiT server in 2-server mode so
        # image_delay = t_state - t_image stays in the client's clock domain.
        # Without this, DiT server sees t_image=-1.0 and skips image_delay
        # computation entirely.
        self._vl_t_image_capture: float | None = None
        self._vl_embeds_lock = threading.Lock()

        cfg = self._call("get_modality_config", requires_input=False)
        video_cfg = cfg.get("video", {})
        state_cfg = cfg.get("state", {})
        lang_cfg  = cfg.get("language", {})
        self._video_T    = len(video_cfg.get("delta_indices", [0]))
        self._state_T    = len(state_cfg.get("delta_indices", [0]))
        self._video_keys = list(video_cfg.get("modality_keys", ["overhead"]))
        self._state_keys = list(state_cfg.get("modality_keys",
                                              ["arm_joint_position", "hand_joint_position"]))
        self._lang_keys  = list(lang_cfg.get("modality_keys",
                                              ["annotation.human.task_description"]))

    # ---- modality config accessors ------------------------------------------

    @property
    def video_T(self) -> int:
        """Length of the video temporal window the model expects."""
        return self._video_T

    @property
    def state_T(self) -> int:
        """Length of the state temporal window the model expects."""
        return self._state_T

    @property
    def state_keys(self) -> list[str]:
        """State modality keys in the order the server expects them.
        Use this to drive client-side state-vector assembly so the same client
        works for ckpts with different state schemas (e.g., 18-dim
        arm+hand vs 45-dim arm+hand+torque+fingertip_force)."""
        return list(self._state_keys)

    # ---- helpers -------------------------------------------------------------

    # Dimension per state-modality key. Keep in sync with what the dataset's
    # meta/modality.json declares + what training data-config consumes. Caller
    # is responsible for building a flat state_history in self._state_keys
    # order with matching dims; we slice it back into the per-key dict the
    # server expects. Add a new entry here for any new modality.
    _STATE_KEY_DIMS = {
        "arm_joint_position":  6,
        "hand_joint_position": 12,
        "arm_joint_torque":    6,
        "hand_joint_torque":   12,
        "fingertip_force":     15,   # 5 fingers × (fx, fy, fz)
    }

    @property
    def expected_state_dim(self) -> int:
        """Sum of dims for all state_keys the server expects."""
        return sum(self._STATE_KEY_DIMS[k] for k in self._state_keys)

    def _build_state_dict(self, state_history: np.ndarray) -> dict:
        """Slice a flat state_history into the per-modality dict the server
        expects, per self._state_keys in order.

        state_history must be shape (T, sum(dims)). Caller is responsible for
        concatenating per-tick obs in the order self._state_keys advertises —
        use `client.state_keys` to read that order at runtime.

        Raises RuntimeError if any state_key isn't in _STATE_KEY_DIMS, or if
        state_history's last dim doesn't match the expected sum.
        """
        for k in self._state_keys:
            if k not in self._STATE_KEY_DIMS:
                raise RuntimeError(
                    f"Unknown state modality key {k!r}. Add its dim to "
                    f"GR00TClient._STATE_KEY_DIMS and update the caller "
                    f"that builds state_history to concat it in order."
                )
        total = self.expected_state_dim
        if state_history.shape[-1] != total:
            raise RuntimeError(
                f"state_history last dim = {state_history.shape[-1]}, but "
                f"server expects {total} (sum of {self._state_keys})."
            )
        out: dict = {}
        offset = 0
        for k in self._state_keys:
            d = self._STATE_KEY_DIMS[k]
            out[k] = state_history[None, :, offset:offset + d].astype(np.float32)
            offset += d
        return out

    @property
    def state_keys(self) -> list[str]:
        """Server-declared order of state modalities. Caller MUST concat
        per-tick state in this order to match the slicing in _build_state_dict."""
        return list(self._state_keys)

    def _build_obs(self, rgb_history: np.ndarray, state_history: np.ndarray,
                   language: str | None = None) -> dict:
        """Build the 3-modality obs dict the server expects (video + state +
        language). Used by get_action_chunk, seed_streaming_from_obs,
        update_vlm_cache."""
        video_dict = {k: rgb_history[None].astype(np.uint8) for k in self._video_keys}
        state_dict = self._build_state_dict(state_history)
        lang = language if language is not None else self._task
        lang_dict = {k: [[lang]] for k in self._lang_keys}
        return {"video": video_dict, "state": state_dict, "language": lang_dict}

    def _build_options(
        self,
        num_inference_timesteps: int | None = None,
        num_inference_steps_per_call: int | None = None,
        slide_steps: int | None = None,
        inpaint_actions: np.ndarray | None = None,
        force_nonstreaming: bool = False,
    ) -> dict | None:
        """Build the server options dict from the common per-call knobs.
        Returns None if no options set (server treats as default)."""
        options: dict = {}
        if num_inference_timesteps is not None:
            options["num_inference_timesteps"] = int(num_inference_timesteps)
        if num_inference_steps_per_call is not None:
            options["num_inference_steps_per_call"] = int(num_inference_steps_per_call)
        if slide_steps is not None:
            options["slide_steps"] = int(slide_steps)
        if force_nonstreaming:
            options["force_nonstreaming"] = True
        if inpaint_actions is not None:
            inp = np.asarray(inpaint_actions, dtype=np.float32)
            if inp.ndim != 2:
                raise ValueError(
                    f"inpaint_actions must be (N, raw_action_dim); got shape {inp.shape}"
                )
            # Canonical key for both Path A (force_nonstreaming flow loop) and
            # Path B (streaming). Server routes on force_nonstreaming itself.
            # Distinct from paper-RTC's action_input["action"] (a different
            # consumer in the same flow loop).
            options["inpaint"]        = inp
            options["action_horizon"] = int(inp.shape[0])
        return options or None

    def _parse_chunk_response(self, response) -> tuple[np.ndarray, dict]:
        """Unpack a server response into (chunk[T, 18], info_dict).
        Handles both (action_dict,) and (action_dict, info) shapes, both
        {arm,hand}_joint_position pairs and single-key chunks."""
        if isinstance(response, (list, tuple)) and len(response) >= 2:
            action_dict, info = response[0], response[1]
        elif isinstance(response, (list, tuple)) and len(response) >= 1:
            action_dict, info = response[0], {}
        else:
            action_dict, info = response, {}
        if "arm_joint_position" in action_dict and "hand_joint_position" in action_dict:
            arm  = np.asarray(action_dict["arm_joint_position"])
            hand = np.asarray(action_dict["hand_joint_position"])
            if arm.ndim == 3:  arm  = arm[0]
            if hand.ndim == 3: hand = hand[0]
            chunk = np.concatenate([arm, hand], axis=-1).astype(np.float32)
        else:
            key = next(iter(action_dict))
            arr = np.asarray(action_dict[key])
            if arr.ndim == 3:
                arr = arr[0]
            chunk = arr.astype(np.float32)
        return chunk, info if isinstance(info, dict) else {}

    # ---- public API ----------------------------------------------------------

    def get_action_chunk(
        self,
        rgb_history: np.ndarray,
        state_history: np.ndarray,
        num_inference_timesteps: int | None = None,
        num_inference_steps_per_call: int | None = None,
        slide_steps: int | None = None,
        inpaint_actions: np.ndarray | None = None,
        force_nonstreaming: bool = False,
    ) -> np.ndarray:
        """Query GR00T and return a flat action chunk (T_a, 18).

        Args:
            num_inference_timesteps: override bucket count N (controls dt=noise_s/N).
                None = checkpoint default. Common: 1, 2, 4, 7, 15, 50, 100.
            num_inference_steps_per_call: for pir2 checkpoints, fixed substep count
                per server call. None = "denoise-until-front-clean" (default).
                Typical Leap recipe: --nfe 100 --substeps-per-call 1. Ignored for
                non-streaming flow.
            slide_steps: for pir2 streaming, override how many positions the server
                slides per call. Default = training-time chunk_size. Use when client
                consumes N actions per chunk (chunk_len=N) to keep the rolling buffer
                aligned with the client's consumption rate.
            inpaint_actions: latency-aware clean-action inpaint. Pass the last N
                executed actions as a (N, raw_action_dim) array in robot joint units.
                The server normalizes + pads to max_action_dim and overwrites the
                rolling buffer's front N slots at τ=noise_s. Required for RTC ckpts
                (train-time front-d clean clamp). Optional but improves pir2 cold-
                start + steady-state error materially.
        """
        obs = self._build_obs(rgb_history, state_history)
        options = self._build_options(
            num_inference_timesteps=num_inference_timesteps,
            num_inference_steps_per_call=num_inference_steps_per_call,
            slide_steps=slide_steps,
            inpaint_actions=inpaint_actions,
            force_nonstreaming=force_nonstreaming,
        )
        response = self._call("get_action",
                              data={"observation": obs, "options": options})
        chunk, _info = self._parse_chunk_response(response)
        return chunk

    def update_vlm_cache(self, rgb_history: np.ndarray,
                         state_history: np.ndarray,
                         language: str | None = None,
                         t_image_capture: float | None = None) -> dict:
        """Push fresh (rgb, state, language) to server-side VLM cache. Background-
        thread friendly: call as fast as the network allows; effective Hz =
        whatever the VLM forward takes. Returns {cache_id, t_image_capture, seq_len}.
        Requires DECOUPLED=1 server.

        ``t_image_capture`` is the CLIENT's wall-clock timestamp of when the
        image was read from the camera. Pass it to keep staleness measurements
        in a single clock domain (server reports it back verbatim). Defaults
        to time.time() at send if omitted.

        NOTE: server only uses video+language for the VLM forward, but its
        _collate_observation pipeline iterates over all three modalities and
        will KeyError without a 'state' entry. We include state (discarded
        server-side) to satisfy that contract.

        In 2-server mode (vlm_host/vlm_port set in __init__): routes the call
        to the VLM server with return_features=True, then stores the returned
        vl_embeds in a client-side cache. get_action_chunk_cached attaches
        them to the DiT server's request.
        """
        obs = self._build_obs(rgb_history, state_history, language)
        payload = {"observation": obs,
                   "t_image_capture": (t_image_capture if t_image_capture is not None
                                       else time.time())}
        if self._two_server:
            payload["return_features"] = True
        reply = self._call_on(self._vlm_socket, self._vlm_lock,
                              "update_vlm_cache", data=payload)
        if self._two_server and isinstance(reply, dict) and "vl_embeds" in reply:
            with self._vl_embeds_lock:
                self._vl_embeds = reply["vl_embeds"]
                self._vl_cache_id = int(reply.get("cache_id", -1))
                # Capture the image timestamp (in CLIENT's clock domain — server
                # echoes back the same t_image_capture we sent). Used by
                # get_action_chunk_cached to populate t_image_capture on the DiT
                # call so image_delay = t_state - t_image is well-defined.
                t_img = reply.get("t_image_capture")
                if t_img is not None:
                    self._vl_t_image_capture = float(t_img)
        return reply

    def get_action_chunk_cached(
        self,
        rgb_history: np.ndarray,
        state_history: np.ndarray,
        num_inference_timesteps: int | None = None,
        num_inference_steps_per_call: int | None = None,
        slide_steps: int | None = None,
        inpaint_actions: np.ndarray | None = None,
        force_nonstreaming: bool = False,
        t_state_capture: float | None = None,
        t_image_capture: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Per-tick query: server uses cached VLM features (saves the 80-100 ms
        VLM forward) but its processor pipeline still expects video + state +
        language in the observation dict to construct the input template. Pass
        the same `rgb_history` you'd give to get_action_chunk — it goes through
        the lightweight CPU image transform (~5-15 ms) but is NOT re-encoded by
        the VLM. The VLM features come from the most recent update_vlm_cache.

        Returns (action_chunk, info) where info contains cache_id_used,
        t_image_capture, t_state_capture for staleness diagnostics in the
        CLIENT's clock domain. Requires DECOUPLED=1 server.

        ``t_state_capture`` is the wall-clock time the state was read this
        tick. Pass it so info["t_state_capture"] echoes back in your clock.
        ``t_image_capture`` only used in cold-start or 2-server mode (when
        the client supplies vl_embeds directly); otherwise server returns
        the cached value from the last update_vlm_cache.

        STATE-ONLY FAST PATH (default): sends only {state} on the wire —
        rgb_history is accepted but discarded. Server uses its
        process_state_only path: skips image transforms + tokenization (~3-4 ms
        CPU saved) and forward_dit_action_only bypasses prepare_input's
        backbone half. Also drops the ~150 KB image payload over zmq (~5 ms
        saved on tunneled deployments). The cached vl_embeds (either local
        or supplied) carry all the visual info forward_dit needs.
        """
        del rgb_history  # accepted for API compat with the legacy full-obs path
        obs = {"state": self._build_state_dict(state_history)}    # state-only fast path
        options = self._build_options(
            num_inference_timesteps=num_inference_timesteps,
            num_inference_steps_per_call=num_inference_steps_per_call,
            slide_steps=slide_steps,
            inpaint_actions=inpaint_actions,
            force_nonstreaming=force_nonstreaming,
        )

        data = {"observation": obs, "options": options,
                "t_state_capture": (t_state_capture if t_state_capture is not None
                                    else time.time())}
        if self._control_period_ms is not None:
            data["period_ms"] = float(self._control_period_ms)
        if t_image_capture is not None:
            data["t_image_capture"] = t_image_capture
        if self._two_server:
            with self._vl_embeds_lock:
                if self._vl_embeds is None:
                    raise RuntimeError(
                        "2-server mode: vl_embeds not cached yet — call "
                        "update_vlm_cache at least once before get_action_chunk_cached."
                    )
                data["vl_embeds"] = self._vl_embeds
                # Auto-attach the cached t_image_capture so the DiT server can
                # compute image_delay = t_state - t_image in the client's clock
                # domain. Without this, DiT server sees t_image=-1.0 (legacy
                # sentinel) and image_delay reporting is suppressed.
                if "t_image_capture" not in data and self._vl_t_image_capture is not None:
                    data["t_image_capture"] = self._vl_t_image_capture
        response = self._call("get_action_chunk_cached", data=data)
        return self._parse_chunk_response(response)

    def reset_streaming_buffer(self) -> dict:
        """Clear server-side streaming-inference rolling buffer. Call between
        episodes when deploying a streaming-trained (pir2) checkpoint. No-op
        for non-streaming checkpoints. Requires server launched with DECOUPLED=1."""
        return self._call("reset_streaming_buffer", requires_input=False)

    def seed_streaming_from_obs(self, rgb_history: np.ndarray,
                                 state_history: np.ndarray,
                                 language: str | None = None,
                                 t_image_capture: float | None = None,
                                 slide_steps: int | None = None,
                                 ) -> tuple[np.ndarray, dict]:
        """Warm-start the server's streaming buffer at episode start.

        Server runs non-streaming flow on the supplied obs, then seeds the streaming
        buffer at the on-manifold point for each slot's init τ. Returns the clean
        chunk so the client can use it for the home → first_action ramp (replaces
        the get_action_chunk warmup query for pir2 ckpts).

        ``t_image_capture`` is the CLIENT's wall-clock timestamp of when the
        warm-start image was read (defaults to time.time() at send). Used to
        seed the server's VLM cache timestamp in the client's clock domain.

        ``slide_steps`` is the deployment d (sub-chunk size for streaming
        cycle close). Forwarded to seed_streaming_buffer so the seeded buf_t
        shape (v0 staircase or v2 ramp) matches what the subsequent streaming
        calls will use. Default None falls back to streaming_num_chunks on
        the server, which is d=1 for ckpts with streaming_num_chunks=-1 —
        a train-test mismatch for v2 ckpts deployed at d=5 or 10. Pass the
        same value you'll use for slide_steps in get_action_chunk calls.

        Requires server with DECOUPLED=1 and streaming-trained ckpt.
        """
        obs = self._build_obs(rgb_history, state_history, language)
        payload = {"observation": obs,
                   "t_image_capture": (t_image_capture if t_image_capture is not None
                                       else time.time())}
        if slide_steps is not None:
            payload["slide_steps"] = int(slide_steps)
        response = self._call("seed_streaming_from_obs", data=payload)
        return self._parse_chunk_response(response)

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._socket.close()
        if self._two_server:
            self._vlm_socket.close()

    # ---- internals -----------------------------------------------------------

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        """Send a request to the MAIN (DiT) server."""
        return self._call_on(self._socket, self._lock, endpoint, data, requires_input)

    def _call_on(self, socket, lock, endpoint: str, data: dict | None = None,
                 requires_input: bool = True):
        """Send a request to a specific (socket, lock) pair. Used to route
        update_vlm_cache to the VLM server in 2-server mode while keeping
        every other endpoint on the main socket."""
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        # Serialize the send→recv pair: REQ socket can't tolerate concurrent
        # send()s. _pack/_unpack are outside the critical section so CPU work
        # doesn't extend the lock-held duration beyond the I/O round-trip.
        payload = _pack(request)
        with lock:
            socket.send(payload)
            raw = socket.recv()
        response = _unpack(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response
