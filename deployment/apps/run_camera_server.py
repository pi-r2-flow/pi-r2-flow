"""Launch RealSense ZMQ camera server.

    python apps/run_camera_server.py [--port 5000] [--device-id <serial>] [--flip]

Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindex.cameras.realsense import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device-id", default=None, help="RealSense serial number")
    ap.add_argument("--flip", action="store_true", help="rotate 180°")
    ap.add_argument("--depth", action="store_true", help="enable depth stream (requires USB3)")
    args = ap.parse_args()
    serve(port=args.port, device_id=args.device_id, flip=args.flip, depth=args.depth)


if __name__ == "__main__":
    main()
