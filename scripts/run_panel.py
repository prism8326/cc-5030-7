#!/usr/bin/env python3
"""Start the interactive-beat production panel (web UI + 3 UDP apps).

    python3 scripts/run_panel.py [--host 127.0.0.1] [--port 8410]

Then open http://127.0.0.1:8410/ in a browser.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, ".pylocal"), ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from interbeat.panel import serve

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8410)
    args = ap.parse_args()
    serve(args.host, args.port)
