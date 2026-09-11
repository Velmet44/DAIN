"""PyInstaller entry point for the node agent executable.

Runs the same code as ``python -m dain_node`` but as a standalone .exe so node
PCs need no Python/uv/git installed. The default onefile build extracts to
%TEMP% at launch (safe on OneDrive/network folders, at the cost of a slower
startup with torch/transformers); use -Mode onedir when startup latency matters.

usage (set env, then the exe):
    $env:DAIN_COORD_URL="wss://<host>"
    $env:DAIN_JOIN_TOKEN="<same as coordinator>"
    $env:DAIN_NODE_ID="pc-1"
    $env:DAIN_MODEL="dain-tiny-16L"
    $env:DAIN_MODEL_CACHE="shard_cache"
    dain-node.exe
"""

import sys

from dain_node.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
