#!/bin/bash
set -e
cd /app

# SynFit source code is expected to be mounted at /app at runtime
# (Tamarind mounts your repo contents into the container).
export PYTHONPATH=/app:/app/SynFit:${PYTHONPATH:-}

python predict.py
