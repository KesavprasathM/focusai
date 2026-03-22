#!/bin/bash
set -e

pip install torch==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Force eventlet worker for SocketIO support
export GUNICORN_CMD_ARGS="--worker-class eventlet"