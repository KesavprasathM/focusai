#!/bin/bash
set -e

pip install torch==2.9.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt