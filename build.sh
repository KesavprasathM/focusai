#!/bin/bash
set -e
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt