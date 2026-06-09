#!/bin/bash

GPU=${GPU:-0}
CONFIG=${CONFIG:-./config/wikidiverse_dual_tower.yaml}

CUDA_VISIBLE_DEVICES=$GPU python -u ./main.py --config "$CONFIG"
