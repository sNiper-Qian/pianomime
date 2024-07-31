#!/bin/bash

WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python pianomime/multi_task/eval_low_level.py \
    --root-dir ./diffusion/eval_videos/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --trim-silence \
    --gravity-compensation \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --tqdm-bar \
    --action-reward-observation \
    --eval-episodes 1 \
    --camera-id "piano/back" \
