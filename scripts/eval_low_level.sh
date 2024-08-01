#!/bin/bash
# Check if the song name is given
if [ -z "$1" ]; then
  echo "No song name given"
  exit 1
fi

# Capture the argument
ARGUMENT=$1
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python pianomime/multi_task/eval_low_level.py $ARGUMENT