#!/bin/bash
# Check if the song name is given
if [ -z "$1" ]; then
  echo "No song name given"
  exit 1
fi

# Capture the argument
ARGUMENT=$1
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_EGL_DEVICE_ID=0 python pianomime/single_task/test_trained_actions.py $ARGUMENT