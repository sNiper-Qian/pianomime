#!/bin/bash

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_EGL_DEVICE_ID=0 python pianomime/single_task/test_trained_actions.py 