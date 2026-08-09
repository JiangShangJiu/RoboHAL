#!/usr/bin/env bash
# Idempotent Cloud Agent setup for RoboArena.
# Installs the system GL libraries MuJoCo needs and the pinned Python deps.
set -euo pipefail

cd "$(dirname "$0")/.."

# System libraries for MuJoCo:
#  - libosmesa6 / libegl1 / mesa DRI: headless (offscreen) rendering
#  - libglfw3 / libgl1: interactive mujoco.viewer when a display is present
#  - ffmpeg: encode offscreen render frames into videos
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglx-mesa0 libgl1-mesa-dri libgles2 libosmesa6 libglfw3 ffmpeg
fi

# Python dependencies (user site; persists in the environment snapshot).
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

echo "RoboArena setup complete."
