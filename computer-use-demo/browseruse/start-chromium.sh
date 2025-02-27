#!/bin/bash
set -x  # Enable debug output

export DISPLAY=:99
export HOME=/home/browser
export XAUTHORITY=/home/browser/.Xauthority

# Debug information
echo "Current user: $(whoami)"
echo "Display: $DISPLAY"
echo "XAUTHORITY: $XAUTHORITY"
echo "HOME: $HOME"
ls -la /tmp/.X11-unix/
ls -la /dev/shm/
ls -la /run/dbus/
ls -la /run/user/1000/
xdpyinfo || true

# Create Chrome user directory if it doesn't exist
mkdir -p $HOME/.config/chromium
chown -R browser:browser $HOME/.config

# Start Chromium with absolute minimal flags
exec /usr/bin/chromium \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --user-data-dir=$HOME/.config/chromium \
    about:blank 