# Use Debian Bullseye as base
FROM debian:bullseye-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    git \
    python3 \
    python3-pip \
    supervisor \
    x11vnc \
    xvfb \
    openbox \
    menu \
    chromium \
    x11-utils \
    x11-xserver-utils \
    dbus-x11 \
    libpulse0 \
    libgbm1 \
    procps \
    psmisc \
    --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash browser && \
    mkdir -p /home/browser/.config/openbox && \
    chown -R browser:browser /home/browser

# Set up X11 and DBUS permissions
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix && \
    chown browser:browser /tmp/.X11-unix && \
    touch /home/browser/.Xauthority && \
    chown browser:browser /home/browser/.Xauthority && \
    mkdir -p /run/dbus && \
    chown browser:browser /run/dbus && \
    mkdir -p /run/user/1000 && \
    chown browser:browser /run/user/1000 && \
    mkdir -p /dev/shm && \
    chmod 1777 /dev/shm

# Install noVNC
RUN mkdir -p /opt/novnc && \
    git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify.git /opt/novnc/utils/websockify && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html && \
    chown -R browser:browser /opt/novnc

# Copy and set up Chromium starter script
COPY start-chromium.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-chromium.sh && \
    chown browser:browser /usr/local/bin/start-chromium.sh

# Configure Supervisor to manage services
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create required directories with correct permissions
RUN mkdir -p /var/log/supervisor /var/run/supervisor && \
    chown -R browser:browser /var/log/supervisor /var/run/supervisor

# Expose ports for noVNC
EXPOSE 6080

# Start Supervisor to manage services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]