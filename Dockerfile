# Lightweight Dockerfile for MARL development
# Uses Python 3.11 slim base for MineLand compatibility
# PyTorch automatically uses GPU if available via Docker GPU passthrough

FROM python:3.11-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/workspace:/workspace/src:/workspace/vendor/multigrid:/workspace/vendor/ai_transport:/workspace/vendor/l2p

# Set working directory
WORKDIR /workspace

# Install system dependencies (no CUDA libraries)
# Use BuildKit cache mount for apt to avoid re-downloading packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    graphviz \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY setup/requirements.txt /tmp/requirements.txt
COPY setup/requirements-dev.txt /tmp/requirements-dev.txt
COPY setup/requirements-hierarchical.txt /tmp/requirements-hierarchical.txt

# Install Python dependencies
# Use CPU-only PyTorch to avoid downloading large CUDA packages (~2GB vs ~5GB)
# GPU will still work if available via Docker GPU passthrough (--gpus flag)
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r /tmp/requirements.txt

# Install dev dependencies only if DEV_MODE is set (for Docker Compose)
# This allows the same Dockerfile to be used for both dev and production
ARG DEV_MODE=false
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$DEV_MODE" = "true" ] ; then \
    pip install -r /tmp/requirements-dev.txt ; \
    fi

# Install hierarchical dependencies only if HIERARCHICAL_MODE is set
# MineLand requires Java JDK 17, Node.js 18.x, and xvfb for headless rendering
# MineLand is a multi-agent Minecraft RL platform from https://github.com/cocacola-lab/MineLand
ARG HIERARCHICAL_MODE=false
RUN echo "HIERARCHICAL_MODE is set to: $HIERARCHICAL_MODE"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing Java, xvfb, and xauth for MineLand..." && \
    apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        xvfb \
        xauth ; \
    fi
# Install Node.js 18.x for MineLand (using NodeSource repository)
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing Node.js 18.x for MineLand..." && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs ; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing hierarchical Python dependencies..." && \
    pip install -r /tmp/requirements-hierarchical.txt ; \
    fi
# Clone MineLand from GitHub first (separate layer for better caching)
# Also fix their broken setup.py (where='mineland' should be where='.')
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Cloning MineLand from GitHub..." && \
    git clone --depth 1 https://github.com/cocacola-lab/MineLand.git /opt/MineLand && \
    echo "Fixing MineLand setup.py bug (wrong find_packages where parameter)..." && \
    sed -i "s/where='mineland'/where='.'/" /opt/MineLand/setup.py && \
    cat /opt/MineLand/setup.py ; \
    fi

# Install MineLand's requirements.txt first (heavy dependencies like chromadb, langchain)
# This allows Docker to cache these large dependencies separately
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing MineLand dependencies (this may take a while)..." && \
    pip install -r /opt/MineLand/requirements.txt && \
    echo "✓ MineLand dependencies installed" ; \
    fi

# Install MineLand package itself (editable install, much faster after deps are installed)
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing MineLand package..." && \
    cd /opt/MineLand && \
    pip install -e . --no-deps && \
    echo "✓ MineLand package installed" ; \
    fi

# Install MineLand's Node.js dependencies (mineflayer)
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    echo "Installing MineLand mineflayer dependencies..." && \
    cd /opt/MineLand/mineland/sim/mineflayer && \
    npm ci && \
    echo "✓ MineLand mineflayer installed" ; \
    fi

# Verify MineLand installation
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    python -c "import mineland; print('✓ MineLand import verification passed')" ; \
    else \
    echo "HIERARCHICAL_MODE is not true ($HIERARCHICAL_MODE), skipping MineLand installation" ; \
    fi

# Create a non-root user for better security
# Default to 1001 to match docker-compose.yml defaults
ARG USER_ID=1001
ARG GROUP_ID=1001
RUN if getent group ${GROUP_ID} > /dev/null 2>&1; then \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash appuser; \
    else \
        groupadd -g ${GROUP_ID} appuser && \
        useradd -m -u ${USER_ID} -g appuser -s /bin/bash appuser; \
    fi

# Create workspace directory and set permissions
# Also fix MineLand permissions (installed as root but needs to be writable by appuser)
RUN mkdir -p /workspace && chown -R ${USER_ID}:${GROUP_ID} /workspace
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    chown -R ${USER_ID}:${GROUP_ID} /opt/MineLand ; \
    fi

# Switch to non-root user
USER appuser

# Create common output directories
RUN mkdir -p /workspace/outputs /workspace/logs

# Copy project files (required for CI testing)
# In development, this is typically overridden by volume mounts via docker-compose
# Security: relies on .dockerignore to exclude sensitive files (.env, .git, etc.)
COPY --chown=appuser:appuser . /workspace/

# Default command (can be overridden)
CMD ["/bin/bash"]
