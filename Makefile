.PHONY: help build up down down-dev restart shell logs clean test test-local test-hierarchical lint
.PHONY: build-gpu push-gpu build-sif up-gpu-docker-hub up-gpu-sif-file
.PHONY: build-hierarchical build-gpu-hierarchical test-mineland test-mineland-integration up-hierarchical

# Load .env file if it exists
-include .env
export

# Enable Docker BuildKit for faster builds and cache mounts
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Docker registry configuration (set in .env or environment)
DOCKER_REGISTRY ?= docker.io
DOCKER_USERNAME ?= $(shell whoami)
DOCKER_IMAGE_NAME ?= empo
GPU_IMAGE_TAG ?= gpu-latest
SIF_FILE ?= empo-gpu.sif

# Default target
help:
	@echo "EMPO Development Commands"
	@echo "========================="
	@echo "Local Development:"
	@echo "  make build          - Build Docker image (CPU)"
	@echo "  make build-hierarchical - Build Docker image with hierarchical deps (Ollama client, MineLand)"
	@echo "  make up             - Start development environment (auto-detects GPU)"
	@echo "  make up-hierarchical - Start with Ollama server container (for LLM inference)"
	@echo "  make down           - Stop development environment"
	@echo "  make restart        - Restart development environment"
	@echo "  make shell          - Open shell in container"
	@echo "  make logs           - Show container logs"
	@echo "  make train          - Run training script"
	@echo "  make example        - Run simple example"
	@echo "  make test           - Run tests (in Docker)"
	@echo "  make test-local     - Run tests locally (no Docker needed)"
	@echo "  make test-hierarchical - Run hierarchical world model tests locally"
	@echo "  make test-mineland  - Test MineLand installation (basic import tests)"
	@echo "  make test-mineland-integration - Test MineLand + Ollama vision (full integration)"
	@echo "  make lint           - Run linters"
	@echo "  make clean          - Clean up outputs and cache"
	@echo ""
	@echo "Cluster Deployment (GPU):"
	@echo "  make build-gpu              - Build GPU Docker image only"
	@echo "  make build-gpu-hierarchical - Build GPU image with hierarchical deps"
	@echo "  make up-gpu-docker-hub      - Build GPU image and push to Docker Hub"
	@echo "  make up-gpu-sif-file        - Build GPU image and convert to SIF file locally"
	@echo "  make push-gpu               - Push GPU image to Docker Hub"
	@echo "  make build-sif              - Convert GPU Docker image to SIF file"
	@echo ""
	@echo "Configuration via .env or environment:"
	@echo "  DOCKER_USERNAME      - Docker Hub username (default: $(DOCKER_USERNAME))"
	@echo "  DOCKER_REGISTRY      - Docker registry (default: $(DOCKER_REGISTRY))"
	@echo "  GPU_IMAGE_TAG        - GPU image tag (default: $(GPU_IMAGE_TAG))"
	@echo "  SIF_FILE             - Output SIF filename (default: $(SIF_FILE))"
	@echo "  HIERARCHICAL_MODE    - Enable hierarchical deps in build (default: false)"

# Export USER_ID and GROUP_ID for Docker (used in docker-compose.yml)
export USER_ID ?= $(shell id -u)
export GROUP_ID ?= $(shell id -g)

# Compute safe memory limit: total RAM minus 2 GB reserve (prevents host swap pressure).
# The container's mem_limit + memswap_limit are both set to this value (zero swap for container).
# Override with: CONTAINER_MEM_LIMIT=8g make up
CONTAINER_MEM_LIMIT_AUTO := $(shell awk '/MemTotal/{v=int($$2/1024)-2048; if(v<1024) v=1024; printf "%dm",v}' /proc/meminfo 2>/dev/null || echo "4096m")
export CONTAINER_MEM_LIMIT ?= $(CONTAINER_MEM_LIMIT_AUTO)

# Docker Compose commands
build:
	@echo "Building with USER_ID=$(USER_ID), GROUP_ID=$(GROUP_ID)"
	@docker compose build

up:
	@echo "Starting development environment..."
	@echo "✓ Using USER_ID=$(USER_ID), GROUP_ID=$(GROUP_ID) for file permissions"
	@echo "✓ Container memory limit: $(CONTAINER_MEM_LIMIT) (override with CONTAINER_MEM_LIMIT=Xg)"
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "✓ GPU detected - GPU will be available in container"; \
		docker compose up -d --build; \
	else \
		echo "✓ No GPU detected - running in CPU mode"; \
		docker compose up -d --build; \
	fi
	@echo "Development environment started. Use 'make shell' to enter."
	@echo ""
	@echo "Port mappings (set HOST_*_PORT env vars to customize):"
	@docker compose port empo-dev 8888 2>/dev/null | sed 's/^/  Jupyter:     http:\/\//' || true
	@docker compose port empo-dev 6006 2>/dev/null | sed 's/^/  TensorBoard: http:\/\//' || true
	@docker compose port empo-dev 5678 2>/dev/null | sed 's/^/  Debugger:    /' || true

down:
	@# Stop all containers including hierarchical profile services
	docker compose --profile hierarchical down

down-dev:
	@# Stop only the main development container (preserves Ollama if running)
	docker compose down

restart:
	docker compose restart

# Start development environment with Ollama server for LLM inference
up-hierarchical:
	@echo "Starting development environment with Ollama server..."
	@echo "✓ Using USER_ID=$(USER_ID), GROUP_ID=$(GROUP_ID) for file permissions"
	@echo "✓ Container memory limit: $(CONTAINER_MEM_LIMIT) (override with CONTAINER_MEM_LIMIT=Xg)"
	@echo "✓ Building with HIERARCHICAL_MODE=true (uses Docker cache for faster rebuilds)"
	@HIERARCHICAL_MODE=true docker compose --profile hierarchical build
	@HIERARCHICAL_MODE=true docker compose --profile hierarchical up -d
	@echo "Development environment with Ollama started."
	@echo "Use 'make shell' to enter the dev container."
	@echo "Ollama API available at http://localhost:11434"
	@echo "Pull a vision model with: docker exec ollama ollama pull qwen2.5vl:7b"
	@echo ""
	@echo "Port mappings (set HOST_*_PORT env vars to customize):"
	@docker compose port empo-dev 8888 2>/dev/null | sed 's/^/  Jupyter:     http:\/\//' || true
	@docker compose port empo-dev 6006 2>/dev/null | sed 's/^/  TensorBoard: http:\/\//' || true
	@docker compose port empo-dev 5678 2>/dev/null | sed 's/^/  Debugger:    /' || true

shell:
	docker compose exec empo-dev bash

logs:
	docker compose logs -f

# Training commands
train:
	docker compose exec empo-dev python train.py --num-episodes 100

example:
	docker compose exec empo-dev python examples/multigrid/simple_example.py

# Development commands
test:
	@echo "Running tests..."
	docker compose exec empo-dev python -m pytest tests/ -v \
		--ignore=tests/test_mineland_installation.py \
		--ignore=tests/debug_dag_by_timestep.py \
		--ignore=tests/debug_dag_parallel.py

# Run tests locally without Docker (requires: pip install -r setup/requirements.txt pytest)
test-local:
	@echo "Running tests locally..."
	PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:vendor/l2p:multigrid_worlds python -m pytest tests/ -v \
		--ignore=tests/test_mineland_installation.py \
		--ignore=tests/debug_dag_by_timestep.py \
		--ignore=tests/debug_dag_parallel.py

# Run only hierarchical world model tests (Tasks 1-11) locally
test-hierarchical:
	@echo "Running hierarchical world model tests..."
	PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:vendor/l2p:multigrid_worlds python -m pytest \
		tests/test_world_model_duration.py \
		tests/test_duration_discounting.py \
		tests/test_hierarchical_base.py \
		tests/test_cell_partition.py \
		tests/test_macro_grid_env.py \
		tests/test_multigrid_level_mapper.py \
		tests/test_two_level_multigrid.py \
		tests/test_macro_goals.py \
		tests/test_macro_heuristic_policy.py \
		tests/test_hierarchical_backward_induction.py \
		-v

lint:
	@echo "Running linters..."
	docker compose exec empo-dev ruff check .
	docker compose exec empo-dev black --check .

# Cleanup
clean:
	@echo "Cleaning up outputs and cache..."
	rm -rf outputs/ logs/ __pycache__/ .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# GPU Docker image build
build-gpu:
	@echo "Building GPU-enabled Docker image for cluster..."
	@echo "Image: $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	docker build -f Dockerfile.gpu \
		-t $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG) \
		-t $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG) \
		.
	@echo "✓ GPU image built successfully"

# Push GPU image to Docker Hub
push-gpu:
	@echo "Pushing GPU image to $(DOCKER_REGISTRY)..."
	@echo "Image: $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	@if [ "$(DOCKER_USERNAME)" = "$(shell whoami)" ]; then \
		echo ""; \
		echo "WARNING: DOCKER_USERNAME not set, using system username: $(DOCKER_USERNAME)"; \
		echo "Set DOCKER_USERNAME in .env or environment to use your Docker Hub username"; \
		echo ""; \
	fi
	docker push $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)
	@echo "✓ GPU image pushed successfully"
	@echo ""
	@echo "On cluster, pull with:"
	@echo "  apptainer pull $(SIF_FILE) docker://$(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"

# Build SIF file from GPU Docker image
build-sif:
	@echo "Converting GPU Docker image to Singularity SIF file..."
	@echo "Output: $(SIF_FILE)"
	@if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then \
		echo ""; \
		echo "ERROR: Neither apptainer nor singularity found!"; \
		echo "This target requires Apptainer/Singularity to be installed."; \
		echo ""; \
		echo "Alternatives:"; \
		echo "  1. Use 'make up-gpu-docker-hub' to push to Docker Hub instead"; \
		echo "  2. Install Apptainer: https://apptainer.org/docs/admin/main/installation.html"; \
		echo "  3. Use Docker Desktop with 'docker save' and convert on cluster"; \
		echo ""; \
		exit 1; \
	fi
	@if command -v apptainer &> /dev/null; then \
		apptainer build $(SIF_FILE) docker-daemon://$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG); \
	else \
		singularity build $(SIF_FILE) docker-daemon://$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG); \
	fi
	@echo "✓ SIF file created: $(SIF_FILE)"
	@echo ""
	@echo "Copy to cluster with:"
	@echo "  scp $(SIF_FILE) user@cluster:~/bega/empo/"
	@echo ""
	@echo "On cluster, run with:"
	@echo "  cd ~/bega/empo/git"
	@echo "  sbatch ../setup/scripts/run_cluster_sif.sh"

# Build and push GPU image to Docker Hub (no Singularity needed locally)
up-gpu-docker-hub: build-gpu push-gpu
	@echo ""
	@echo "==================================="
	@echo "GPU image ready on Docker Hub!"
	@echo "==================================="
	@echo ""
	@echo "On cluster, pull and run with:"
	@echo "  cd ~/bega/empo"
	@echo "  mkdir -p git"
	@echo "  cd git && git clone <your-repo-url> . && cd .."
	@echo "  apptainer pull empo.sif docker://$(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	@echo "  cd git && sbatch ../setup/scripts/run_cluster_sif.sh"

# Build GPU image and convert to SIF file locally
up-gpu-sif-file: build-gpu build-sif
	@echo ""
	@echo "==================================="
	@echo "GPU SIF file ready for cluster!"
	@echo "==================================="
	@echo ""
	@echo "Copy to cluster and run:"
	@echo "  scp $(SIF_FILE) user@cluster:~/bega/empo/"
	@echo "  ssh user@cluster"
	@echo "  cd ~/bega/empo/git"
	@echo "  sbatch ../setup/scripts/run_cluster_sif.sh"

# Build Docker image with hierarchical dependencies (Ollama, MineLand)
# These are large packages that require Java JDK 17 and Node.js 18
build-hierarchical:
	@echo "Building Docker image with hierarchical dependencies..."
	docker build --build-arg DEV_MODE=true --build-arg HIERARCHICAL_MODE=true \
		-t $(DOCKER_IMAGE_NAME):hierarchical .
	@echo "✓ Hierarchical image built successfully"

# Build GPU Docker image with hierarchical dependencies
build-gpu-hierarchical:
	@echo "Building GPU Docker image with hierarchical dependencies..."
	@echo "Image: $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical"
	docker build -f Dockerfile.gpu \
		--build-arg HIERARCHICAL_MODE=true \
		-t $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical \
		-t $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical \
		.
	@echo "✓ GPU hierarchical image built successfully"

# Test MineLand installation (requires hierarchical build)
test-mineland:
	@echo "Testing MineLand installation (basic import tests)..."
	docker compose exec empo-dev python tests/test_mineland_installation.py

# Validate MineLand setup
# Note: MineLand spawns Minecraft internally in empo-dev, so there's no separate server
test-mineland-validate:
	@echo "Validating MineLand setup..."
	@echo ""
	@echo "Checking if containers are running..."
	@docker ps --filter "name=empo-dev" --format "{{.Status}}" | grep -q "Up" && echo "✓ empo-dev container is running" || (echo "✗ empo-dev container is not running. Start with: make up-hierarchical" && exit 1)
	@docker ps --filter "name=ollama" --format "{{.Status}}" | grep -q "Up" && echo "✓ ollama container is running" || echo "⚠ ollama container is not running"
	@echo ""
	@echo "Architecture (simplified):"
	@echo "  empo-dev  -> Your RL code + MineLand (spawns Minecraft internally via headless mode)"
	@echo "  ollama    -> LLM server (accessible at ollama:11434)"
	@echo ""
	@echo "No separate Minecraft server needed - MineLand handles everything."

# Test MineLand + Ollama integration (runs in empo-dev)
# Note: MineLand spawns Minecraft internally - may take 1-2 minutes on first run
test-mineland-integration:
	@echo "Testing MineLand + Ollama integration..."
	@echo ""
	@echo "Architecture:"
	@echo "  empo-dev  -> your RL/planning code + MineLand (spawns Minecraft internally)"
	@echo "  ollama    -> LLM server (ollama:11434)"
	@echo ""
	@echo "Make sure you have:"
	@echo "  1. Started with: make up-hierarchical"
	@echo "  2. Pulled model: docker exec ollama ollama pull qwen2.5vl:7b"
	@echo ""
	@echo "Note: First run may take 1-2 minutes to download Minecraft"
	@echo ""
	docker compose exec empo-dev python tests/test_mineland_installation.py --integration
