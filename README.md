# EMPO – Human Empowerment AI Agents

A framework for studying the soft maximization of aggregate human power by AI agents in multigrid and other multi-agent model worlds related to [this theoretical paper](https://arxiv.org/html/2508.00159v2).

*See the Devin-generated interactive Wiki here: https://deepwiki.com/mensch72/empo*

## What this is and what this is *not*

The fact that some of the code uses reinforcement learning (RL) algorithms can lead to certain confusions about the purpose of those algorithms in the context of this project. To avoid confusion, we want to clarify what this project is and is *not*: 

### Solution approximation, not policy optimization
- We are *not* trying to learn policies that are in any sense "optimal", "best", or that maximize some form of expected "return" or "utility". Instead, we are simply trying to solve certain equations from the paper that define certain policies. We are not trying to "improve" these policies, we only try to *approximately calculate* them as solutions to these equations.
- The equations we want to solve define two kinds of policies: (i) a **human policy prior** and (ii) a **robot policy**. The robot policy aims to softly (!) maximize a particular notion of aggregate human power. The human policy prior is used by the robot as a kind of conservative assumption about what humans will do in case they had certain goals, which the robot needs to calculate its assessment of human power. The paper cited above explains all this in very much detail.  
- Crucially, the policies are *not* defined by appealing to any form of "optimality" or "utility". Still, the equations that define these policies and that we want to solve are *formally* very similar to the Bellman equations from standard dynamic programming and reinforcement learning. This is the reason why we adapt existing methods from deep reinforcement learning (DRL) to approximate the solutions to these equations.
- More precisely, the equations we want to solve define the probability distribution of actions taken at some state *s* (which we can call the *local policy*) as a function of a particular form of *state-action values* (aka *Q values*). These Q values are a function of (i) some notion of *hypothetical or intrinsic short-term reward*, (ii) the robot's expectation for the Q value of the state and action of the subsequent decision time point. That expectation depends on the probability distribution of successor states and actions, which depends on (i) the robot's beliefs about what actions have what consequences with what probabilities (these constitute the robot's *world model*), and (ii) the probability distribution of future actions. In other words, the policy depends on itself – the equations are recursive – they look very similar to standard Bellman equations.

### Real world, world model, "environment"
- The policy "learning" does *not* happen in the *real world.* It is *not* based on a "trial and error" process in which the humans or robot test actions in the real world, observe their consequences, and update their Q estimates and policy in turn. *No action* is ever taken before the policy has been fully approximated by the algorithm. Instead, the policy approximation by means of adapted DRL methods is performed *in the brain of the robot* on the basis of its **world model** that represents everything that we make it believe about the real world and the behavior of the humans in it. In the field of RL, this would thus be classified as **model-based planning**.
- For the purpose of this project, we treat the world model as a *given* rather than as something that needs to be learned. Currently, the repo uses three different world models. The main part of the project will use the *multigrid* world model where several humans and robots can move in a 2D grid that contains several types of objects. We will also use a *transport* world model where several autonomous vehicles (robots) can transport several humans on a road network. We might also use an LLM to produce situational world models for the *minecraft* game. (Later, in a real-world application of the approach explored by this project, our algorithms would be combined with world models that themselves might be formed or updated by some form of supervised learning, but that is a different form of learning and a different challenge that we leave to other teams)
- A world model is *more* than what RL people typically call an "environment". While an environment can only *simulate* individual realizations of the world state trajectory starting from a fixed initial state (called "rollouts"), a world model can also *predict* the probability distribution of successor states of any possible state under any possible action. In other words, while an environment (basically) only provides `reset()` and `step(action)`, a world model also provides an oracle method `transition_probabilities(state, action)` returning a dict of the form `{ successor: probability }`. The `transition_probabilities()` method will eventually be used to improve the convergence of the RL-based solution algorithms because it allows to compute part of the expected value over successor states explicitly as a weighted sum rather than having to estimate it from sampled successor states as in standard temporal difference learning (but this is not implemented yet). At the moment, the `transition_probabilities()` method is only used for very small world models that have so few states that the solution to the policy equations can be calculated exactly by means of a "backward induction" process instead of approximating the solution via RL.
 
### Inverse temperature, discount factors
- The purpose of the *inverse temperature* quantities `beta_h` and `beta_r` we use in our DRL algorithms is *not* to improve convergence of the training via exploration. The quantities `beta_h` and `beta_r` are not hyperparameters of the training algorithm at all.
- Instead, `beta_h` is part of the robot's belief system about human behavior. It is in this sense *descriptive* rather than technical. It represents the robot's belief about the level of human bounded rationality. As such, `beta_h` might later depend on the human, the goal of the human, and the state. There *is* however *also* a hyperparameter that steers exploration to improve convergence: the `epsilon` parameter for "epsilon-greedy" exploration during training. It is important to understand that while `beta_h` is part of the policy definition, `epsilon` is *not*. `epsilon` only influences the speed of convergence to the solution and the approximation quality of the solution, but does *not* influence what the sought solution is. In particular, we might choose to phase out exploration for later training episodes by using a schedule for `epsilon` that makes it converge to zero eventually, while keeping `beta_h` at its constant value that represents what type of humans we currently want to simulate.
- Also `beta_r` is not a hyperparameter of the training algorithm. It is also not a variable that represents some element of the robot's belief system. Instead, it is a *normative parameter of the theory*, so it is *normative* rather than descriptive or technical. Humanity can use `beta_r` to steer the robot's level of *soft* maximization. `beta_r` determines how much the robot will prefer actions that lead to higher aggregate human power over actions leading to lower aggregate human power.
- Similarly, the purpose of the *discount factors* `gamma_h` and `gamma_r` is *not* to improve training. Like `beta_h`, `gamma_h` is descriptive rather than technical. It represents the robot's beliefs about the time-preferences of the humans. Like `beta_r`, `gamma_r` is normative rather than descriptive or technical. Humanity can use `gamma_r` to steer the robot's valuation of future aggregate human power as compared to current aggregate human power. We *might* however *also* use additional hyperparameters `warmup_gamma_h` and `warmup_gamma_r` that determine the initial discount factors used in the initial episodes of training if that improves convergence, but if we do so, then we must use a training schedule that ultimately lets the discount factors converge to the "correct" values given by `gamma_h` and `gamma_r`.

### Beliefs about others' behavior
- Although we have many agents, both humans and robots, we do *not* aim to learn some form of "equilibrium" (such as a Nash equilibrium or a quantal response equilibrium) between the policies of these many agents. Instead, when approximately calculating the human policy prior for a particular human and a particular possible goal of that human, *the robot assumes that each human has certain unchanging beliefs about the behavior of the other humans and about the behavior of the robots,* and these assumed human beliefs about each other's behavior are *not* updated during training (as it would be done in a typical MARL problem). Rather, the beliefs of a human about other humans' behavior are encoded by the object `believed_others_policy` that corresponds to the quantity $\mu_{-h}$ in equation (1) of the cited paper. For simplicity, for most of the project we will make all humans have the same `believed_others_policy`, i.e., all humans share the same beliefs about what everyone might do. We make this assumption because it allows us to train the human policy prior of all humans simultaneously using the same training data, generated by the common `believed_others_policy`. (Later, one should consider `believed_others_policy` a part of the world model, and will expect that the world model describes how these beliefs will change over time in reaction to humans' real world belief updating.)
- A major advantage of this approach over typical MARL problems is that this way the human behavior priors do not depend on each other and do not depend on the actual robot policy! This allows us to calculate the human behavior priors "in parallel" and before calculating the robot policy. We call the *human policy prior calculation "phase 1"* and the *robot policy calculation "phase 2"* of the approach.

### Rewards: goal-specific hypothetical, shaping, or intrinsic
- A major difference to standard RL is that the "rewards" that occur in the humans' and robot's Bellman equation are *never* given by the environment (the real world or the world model). They do not represent "game scores" or "utility" or "money" or "fitness" or anything like that.
- The reward denoted as $U_h$ used in the **human** policy (equation (1) in the paper) represents whether or not a *possible goal* that the robot considers the human *might* have is *achieved or not achieved* in the current state. As soon as the possible goal is achieved, the human receives a reward $U_h=1$ and does not receive any further rewards in the same simulated episode. Before that the human receives a reward $U_h=0$. As a consequence, a human's total reward for a simulated episode is either 0 or 1, so the expected discounted total reward (represented by `Vh(initial_state)`) is between 0 and 1. This reward is not "given by the environment" because it completely depends on what the hypothetical goal of the human is. So it is a *goal-specific hypothetical* reward assigned by the robot on the basis of its world model.
- The robot is *not* assuming or forming any belief about what the humans' *actual* goals are. It always considers *all kinds of possible goals*. When the robot aggregates quantities over all possible goals (like in equations (4), (6), (7) of the paper), it might sometimes use different weights for different possible goals, but these weights do *not* represent the likelihood of having that goal. These weights are rather a theoretical instrument to make the resulting policies have certain desirable formal properties. In particular, these weights will always be highly symmetric, e.g., all goals of the form "reach this 5-by-5-sized region in the grid" will get the same weight, but goals of the form "reach this particular grid cell" might get a smaller weight.
- Since the human reward structure is sparse (rewards only accrue when the possible goal is achieved, not on the way towards the possible goal), it is hard to learn. For this reason, to improve the training convergence, the training data will typically be modified to include an additional *shaping reward*. Like `epsilon`, the shaping reward does *not* influence the solution to the policy equations, but only influences the convergence and approximation quality. We use standard, potential-based shaping rewards mostly based on distance-to-goal.
- The reward denoted as $U_r$ used in the **robot** policy (equation (8) in the paper) does *not* represent goal achievement like for humans, and is also not given by the environment. It represents the robot's assessment of total human power *in the current state, given the current policies*. So the robot sets its own reward, and hence this is an example of an *intrinsic* reward.

### Technical challenge: mutual dependency between robot policy and power metric
- The **main technical challenge** of the project is that *this intrinsic reward depends on the policies* rather than only on the state. This is because the power metric we use estimates human's *ability to reach goals in the future* rather than whether humans have already achieved goals. This ability depends on what everyone is doing in the future, so it depends on the policies, both human and robot. The human policy prior can be calculated in phase 1 *before* the power metric. But the robot policy must be calculated *simultaneously* with the power metric as each influences the other in a kind of feedback loop (in the paper, we can see this loop in equations (4)-(9): (4) depends on (9), (9) on (4) and (8), (8) on (7), (7) on (6), (6) on (5), and (5) on (4), closing the loop). This mutual dependency is *different* from the mutual dependencies that occur in other MARL problems, where it is typically the policies of two different agents that depend on each other, while here it is a policy and a power metric that depend on each other. Still, we expect to be able to learn something from looking at how MARL approaches mutual dependencies, e.g., via time-scale separation or working with frozen copies of quantities, etc.

## Core Framework

The EMPO framework so far provides:

### World Model Abstraction (`src/empo/`)
- **WorldModel**: Abstract base class for environments with explicit state management
  - `get_state()` / `set_state()`: Hashable state representation and restoration
  - `transition_probabilities()`: Exact probabilistic transition computation
  - `get_dag()`: Compute the state-space DAG for finite environments

### Human Behavior Modeling
- **PossibleGoal**: Abstract class for goal specification (0/1 reward functions)
- **PossibleGoalGenerator**: Enumerate possible human goals with weights
- **HumanPolicyPrior**: Model human behavior as goal-directed policies

### Policy Computation
- **compute_human_policy_prior**: Backward induction to compute Boltzmann policies
  - Supports parallel computation for large state spaces
  - Configurable temperature (β) for policy stochasticity

### Vendored Test Environments

#### MultiGrid (`vendor/multigrid/`)
Extended multi-agent gridworld environment with:
- State management and transition probability computation
- New object types: Rock, Block, Bush, UnsteadyGround, MagicWall
- Map-based environment specification
- Agent-specific capabilities (can_push_rocks, can_enter_magic_walls)

#### Transport (`vendor/ai_transport/`)
An environment in which a fleet of autonomous passenger vehicles (robots) can transport humans between nodes in a road network.

See [docs/API.md](docs/API.md) for API reference.

## Features

- Unified Docker image for development and cluster deployment
- **Pre-built container images** on GitHub Container Registry for instant setup
- Exact planning algorithms via backward induction on state DAGs
- Multi-Agent Reinforcement Learning support (work in progress)
- Easy local development with Docker Compose
- Cluster-ready with Singularity/Apptainer support
- GPU acceleration support (NVIDIA CUDA)
- Integration with TensorBoard and Weights & Biases

## Pre-built Container Images

For instant development setup without rebuilding, use our pre-built Docker images:

```bash
# Pull the latest image from GitHub Container Registry
docker pull ghcr.io/mensch72/empo:main

# Run with your local code mounted
docker run -it --rm -v $(pwd):/workspace ghcr.io/mensch72/empo:main bash
```

The repository also includes a `.devcontainer` configuration for:
- **GitHub Codespaces**: Click "Code" → "Codespaces" → "Create codespace on main"
- **VS Code Dev Containers**: Open repo and select "Reopen in Container"
- **AI Coding Assistants**: Automatically detected for faster session startup

See [docs/PREBUILT_IMAGES.md](docs/PREBUILT_IMAGES.md) for more details.

## Quick Start

### Prerequisites

**For Local Development:**
- Docker Engine 20.10+ with Docker Compose v2
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers (for GPU support)

**For Cluster Deployment:**
- Singularity/Apptainer 1.0+
- SLURM or similar job scheduler (optional)

### Installation

Clone the repository:

```bash
git clone https://github.com/mensch72/empo.git
cd empo
```

For local Python tooling outside Docker, install the root package in editable mode:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ./vendor/multigrid -e ./vendor/ai_transport
```

This replaces the need to add `src/` to `PYTHONPATH` for the main `empo` package. The
vendored packages can still be installed as editable siblings, or kept on `PYTHONPATH`
if you prefer that workflow.

## Google Colab (Recommended for Quick Start)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/empo_colab_demo.ipynb)

The fastest way to try EMPO is via Google Colab. Click the badge above or follow these steps:

```python
# 1. Clone the repository
!git clone --depth 1 https://github.com/mensch72/empo.git
%cd empo

# 2. Install system dependencies (for DAG visualization)
!apt-get update -qq && apt-get install -qq graphviz > /dev/null 2>&1

# 3. Install Python dependencies
!pip install -q -r setup/requirements-colab.txt

# 4. Set up Python paths
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'vendor', 'multigrid'))

# 5. Verify installation
from empo import WorldModel, PossibleGoal
from envs.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
print("✓ EMPO is ready!")
```

See [notebooks/empo_colab_demo.ipynb](notebooks/empo_colab_demo.ipynb) for a complete interactive tutorial.

**Colab Limitations:**
- MPI distributed training is not supported (use `parallel=False`)
- Docker is not available in Colab
- Sessions timeout after ~12 hours

## Local Development

### 1. Build and Start the Development Environment

```bash
# Single command that works everywhere
make up

# Or using docker compose directly
docker compose up -d
```

The setup automatically:
- Uses a lightweight Ubuntu-based image (~2GB)
- Detects if you have an NVIDIA GPU
- Shows you whether GPU is available or running in CPU mode
- No CUDA libraries downloaded unless needed for cluster deployment
- **Caches apt and pip packages** for much faster rebuilds
  - First build: ~5-10 minutes (downloads packages)
  - Subsequent builds: ~30 seconds (uses cached packages)
  - Only rebuilds changed layers (e.g., when setup/requirements.txt changes)

### 2. Enter the Container

```bash
# Attach to the running container
docker compose exec empo-dev bash

# Or use docker exec
docker exec -it empo-dev bash
```

### 3. Run Training

Inside the container:

```bash
# Run the example training script
python train.py --num-episodes 100

# Or with custom arguments
python train.py \
  --env-name CartPole-v1 \
  --num-episodes 1000 \
  --lr 0.001 \
  --output-dir ./outputs
```

### 4. Development Workflow

The repository is bind-mounted at `/workspace`, so any changes you make locally are immediately reflected in the container:

```bash
# Edit files on your host machine with your favorite editor
vim train.py

# Changes are immediately available in the container
docker compose exec empo-dev python train.py
```

### 5. GPU Support

GPU support is automatically detected when you run `make up`:

```bash
# Just use the standard command
make up
```

The system will:
- Detect if you have an NVIDIA GPU with `nvidia-smi`
- Display "GPU detected" or "No GPU detected"  
- Work correctly either way - no configuration needed

**Verifying GPU access (if GPU is available):**

```bash
# This will work if GPU was detected
docker compose exec empo-dev nvidia-smi
docker compose exec empo-dev python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** The Docker image uses a lightweight Ubuntu base (~2GB), not CUDA base, so it's fast to download on any system. PyTorch automatically uses GPU if available, or CPU otherwise.

### 6. Jupyter Notebook (Optional)

```bash
# Start Jupyter inside the container
docker compose exec empo-dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access at http://localhost:8888
```

### 7. TensorBoard (Optional)

The training script automatically logs metrics to TensorBoard. To view them:

```bash
# Start TensorBoard (from within the container)
docker compose exec empo-dev tensorboard --logdir=./outputs --host=0.0.0.0

# Or from your host machine (if you have tensorboard installed)
tensorboard --logdir=./outputs

# Access at http://localhost:6006
```

The training script writes metrics like episode rewards, episode lengths, and learning rates to TensorBoard. Even in demo mode (without a real environment), it logs sample data so you can verify TensorBoard is working correctly.

### 8. Stop the Environment

```bash
# Stop the container
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Cluster Deployment

EMPO provides streamlined GPU-enabled cluster deployment with two methods. See [CLUSTER.md](CLUSTER.md) for the complete guide.

### Quick Deploy to Cluster

#### Method 1: Via Docker Hub (Recommended)

Build locally and push to Docker Hub, then pull on cluster:

```bash
# On local machine:
# 1. Configure Docker Hub credentials in .env
cp .env.example .env
# Edit .env and set DOCKER_USERNAME

# 2. Build and push GPU image
make up-gpu-docker-hub

# On cluster:
# 3. Pull and run
cd ~/bega/empo
mkdir -p git && cd git && git clone <your-repo-url> . && cd ..
apptainer pull empo.sif docker://yourusername/empo:gpu-latest
cd git && sbatch ../setup/scripts/run_cluster_sif.sh
```

#### Method 2: Direct SIF Transfer

Build SIF file locally and copy directly to cluster (requires Apptainer/Singularity locally):

```bash
# On local machine:
# 1. Build SIF file
make up-gpu-sif-file

# 2. Copy to cluster
scp empo-gpu.sif user@cluster:~/bega/empo/

# On cluster:
# 3. Run training
cd ~/bega/empo/git
sbatch ../setup/scripts/run_cluster_sif.sh
```

### Key Features

- ✅ **GPU Support**: Full CUDA 12.1 support for cluster GPUs
- ✅ **No Rebuild**: Same workflows as local development
- ✅ **Fixed Working Directory**: No more "chdir" warnings
- ✅ **SLURM Ready**: Pre-configured job scripts included

### Cluster Deployment (Legacy Instructions)

<details>
<summary>Click to expand older deployment methods</summary>

### 1. Build the Docker Image

First, build the production Docker image (without dev dependencies):

```bash
# Build production image
docker build -t empo:latest .

# Or build with a specific tag
docker build -t empo:v0.1.0 .
```

### 2. Convert to Singularity/Apptainer Image

There are several ways to get the image on your cluster:

#### Option A: Pull from a Registry (Recommended)

```bash
# Push to Docker Hub or GitHub Container Registry
docker tag empo:latest yourusername/empo:latest
docker push yourusername/empo:latest

# On the cluster, pull and convert to SIF format
apptainer pull empo.sif docker://yourusername/empo:latest
```

#### Option B: Build Directly from Dockerfile

```bash
# On the cluster with Apptainer installed
apptainer build empo.sif Dockerfile
```

#### Option C: Transfer from Local Docker

```bash
# Save Docker image to a tar file
docker save empo:latest -o empo.tar

# Transfer to cluster (e.g., via scp)
scp empo.tar cluster:/path/to/destination/

# On the cluster, load and convert
apptainer build empo.sif docker-archive://empo.tar
```

### 3. Test the Singularity Image

```bash
# Test basic functionality
apptainer exec empo.sif python3 --version

# Test with GPU support
apptainer exec --nv empo.sif python3 -c "import torch; print(torch.cuda.is_available())"

# Run the training script
apptainer exec --nv -B $(pwd):/workspace empo.sif python /workspace/train.py --num-episodes 10
```

### 4. Submit a SLURM Job

Edit the provided SLURM script and submit:

```bash
# Create logs directory
mkdir -p logs

# Edit the script with your parameters
vim setup/scripts/run_cluster.sh

# Submit the job
sbatch setup/scripts/run_cluster.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/empo_<job_id>.out
```

### 5. Interactive Cluster Session

For interactive development on the cluster:

```bash
# Request an interactive GPU node
srun --partition=gpu --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash

# Run commands interactively
apptainer shell --nv -B $(pwd):/workspace empo.sif
python /workspace/train.py --num-episodes 100
```

## Project Structure

```
empo/
├── Dockerfile                 # Unified Docker image definition
├── docker-compose.yml         # Local development setup
├── setup/                     # Setup and packaging helpers
│   ├── requirements.txt           # Python dependencies
│   └── requirements-dev.txt       # Development dependencies
├── train.py                   # Main training script
├── src/
│   ├── empo/                  # Core EMPO package
│   │   ├── __init__.py        # Package exports
│   │   ├── world_model.py     # WorldModel abstract base class
│   │   ├── possible_goal.py   # Goal abstractions
│   │   ├── human_policy_prior.py  # Human behavior modeling
│   │   ├── backward_induction.py  # Policy computation
│   │   └── hierarchical/      # Hierarchical planning (WIP)
│   ├── envs/                  # Custom environments
│   │   └── one_or_three_chambers.py  # Multi-chamber gridworld
│   └── llm_hierarchical_modeler/  # LLM-based Minecraft world generation
├── vendor/
│   └── multigrid/             # Vendored Multigrid (extensively modified)
│       ├── gym_multigrid/
│       │   └── multigrid.py   # Core MultiGridEnv + state management
│       └── PROBABILISTIC_TRANSITIONS.md
├── docs/
│   ├── API.md                 # API reference
│   ├── ISSUES.md              # Known issues and improvements
│   └── plans/README.md        # Status index for design plans
├── tests/                     # Test suite
├── setup/scripts/
│   ├── run_cluster.sh         # SLURM job script
│   └── setup_cluster_image.sh # Cluster image setup helper
├── examples/                  # Example scripts and notebooks
├── VENDOR.md                  # Documentation for vendored dependencies
└── README.md                  # This file
```

## Vendored Dependencies

This repository includes the [Multigrid](https://github.com/ArnaudFickinger/gym-multigrid) source code in `vendor/multigrid/` to enable live editing without container rebuilds.

**How it works:**
- Multigrid is imported via `PYTHONPATH` (not pip installed)
- Edit files in `vendor/multigrid/gym_multigrid/` and changes take effect immediately
- No Docker rebuild needed for modifications
- Perfect for making extensive changes to environments

**Modifying Multigrid:**
```bash
# 1. Edit source files
vim vendor/multigrid/gym_multigrid/envs/collect_game.py

# 2. Restart Python or re-import (no rebuild needed)
docker compose restart empo-dev
```

**Updating from upstream:**
```bash
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git master --squash
```

See [VENDOR.md](VENDOR.md) for detailed documentation on managing vendored dependencies.


## Environment Variables

### Docker Compose

Set these in a `.env` file or export before running:

```bash
# User ID mapping (for file permissions)
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs

# Weights & Biases
export WANDB_API_KEY=your_key_here
```

### Cluster

Set these in your job script or environment:

```bash
export REPO_PATH=/path/to/empo
export IMAGE_PATH=/path/to/empo.sif
export SCRIPT_PATH=train.py
```

## Troubleshooting

### Docker Compose Issues

**GPU not detected:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker Compose GPU syntax
docker compose config
```

**Permission issues:**

The `make up` command automatically sets USER_ID and GROUP_ID to match your host user. If you encounter permission issues:

```bash
# Make sure you're using make up (recommended)
make up

# Or manually set user IDs with docker compose
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
docker compose up --build
```

If you still have issues, ensure you have write permissions to the repository directory on your host system.

### Singularity/Apptainer Issues

**GPU not available:**
```bash
# Ensure --nv flag is used
apptainer exec --nv empo.sif nvidia-smi

# Check CUDA libraries
apptainer exec --nv empo.sif python -c "import torch; print(torch.version.cuda)"
```

**Mount point issues:**
```bash
# Ensure bind mount paths exist and are accessible
apptainer exec -B /full/path/to/repo:/workspace empo.sif ls /workspace
```

**Image building issues:**
```bash
# Use --fakeroot if you don't have root privileges
apptainer build --fakeroot empo.sif Dockerfile
```

## Advanced Usage

### Custom Dependencies

Edit `pyproject.toml` to add your dependencies:

```bash
# Add to [project.dependencies] or an optional dependency group
your-package>=1.0.0

# Rebuild the image
docker compose up --build
```

### Multi-GPU Training

```bash
# Docker Compose (use specific GPUs)
CUDA_VISIBLE_DEVICES=0,1 docker compose up

# Cluster (request multiple GPUs)
#SBATCH --gres=gpu:2
```

### Distributed Training with MPI

```bash
# In the container or on the cluster
mpirun -np 4 python train.py --distributed
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Docker and Singularity
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Documentation

- **[README.md](README.md)** - This file, comprehensive setup and usage guide
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed implementation notes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[VENDOR.md](VENDOR.md)** - Managing vendored dependencies (Multigrid)
- **[docs/PREBUILT_IMAGES.md](docs/PREBUILT_IMAGES.md)** - Using pre-built container images
- **[docs/BUSHWORLD.md](docs/BUSHWORLD.md)** - The BushWorld world model
- **[docs/IMPLEMENTING_A_NEW_WORLDMODEL.md](docs/IMPLEMENTING_A_NEW_WORLDMODEL.md)** - Step-by-step guide to adding a new WorldModel
- **[.env.example](.env.example)** - Environment variables template

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on PyTorch and Gymnasium
- Supports PettingZoo and Multigrid environments
- Inspired by empowerment-driven intrinsic motivation research

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- Refer to the troubleshooting section above

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{empo2024,
  title = {EMPO: Empowerment-based Multi-Agent Reinforcement Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/pik-gane/empo}
}
```
