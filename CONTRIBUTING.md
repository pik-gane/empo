# Contributing to EMPO

Thank you for your interest in contributing to EMPO! This document provides guidelines for contributing to the project.

## Most Important

- Before coding something new, make sure the functionality you need is not already there somewhere under `src` or `vendor`. If not, maybe some example under `examples` already has it. If that is the case, turn that functionality into a package functionality  under `src` or `vendor` so that it can be reused. If what you need is really nowhere in the codebase, consider adding it to the package functionality under `src` or `vendor` in a reusable way, rather than just putting it in your script.
- Try adopting a similar coding and documentation style as the one you see in the existing codebase.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/empo.git
cd empo
```

(EMPO project members: you should *not* fork but should work with the main repo directly and be given write permission)

### 2. Set Up Development Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
vim .env

# Start development environment
make up

# Enter container
make shell
```

### 3. Verify Setup

```bash
# Run verification script
bash setup/scripts/verify_setup.sh

# Run tests
python tests/test_structure.py
```

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Then make your changes in some editor

2. Execute code in the container:
   ```bash
   make shell

   # inside container:
   python examples/<category>/whatever.py
   ```

3. Run tests from outside the container:
   ```bash
   make test
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python

- Follow PEP 8 style guide, but not slavishly
- Use type hints where appropriate
- Write docstrings for public functions/classes
- maybe format code with `black`
- maybe lint with `ruff`

### Import Conventions

The codebase uses a **hybrid import strategy** combining absolute and relative imports:

#### Use Absolute Imports for:

- **Cross-package imports** (different top-level `empo` subpackages):
  ```python
  from empo.possible_goal import PossibleGoal
  from empo.learning_based.phase1.neural_policy_prior import BaseNeuralHumanPolicyPrior
  ```

- **All imports in tests and examples**:
  ```python
  from empo.world_model import WorldModel
  from empo.backward_induction.phase1 import compute_human_policy_prior
  ```

- **Imports from parent packages or base classes**:
  ```python
  from empo.robot_policy import RobotPolicy
  ```

- **All vendor package imports** (`gym_multigrid`, `ai_transport`):
  ```python
  from gym_multigrid.multigrid import MultiGridEnv
  from ai_transport.transport_env import TransportEnv
  ```

#### Use Relative Imports for:

- **Sibling modules in the same directory**:
  ```python
  # In __init__.py files
  from .q_network import BaseQNetwork
  from .policy_prior_network import BasePolicyPriorNetwork
  ```

- **Closely related modules within the same subpackage**:
  ```python
  # Within empo/learning_based/multigrid/phase1/
  from .constants import OBJECT_TYPE_TO_CHANNEL
  ```

#### Why This Pattern?

- **Absolute imports** are more explicit and easier for IDE navigation
- **Relative imports** reduce verbosity for intra-package references
- This hybrid approach balances readability with practicality

#### ⚠️ Important: Prefer Absolute Imports When Possible

**When modules are moved or refactored, VS Code/Pylance automatically adjusts absolute imports but fails to adjust relative imports.** This can lead to broken imports after refactoring.

For maximum maintainability:
- Use **absolute imports** whenever practical, especially for frequently-moved modules
- Reserve **relative imports** for stable intra-package references (e.g., `__init__.py` files)
- After moving modules, always search for and manually fix relative imports

#### Note on PYTHONPATH

When running examples outside Docker, first install the main package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

The vendored packages still need to be installed separately or added to `PYTHONPATH`.
One working option is:

```bash
python -m pip install -e ./vendor/multigrid -e ./vendor/ai_transport
```

If you prefer the existing path-based workflow, you can still set `PYTHONPATH`:

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds python examples/<category>/script.py
```

Inside the Docker container (via `make shell`), `PYTHONPATH` is pre-configured.

## Testing

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_feature.py
import pytest
from empo import feature


def test_feature_functionality():
    """Test that feature works correctly."""
    result = feature.do_something()
    assert result == expected_value
```

### Running tests inside container

```bash
# In container
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/empo --cov-report=html
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

### README Updates

If your changes affect setup or usage:
- Update README.md
- Update QUICKSTART.md if needed
- Add examples for new features

## Docker and Deployment

### Testing Docker Changes

```bash
# Test build
docker build -t empo:test .

# Test with compose
docker compose up --build

# Verify GPU support
docker compose exec empo-dev nvidia-smi
```

### Testing Singularity Changes

If you have Singularity/Apptainer installed:

```bash
# Build SIF
apptainer build empo.sif Dockerfile

# Test execution
apptainer exec empo.sif python --version

# Test with GPU
apptainer exec --nv empo.sif nvidia-smi
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] Commits are meaningful and well-formatted
- [ ] Changes work in both Docker and Singularity (if applicable)

### PR Description

Include:
1. **What**: Brief description of changes
2. **Why**: Motivation and context
3. **How**: Implementation approach
4. **Testing**: How you tested the changes
5. **Screenshots**: If UI/output changes

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: Add support for PPO algorithm
fix: Resolve GPU memory leak in training loop
docs: Update cluster deployment instructions
```

## Code Review Process

1. Automated checks run on PR
2. Maintainer reviews code
3. Address feedback
4. Approval and merge

## Example Scripts Guidelines

When creating example scripts (in the `examples/` directory):

### Quick Test Mode

**Always include a command-line parameter for shortened test runs.**

Long-running example scripts should include `--quick`, `--test`, or `--fast` flags that reduce training episodes, environments, and other time-consuming parameters. This allows developers to quickly verify the script works without waiting for the full run.

Example implementation:
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Example Demo")
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced parameters'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Use reduced parameters in quick mode
    num_episodes = 50 if args.quick else 500
    num_envs = 3 if args.quick else 10
    
    # ... rest of script
```

Usage:
```bash
# Full run
python examples/<category>/my_demo.py

# Quick test run
python examples/<category>/my_demo.py --quick
```

This pattern:
- Enables CI/CD testing of example scripts without timeouts
- Allows developers to quickly verify scripts work before long runs
- Documents expected run time differences

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

Feel free to open an issue or discussion if you have any questions about contributing!
