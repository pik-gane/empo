"""
Basic tests for the EMPO framework.

Run with: pytest tests/
"""

import pytest
import tomllib
from pathlib import Path


def test_import_empo():
    """Test that the empo package can be imported."""
    try:
        import empo

        assert empo.__version__ == "0.1.0"
    except ImportError:
        pytest.skip("Missing optional runtime dependencies for empo import")


def test_requirements_exist():
    """Test that requirement files exist."""
    req_file = Path(__file__).parent.parent / "setup" / "requirements.txt"
    req_dev_file = Path(__file__).parent.parent / "setup" / "requirements-dev.txt"
    assert req_file.exists()
    assert req_dev_file.exists()


def test_pyproject_exists_and_has_project_metadata():
    """Test that the root pyproject.toml exists and exposes package metadata."""
    pyproject_file = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_file.exists()

    pyproject = tomllib.loads(pyproject_file.read_text())
    assert pyproject["project"]["name"] == "empo"
    assert "dev" in pyproject["project"]["optional-dependencies"]


def test_dockerfile_exists():
    """Test that Dockerfile exists."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    assert dockerfile.exists()


def test_docker_compose_exists():
    """Test that docker-compose.yml exists."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_file.exists()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
