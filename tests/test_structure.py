#!/usr/bin/env python3
"""
Simple test script that can run without pytest.
Verifies basic repository structure and imports.
"""

import sys
import tomllib
from pathlib import Path

# Get project root directory (resolve to handle symlinks properly)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_import_empo():
    """Test that the empo package can be imported.

    This test is skipped if dependencies like gymnasium aren't installed,
    since the Docker build will catch real import issues.
    """
    try:
        import empo

        assert empo.__version__ == "0.1.0"
    except ImportError:
        # Dependencies not installed - skip this test gracefully
        # The Docker build will catch real import issues
        print("SKIPPED: Missing dependencies - run in Docker for full validation")
        return  # Skip without pytest


def test_requirements_exist():
    """Test that requirement files exist."""
    req_file = PROJECT_ROOT / "setup" / "requirements.txt"
    req_dev_file = PROJECT_ROOT / "setup" / "requirements-dev.txt"
    assert req_file.exists(), f"requirements.txt not found at {req_file}"
    assert req_dev_file.exists(), f"requirements-dev.txt not found at {req_dev_file}"


def test_pyproject_exists():
    """Test that the root pyproject.toml exists and is minimally configured."""
    pyproject_file = PROJECT_ROOT / "pyproject.toml"
    assert pyproject_file.exists(), f"pyproject.toml not found at {pyproject_file}"

    pyproject = tomllib.loads(pyproject_file.read_text())
    assert pyproject["project"]["name"] == "empo"
    assert "dev" in pyproject["project"]["optional-dependencies"]


def test_dockerfile_exists():
    """Test that Dockerfile exists.

    Skip this test when running inside Docker container since
    Docker files are excluded via .dockerignore.
    """
    dockerfile = PROJECT_ROOT / "Dockerfile"
    if not dockerfile.exists() and Path("/.dockerenv").exists():
        # Running inside Docker - skip gracefully
        import pytest

        pytest.skip("Dockerfile not copied into Docker container (per .dockerignore)")
    assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"


def test_docker_compose_exists():
    """Test that docker-compose.yml exists.

    Skip this test when running inside Docker container since
    Docker files are excluded via .dockerignore.
    """
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    if not compose_file.exists() and Path("/.dockerenv").exists():
        # Running inside Docker - skip gracefully
        import pytest

        pytest.skip("docker-compose.yml not copied into Docker container (per .dockerignore)")
    assert compose_file.exists(), f"docker-compose.yml not found at {compose_file}"


def main():
    """Run all tests when executed directly."""
    tests = [
        test_import_empo,
        test_requirements_exist,
        test_pyproject_exists,
        test_dockerfile_exists,
        test_docker_compose_exists,
    ]

    failed = []
    for test in tests:
        try:
            print(f"Running {test.__name__}...", end=" ")
            test()
            print("PASSED")
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(test.__name__)

    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed!")


if __name__ == "__main__":
    main()
