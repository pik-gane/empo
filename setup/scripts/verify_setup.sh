#!/bin/bash
# Quick verification script to test the setup
# This script verifies the repository structure and files are correct

set -e

echo "=================================="
echo "EMPO Repository Verification"
echo "=================================="
echo ""

# Check required files
echo "1. Checking required files..."
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "pyproject.toml"
    "setup/requirements.txt"
    "setup/requirements-dev.txt"
    "README.md"
    "Makefile"
    ".dockerignore"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (missing)"
        exit 1
    fi
done

# Check required directories
echo ""
echo "2. Checking directory structure..."
required_dirs=(
    "src/empo"
    "setup/scripts"
    "examples"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ✗ $dir/ (missing)"
        exit 1
    fi
done

# Check Python syntax
echo ""
echo "3. Checking Python syntax..."
python_files=$(find . -name "*.py" \
    -not -path "./.venv/*" \
    -not -path "./venv/*" \
    -not -path "./__pycache__/*" \
    -not -path "*/__pycache__/*" \
    -not -path "./.pytest_cache/*" \
    -not -path "./build/*" \
    -not -path "./dist/*")
for file in $python_files; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (syntax error)"
        exit 1
    fi
done

# Check Docker Compose syntax
echo ""
echo "5. Checking Docker Compose syntax..."
if command -v docker &> /dev/null; then
    if docker compose config --quiet 2>/dev/null; then
        echo "   ✓ docker-compose.yml"
    else
        echo "   ✗ docker-compose.yml (syntax error)"
        exit 1
    fi
else
    echo "   ⚠ Docker not available, skipping"
fi

# Check shell scripts
echo ""
echo "6. Checking shell scripts..."
shell_scripts=$(find setup/scripts -name "*.sh")
for script in $shell_scripts; do
    if bash -n "$script" 2>/dev/null; then
        echo "   ✓ $script"
    else
        echo "   ✗ $script (syntax error)"
        exit 1
    fi
done

echo ""
echo "=================================="
echo "✓ All checks passed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  - For local development: make up && make shell"
echo "  - For cluster deployment: see README.md cluster section"
echo "  - Run training: make train"
echo "  - Run example: make example"
