#!/bin/zsh
# Auto test + build + publish ksom-lab to PyPI.
# Usage:
#   ./scripts/publish.sh            # publish to PyPI
#   ./scripts/publish.sh --testpypi # publish to TestPyPI
set -e

cd "$(dirname "$0")/.."
VENV=.venv
REPO=pypi
[[ "$1" == "--testpypi" ]] && REPO=testpypi

echo "==> Running tests"
$VENV/bin/python -m pytest tests -q

echo "==> Syncing dev dependencies"
uv sync --quiet

echo "==> Cleaning old builds"
rm -rf dist build *.egg-info

echo "==> Building package"
$VENV/bin/python -m build

echo "==> Checking package"
$VENV/bin/twine check dist/*

if [[ "$REPO" == "testpypi" ]]; then
  echo "==> Uploading to TestPyPI"
  $VENV/bin/twine upload --repository testpypi dist/*
else
  echo "==> Uploading to PyPI"
  $VENV/bin/twine upload dist/*
fi

echo "==> Done"
