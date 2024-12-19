#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: ./deploy.sh [patch|minor|major]"
  exit 1
fi

poetry version $1
VERSION=$(poetry version --short)

git add .
git commit -m "Bump version to $VERSION"
git tag v$VERSION

poetry publish --build

git push origin main --tags

echo "✅ Successfully deployed version $VERSION to PyPI!"
