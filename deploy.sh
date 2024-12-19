#!/bin/bash

set -e

CURRENT_VERSION=$(poetry version --short)
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

NEXT_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"

if [ "$1" == "minor" ]; then
  NEXT_VERSION="$MAJOR.$((MINOR + 1)).0"
elif [ "$1" == "major" ]; then
  NEXT_VERSION="$((MAJOR + 1)).0.0"
fi

echo "Current version: $CURRENT_VERSION"
echo "Next version: $NEXT_VERSION"

poetry version "$NEXT_VERSION"

git add .
git commit -m "Bump version to $NEXT_VERSION"
git tag "v$NEXT_VERSION"

git push origin main --tags

echo "✅ Successfully deployed version $NEXT_VERSION to PyPI!"
