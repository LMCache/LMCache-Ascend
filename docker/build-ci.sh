#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-hub.oepkgs.net/ai/lmcache-ascend}"
TAG="${TAG:-a2-v0.3.7-vllm-ascend-v0.10.2rc1-cann-8.2rc1-py3.11-openeuler-24.03}"

# Get directory logic
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR=$(dirname "$SCRIPTDIR")

echo ">>> Building image: $IMAGE:$TAG"

DOCKER_ARGS=()

DOCKER_ARGS+=( "--build-arg" "http_proxy=${http_proxy:-}" )
DOCKER_ARGS+=( "--build-arg" "https_proxy=${https_proxy:-}" )
DOCKER_ARGS+=( "--build-arg" "no_proxy=${no_proxy:-}" )

echo ">>> Running Docker Build..."
docker build \
  "${DOCKER_ARGS[@]}" \
  -f "$SCRIPTDIR/Dockerfile.a2.openEuler" \
  -t "$IMAGE:$TAG" \
  "$BASEDIR"

echo ">>> Build completed: $IMAGE:$TAG"