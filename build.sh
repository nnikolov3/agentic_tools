#!/bin/bash

# --- Configuration ---
IMAGE_NAME="agentic-tools"
CONTAINER_NAME="agentic-tools-dev"
BUILD_CONTEXT="."

# --- Build Function ---
echo "Building Podman image: ${IMAGE_NAME}..."
podman build -t "${IMAGE_NAME}" "${BUILD_CONTEXT}"

if [ $? -ne 0 ]; then
    echo "Error: Image build failed."
    exit 1
fi

echo "Build successful. Image: ${IMAGE_NAME}:latest"
echo ""
echo "--- Run Instructions ---"
echo "To run the container and execute an agent, use the following command structure:"
echo ""
echo "podman run --rm -it \
  --network=host \
  -v \"$(pwd)\":/app:z \
  -e GEMINI_API_KEY_EXPERT=\"<YOUR_API_KEY>\" \
  	${IMAGE_NAME} run-agent linter_analyst"
echo ""
echo "To use Podman Secrets for the API key, follow the previous instructions."
