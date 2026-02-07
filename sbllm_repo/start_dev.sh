#!/bin/bash
CONTAINER_NAME="project1-dev"
IMAGE_NAME="riscv-opt-env"

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME already exists."
    
    # Check if running
    if docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
        echo "Container is running. Entering..."
        docker exec -it $CONTAINER_NAME bash
    else
        echo "Container is stopped. Starting..."
        docker start -ai $CONTAINER_NAME
    fi
else
    echo "Creating new persistent container $CONTAINER_NAME..."
    docker run -it --name $CONTAINER_NAME \
        -v "$(pwd)":/app \
        $IMAGE_NAME
fi
