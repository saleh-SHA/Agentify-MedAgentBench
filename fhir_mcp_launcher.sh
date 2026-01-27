#!/bin/bash

# FHIR and MCP Server Launcher Script
# This script pulls, tags, and runs both the MedAgentBench FHIR server
# and the MCP server containers

set -e  # Exit on error

# Configuration
FHIR_PORT=8080
MCP_PORT=8002
FHIR_CONTAINER_NAME="medagentbench-fhir"
MCP_CONTAINER_NAME="medagentbench-mcp"
MCP_IMAGE_NAME="medagentbench-mcp"

# Function to check if a container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Function to wait for FHIR server to be ready
wait_for_fhir() {
    echo ""
    echo "Waiting for FHIR server to be ready..."
    echo "NOTE: The FHIR server (HAPI FHIR) takes about 1-2 minutes to initialize."
    echo "      Please be patient during this startup process."
    echo ""
    local max_attempts=120
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:${FHIR_PORT}/fhir/metadata" > /dev/null 2>&1; then
            echo ""
            echo "FHIR server is ready!"
            return 0
        fi
        # Show progress every 10 attempts (20 seconds)
        if [ $((attempt % 10)) -eq 0 ]; then
            echo "  Still waiting... ($attempt/$max_attempts attempts, ~$((attempt * 2)) seconds elapsed)"
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "Error: FHIR server did not become ready in time"
    return 1
}

# Function to cleanup containers
cleanup() {
    echo "Cleaning up existing containers..."
    docker rm -f "$FHIR_CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$MCP_CONTAINER_NAME" 2>/dev/null || true
}

# Parse command line arguments
CLEANUP_ONLY=false
if [ "$1" = "--cleanup" ] || [ "$1" = "-c" ]; then
    CLEANUP_ONLY=true
fi

if [ "$CLEANUP_ONLY" = true ]; then
    cleanup
    echo "Cleanup complete."
    exit 0
fi

# Cleanup any existing containers first
cleanup

echo "========================================="
echo "Starting MedAgentBench Services"
echo "========================================="

# Step 1: Pull and start FHIR server
echo ""
echo "Step 1: Setting up FHIR Server"
echo "-----------------------------------------"
echo "Pulling MedAgentBench Docker image..."
docker pull jyxsu6/medagentbench:latest

echo "Tagging image as medagentbench..."
docker tag jyxsu6/medagentbench:latest medagentbench

echo "Starting FHIR server container on port ${FHIR_PORT}..."
docker run -d \
    --name "$FHIR_CONTAINER_NAME" \
    -p ${FHIR_PORT}:8080 \
    medagentbench

# Wait for FHIR server to be ready
wait_for_fhir

# Step 2: Build and start MCP server
echo ""
echo "Step 2: Setting up MCP Server"
echo "-----------------------------------------"
echo "Building MCP server Docker image..."
docker build -t "$MCP_IMAGE_NAME" -f src/mcp/Dockerfile src/mcp/

echo "Starting MCP server container on port ${MCP_PORT}..."
docker run -d \
    --name "$MCP_CONTAINER_NAME" \
    -p ${MCP_PORT}:8002 \
    -e MCP_FHIR_API_BASE=http://host.docker.internal:${FHIR_PORT}/fhir/ \
    --add-host=host.docker.internal:host-gateway \
    "$MCP_IMAGE_NAME"

# Wait a moment for MCP server to start
sleep 3

echo ""
echo "========================================="
echo "Services Started Successfully!"
echo "========================================="
echo ""
echo "FHIR Server:  http://localhost:${FHIR_PORT}/fhir/"
echo "MCP Server:   http://localhost:${MCP_PORT}"
echo ""
echo "Container names:"
echo "  - FHIR: $FHIR_CONTAINER_NAME"
echo "  - MCP:  $MCP_CONTAINER_NAME"
echo ""
echo "To view logs:"
echo "  docker logs -f $FHIR_CONTAINER_NAME"
echo "  docker logs -f $MCP_CONTAINER_NAME"
echo ""
echo "To stop services:"
echo "  ./fhir_launcher.sh --cleanup"
echo ""
