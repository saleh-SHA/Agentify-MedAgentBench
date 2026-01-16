#!/bin/bash

# =============================================================================
# MedAgentBench - Main Entry Point
# =============================================================================
#
# This is the primary script to run MedAgentBench evaluations.
#
# Prerequisites:
#   1. FHIR server running: ./fhir_launcher.sh (or docker run -p 8080:8080 jyxsu6/medagentbench:latest)
#   2. MCP server running: uv run python -m src.mcp.server
#   3. Environment configured: cp sample.env .env && edit .env
#
# Usage:
#   ./run.sh                              Start green agent server (for AgentBeats controller)
#   ./run.sh batch                        Run ALL available tasks (batch mode)
#   ./run.sh batch --task-indices "0,1,2" Run specific tasks by index
#   ./run.sh launch --task-index 0        Run single task evaluation
#   ./run.sh green                        Start green agent only
#   ./run.sh white                        Start white (purple) agent only
#
# =============================================================================

set -e

# AgentBeats controller sets HOST and AGENT_PORT environment variables
# If not set, use defaults for standalone mode
if [ -z "$HOST" ]; then
    export HOST="0.0.0.0"
fi

if [ -z "$AGENT_PORT" ]; then
    export AGENT_PORT="9009"
fi

echo "================================================================================"
echo "                    MedAgentBench Green Agent"
echo "================================================================================"
echo "Host: $HOST"
echo "Port: $AGENT_PORT"
echo "================================================================================"

# Check if task argument is provided
if [ -z "$1" ]; then
    # No argument provided - start green agent (for AgentBeats controller)
    echo "Starting green agent server..."
    echo "Command: uv run python main.py green"
    echo "================================================================================"
    uv run python main.py green
else
    # Arguments provided - pass through to main.py for evaluation mode
    echo "Running evaluation mode with arguments: $@"
    echo "Command: uv run python main.py $@"
    echo "================================================================================"
    uv run python main.py "$@"
fi
