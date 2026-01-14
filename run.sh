#!/bin/bash

# MedAgentBench Green Agent Launcher Script
#
# This script is used by AgentBeats controller to start the green agent.
# When run standalone, it can also launch evaluations:
#
# Usage:
#   ./run.sh              - Start green agent (for AgentBeats controller)
#   ./run.sh batch        - Run ALL available tasks (batch mode)
#   ./run.sh launch 0     - Run single task with index 0
#   ./run.sh batch --task-indices "0,1,2"  - Run specific tasks
#

set -e

# AgentBeats controller sets HOST and AGENT_PORT environment variables
# If not set, use defaults for standalone mode
if [ -z "$HOST" ]; then
    export HOST="localhost"
fi

if [ -z "$AGENT_PORT" ]; then
    export AGENT_PORT="9001"
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
