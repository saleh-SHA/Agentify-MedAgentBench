#!/bin/bash

# MedAgentBench Launcher Script
#
# Usage:
#   ./run.sh              - Run ALL available tasks (batch mode)
#   ./run.sh 0            - Run single task with index 0
#   ./run.sh 5            - Run single task with index 5
#   ./run.sh 0,1,2        - Run batch evaluation with specific tasks (0, 1, 2)
#   ./run.sh 0,5,10       - Run batch evaluation with tasks 0, 5, and 10
#
# The script automatically:
# - Checks and launches FHIR server if not running (port 8080)
# - Checks and launches MCP server if not running (port 8002)
# - Runs the evaluation
# - Cleans up services it started

set -e

# Display banner
echo "================================================================================"
echo "                    MedAgentBench Evaluation Launcher"
echo "================================================================================"
echo ""

# Check if task argument is provided
if [ -z "$1" ]; then
    # No argument provided - run all tasks in batch mode
    echo "Mode: Batch Evaluation (All Tasks)"
    echo "Command: uv run python main.py batch"
    echo "================================================================================"
    uv run python main.py batch
elif [[ "$1" == *","* ]]; then
    # Multiple tasks specified - batch mode
    TASK_INPUT="$1"
    echo "Mode: Batch Evaluation"
    echo "Tasks: $TASK_INPUT"
    echo "Command: uv run python main.py batch --task-indices \"$TASK_INPUT\""
    echo "================================================================================"
    uv run python main.py batch --task-indices "$TASK_INPUT"
else
    # Single task specified - single task mode
    TASK_INPUT="$1"
    echo "Mode: Single Task Evaluation"
    echo "Task Index: $TASK_INPUT"
    echo "Command: uv run python main.py launch --task-index $TASK_INPUT"
    echo "================================================================================"
    uv run python main.py launch --task-index "$TASK_INPUT"
fi

echo ""
echo "================================================================================"
echo "Evaluation Complete!"
echo "================================================================================"
