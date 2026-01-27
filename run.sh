#!/bin/bash

# =============================================================================
# MedAgentBench - Main Entry Point
# =============================================================================
#
# This script launches the FHIR and MCP servers, then runs the MedAgentBench
# evaluation using AgentBeats.
#
# Usage:
#   ./run.sh                    Launch servers and run evaluation
#   ./run.sh --servers-only     Launch servers only (no evaluation)
#   ./run.sh --cleanup          Stop all running containers and workers
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Using script directory: $SCRIPT_DIR"

# Worker port range (evaluator on 9009, agents on 9019+)
EVALUATOR_PORT=9009
AGENT_BASE_PORT=9019
MAX_WORKERS=50

# Function to kill worker processes on specified ports
cleanup_workers() {
    echo ""
    echo "Cleaning up worker processes..."

    # Kill evaluator process on port 9009
    if lsof -ti:${EVALUATOR_PORT} > /dev/null 2>&1; then
        echo "  Stopping evaluator on port ${EVALUATOR_PORT}..."
        lsof -ti:${EVALUATOR_PORT} | xargs kill -9 2>/dev/null || true
    fi

    # Kill agent worker processes on ports 9019-9068 (up to MAX_WORKERS)
    for ((port=AGENT_BASE_PORT; port<AGENT_BASE_PORT+MAX_WORKERS; port++)); do
        if lsof -ti:${port} > /dev/null 2>&1; then
            echo "  Stopping worker on port ${port}..."
            lsof -ti:${port} | xargs kill -9 2>/dev/null || true
        fi
    done

    # Also kill any remaining Python processes from the scenario
    pkill -f "scenarios/medagentbench/evaluator/src/server.py" 2>/dev/null || true
    pkill -f "scenarios/medagentbench/agent/src/server.py" 2>/dev/null || true

    echo "Worker cleanup complete."
}

# Set up trap to cleanup workers on exit, interrupt, or termination
trap cleanup_workers EXIT INT TERM

# Parse command line arguments
SERVERS_ONLY=false
CLEANUP_ONLY=false

for arg in "$@"; do
    case $arg in
        --servers-only)
            SERVERS_ONLY=true
            shift
            ;;
        --cleanup|-c)
            CLEANUP_ONLY=true
            shift
            ;;
        *)
            ;;
    esac
done

# Handle cleanup
if [ "$CLEANUP_ONLY" = true ]; then
    echo "Stopping MedAgentBench services..."
    cleanup_workers
    "$SCRIPT_DIR/fhir_mcp_launcher.sh" --cleanup
    # Disable trap since we already cleaned up
    trap - EXIT INT TERM
    exit 0
fi

echo "================================================================================"
echo "                         MedAgentBench Runner"
echo "================================================================================"

# Step 1: Launch FHIR and MCP servers
echo ""
echo "Step 1: Starting FHIR and MCP servers..."
echo "--------------------------------------------------------------------------------"

# Check if servers are already running
FHIR_RUNNING=false
MCP_RUNNING=false

if curl -s "http://localhost:8080/fhir/metadata" > /dev/null 2>&1; then
    FHIR_RUNNING=true
    echo "FHIR server is already running on port 8080"
fi

if curl -s "http://localhost:8002/health" > /dev/null 2>&1 || curl -s "http://localhost:8002" > /dev/null 2>&1; then
    MCP_RUNNING=true
    echo "MCP server is already running on port 8002"
fi

# Launch servers if not already running
if [ "$FHIR_RUNNING" = false ] || [ "$MCP_RUNNING" = false ]; then
    echo "Launching servers via fhir_mcp_launcher.sh..."
    "$SCRIPT_DIR/fhir_mcp_launcher.sh"
else
    echo "All servers are already running."
fi

# Verify servers are ready
echo ""
echo "Verifying servers are ready..."

# Verify FHIR server
if curl -s "http://localhost:8080/fhir/metadata" > /dev/null 2>&1; then
    echo "  ✓ FHIR server is ready at http://localhost:8080/fhir/"
else
    echo "  ✗ FHIR server is not responding"
    exit 1
fi

# Verify MCP server
if curl -s "http://localhost:8002" > /dev/null 2>&1; then
    echo "  ✓ MCP server is ready at http://localhost:8002"
else
    echo "  ✗ MCP server is not responding"
    exit 1
fi

# Exit if servers-only mode
if [ "$SERVERS_ONLY" = true ]; then
    echo ""
    echo "================================================================================"
    echo "Servers are running. Use './run.sh --cleanup' to stop them."
    echo "================================================================================"
    # Disable trap since we want servers to keep running
    trap - EXIT INT TERM
    exit 0
fi

# Step 2: Run AgentBeats evaluation
echo ""
echo "Step 2: Running MedAgentBench evaluation..."
echo "--------------------------------------------------------------------------------"
echo "Command: uv run agentbeats-run scenarios/medagentbench/scenario.toml --show-logs"
echo ""
echo "Note: Press Ctrl+C to stop. Workers will be automatically cleaned up."
echo ""

cd "$SCRIPT_DIR"
uv run agentbeats-run scenarios/medagentbench/scenario.toml --show-logs

echo ""
echo "================================================================================"
echo "Evaluation complete. Cleaning up workers..."
echo "================================================================================"
