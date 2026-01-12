# MedAgentBench Evaluation Framework

A complete implementation of MedAgentBench using the A2A (Agent-to-Agent) protocol and MCP (Model Context Protocol) servers for evaluating medical AI agents on clinical reasoning tasks.

## Overview

This project provides a standardized framework for evaluating medical AI agents using FHIR (Fast Healthcare Interoperability Resources) servers. It implements a green-agent/white-agent architecture where:
- **Green agent**: Assessment manager that coordinates evaluations
- **White agent**: The AI agent being tested (uses GPT-5 + function calling)
- **MCP server**: Provides 9 FHIR tools for medical data access and serves evaluation tasks
- **FHIR server**: Contains medical records and patient data

## Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Green     │  Task + MCP URL    │   White     │
│   Agent     │──────────────────>│   Agent     │
│ (Evaluator) │                    │ (GPT-5)     │
└─────────────┘                    └─────────────┘
                                          │
                                          │ Discover & invoke tools
                                          ▼
                                   ┌─────────────┐
                                   │     MCP     │
                                   │   Server    │
                                   │   :8002     │
                                   └─────────────┘
                                          │
                                          │ FHIR requests
                                          ▼
                                   ┌─────────────┐
                                   │    FHIR     │
                                   │   Server    │
                                   │   :8080     │
                                   └─────────────┘
```

## Project Structure

```
agentify-medagentbench/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project dependencies
├── .env                            # API keys (create this)
├── src/
│   ├── green_agent/
│   │   ├── medagent.py             # Green agent (evaluator)
│   │   └── medagent_green_agent.toml
│   ├── white_agent/
│   │   └── medagent.py             # White agent (GPT-5 + MCP)
│   ├── mcp/
│   │   ├── server.py               # MCP server (9 FHIR tools + task serving)
│   │   └── resources/
│   │       └── tasks/
│   │           └── tasks.json      # Test cases served by MCP server
│   ├── medagent_launcher.py        # Evaluation launcher
│   └── my_util/
│       └── my_a2a.py               # A2A protocol utilities
```

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the FHIR Server (Terminal 1)

```bash
docker pull jyxsu6/medagentbench:latest
docker tag jyxsu6/medagentbench:latest medagentbench
docker run -p 8080:8080 medagentbench
```

Wait for: `Started Application in XXX seconds`

Verify: http://localhost:8080/

### 4. Start the MCP Server (Terminal 2)

```bash
uv run python -m src.mcp.server
```

Verify:
```bash
curl http://0.0.0.0:8002/health
# Should return: {"status":"ok","uptime_seconds":X.X}
```

### 5. Run Evaluation (Terminal 3)

```bash
# Run single task
uv run python main.py launch

# Run batch evaluation
uv run python main.py batch --task-indices "0,1,2"
```

## Usage

### Single Task Evaluation

Run one task at a time:

```bash
# Run task at index 0 (default)
uv run python main.py launch

# Run a specific task
uv run python main.py launch --task-index 5

# Customize settings
uv run python main.py launch \
  --task-index 0 \
  --mcp-server http://0.0.0.0:8002 \
  --max-rounds 10
```

### Batch Evaluation

Run multiple tasks in sequence:

```bash
# Run all tasks
uv run python main.py batch

# Run specific tasks (e.g., tasks 0, 1, 2)
uv run python main.py batch --task-indices "0,1,2"

# Run with custom settings
uv run python main.py batch \
  --task-indices "0,5,10" \
  --mcp-server http://0.0.0.0:8002 \
  --max-rounds 10

# Save results to a file
uv run python main.py batch \
  --task-indices "0,1,2" \
  --output-file results.json
```

### Run Individual Components

For debugging or development:

```bash
# Start green agent only (port 9001)
uv run python main.py green

# Start white agent only (port 9002)
uv run python main.py white
```

## CLI Commands

### `launch` - Single Task Evaluation
Run a single task evaluation.

**Options:**
- `--task-index INT` - Task index to run (default: 0)
- `--mcp-server STR` - MCP server URL (default: http://0.0.0.0:8002)
- `--max-rounds INT` - Maximum interaction rounds (default: 8)

### `batch` - Batch Evaluation
Run multiple tasks in sequence.

**Options:**
- `--task-indices STR` - Comma-separated task indices (e.g., "0,1,2"). If not provided, runs all tasks.
- `--mcp-server STR` - MCP server URL (default: http://0.0.0.0:8002)
- `--max-rounds INT` - Maximum interaction rounds (default: 8)
- `--output-file STR` - Path to save results as JSON file (optional). Note: Existing file will be overwritten.

### `green` - Start Green Agent
Start the green agent (assessment manager) only on port 9001.

### `white` - Start White Agent
Start the white agent (agent being tested) only on port 9002.

## How It Works

### 1. Evaluation Flow

1. **Launcher** loads test data from MCP server and starts both agents
2. **Green agent** sends task + MCP server URL to white agent
3. **White agent** discovers available tools from MCP server
4. **White agent** uses GPT-5 with function calling to solve the task
5. **White agent** invokes tools via MCP server to query FHIR data
6. **White agent** returns answer using `FINISH(answer)` format
7. **Green agent** evaluates answer correctness using flexible matching
8. **Green agent** reports metrics (correctness, time, rounds)

### 2. Green Agent (Assessment Manager)

**File**: [src/green_agent/medagent.py](src/green_agent/medagent.py)

The green agent:
- Receives task configuration with MCP server URL
- Sends task to white agent with instructions to use the MCP server
- Waits for white agent responses
- Checks if answer is correct when white agent calls `FINISH(answer)` using **flexible matching**:
  - First tries **exact match**: Full answer matches expected solution
  - Falls back to **substring match**: Expected solution is contained in answer
  - Examples: `"S6534835"` ✅, `"Patient MRN is S6534835"` ✅, `"S6534836"` ❌
- Reports evaluation results with metrics (correctness, time, rounds)

### 3. White Agent (Agent Being Tested)

**File**: [src/white_agent/medagent.py](src/white_agent/medagent.py)

The MedAgentBench white agent:
- Uses **GPT-5** with function calling capability (no temperature parameter support)
- Discovers tools from MCP server at startup
- Converts MCP tool descriptors to OpenAI function format
- Calls tools via MCP server when GPT-5 requests them
- Processes FHIR responses and continues reasoning
- Returns final answer via `FINISH(answer)`

### 4. MCP Server

**File**: [src/mcp/server.py](src/mcp/server.py)

The MCP server provides two main functions:

**A. Task Management**
- Loads and serves evaluation tasks from `src/mcp/resources/tasks/tasks.json`
- Provides tasks via `/resources/medagentbench_tasks` endpoint
- Both single and batch evaluations retrieve tasks from the MCP server

**B. 9 FHIR Tools**
1. **search_patients** - Search for patients by name, DOB, identifier
2. **list_patient_problems** - Get patient conditions/problem list
3. **list_lab_observations** - Get laboratory results
4. **list_vital_signs** - Get vital sign observations
5. **record_vital_observation** - Create a new vital sign observation
6. **list_medication_requests** - Get medication orders
7. **create_medication_request** - Create a new medication order
8. **list_patient_procedures** - Get completed procedures
9. **create_service_request** - Create lab/imaging/consult orders

## Test Data

Test cases are stored in `src/mcp/resources/tasks/tasks.json` and served by the MCP server.

Each test case includes:
- `id`: Task identifier (e.g., "task1_1")
- `instruction`: Medical question
- `context`: Additional constraints or context
- `sol`: Expected solution(s) as a list

Example:
```json
{
  "id": "task1_1",
  "instruction": "What's the MRN of the patient with name Peter Stafford and DOB of 1932-12-29?",
  "context": "If the patient does not exist, the answer should be \"Patient not found\"",
  "sol": ["S6534835"]
}
```

**Note**: The MCP server loads tasks at startup. To check available tasks:
```bash
curl http://0.0.0.0:8002/resources/medagentbench_tasks | jq
```

## Evaluation Metrics

Each evaluation reports:
- **correct**: Whether the answer was correct (True/False)
- **time_used**: Time taken in seconds
- **rounds_used**: Number of interaction rounds
- **status**: Task completion status ("completed", "max_rounds_reached", etc.)

## Evaluation Logic

The green agent uses **flexible matching** to evaluate answers:
1. First tries **exact match**: Full answer matches expected solution
2. Falls back to **substring match**: Expected solution is contained in answer

Examples:
- `"S6534835"` → ✅ (exact match)
- `"Patient MRN is S6534835"` → ✅ (contains "S6534835")
- `"The patient's MRN number is S6534835"` → ✅ (contains "S6534835")
- `"S6534836"` → ❌ (wrong MRN)
- `"Patient not found"` → ✅ (if that's the expected answer)

## Example Output

```
MedAgentBench Evaluation Complete ✅

Task ID: task1_1
Question: What's the MRN of the patient with name Peter Stafford and DOB of 1932-12-29?
Expected Answer: ['S6534835']
Agent Answer: Patient MRN is S6534835
Status: completed
Correct: True
Rounds Used: 1
Time: 61.91s

Metrics: {
  "time_used": 61.91,
  "status": "completed",
  "rounds_used": 1,
  "correct": true
}
```

## Programmatic Usage

You can also use the MedAgentBench components programmatically:

```python
import asyncio
from src.medagent_launcher import launch_medagent_evaluation, launch_medagent_batch_evaluation

# Run a specific task
result = asyncio.run(launch_medagent_evaluation(
    task_index=0,
    mcp_server_url="http://0.0.0.0:8002",
    max_rounds=8
))
print(result)  # Returns dict with task_index, task_id, and response

# Run batch evaluation
results = asyncio.run(launch_medagent_batch_evaluation(
    task_indices=[0, 1, 2],
    mcp_server_url="http://0.0.0.0:8002",
    max_rounds=8,
    output_file="results.json"  # Optional
))
print(f"Evaluated {len(results)} tasks")
```

## Technologies

- **A2A Protocol**: Agent-to-agent communication standard
- **MCP**: Model Context Protocol for tool management
- **FHIR**: Fast Healthcare Interoperability Resources standard
- **GPT-5**: OpenAI's latest model with function calling
- **Docker**: FHIR server containerization
- **FastAPI**: MCP server web framework
- **LiteLLM**: Universal LLM API wrapper
- **Uvicorn**: ASGI web server
- **Typer**: CLI framework

## Troubleshooting

### FHIR Server Not Responding

```bash
# Check if running
docker ps | grep medagentbench

# Check logs
docker logs <container_id>

# Restart
docker restart <container_id>
```

### MCP Server Connection Issues

```bash
# Verify MCP server is running
curl http://0.0.0.0:8002/health

# Check available tools
curl http://0.0.0.0:8002/tools | jq

# Restart MCP server
# Stop with Ctrl+C, then:
uv run python -m src.mcp.server
```

### GPT-5 API Errors

- Check `.env` file has valid `OPENAI_API_KEY`
- Verify API quota at https://platform.openai.com/account/usage
- Note: GPT-5 doesn't support `temperature` parameter

### Agents Not Starting

```bash
# Check if ports are available
lsof -i :9001  # Green agent
lsof -i :9002  # White agent
lsof -i :8002  # MCP server
lsof -i :8080  # FHIR server

# Kill any existing processes
kill -9 <PID>
```

### Common Issues

**Issue**: Schema validation errors from OpenAI API (e.g., "array schema missing items")
- **Cause**: Invalid JSON schema for array-type parameters in tool definitions
- **Solution**: Ensure all array properties in tool schemas include an `items` field. Restart MCP server to load updated tool definitions.

**Issue**: Agent processes not terminating
- **Solution**: Use `ps aux | grep python` and `kill -9 <PID>`

**Issue**: Docker FHIR server not starting
- **Solution**: Check if port 8080 is already in use: `lsof -i :8080`

**Issue**: Tasks not found or empty task list
- **Cause**: MCP server cannot find or load tasks.json file
- **Solution**: Verify `src/mcp/resources/tasks/tasks.json` exists and contains valid JSON. Check MCP server startup logs for task loading status.

## Key Files

- [src/green_agent/medagent.py](src/green_agent/medagent.py) - Green agent implementation
- [src/white_agent/medagent.py](src/white_agent/medagent.py) - White agent with MCP support
- [src/mcp/server.py](src/mcp/server.py) - MCP server providing FHIR tools and task serving
- [src/mcp/resources/tasks/tasks.json](src/mcp/resources/tasks/tasks.json) - Test cases served by MCP server
- [src/medagent_launcher.py](src/medagent_launcher.py) - Evaluation launcher
- [main.py](main.py) - CLI entry point

## Next Steps

- ✅ Batch evaluation support (implemented via `batch` command)
- ✅ Flexible answer matching (exact + substring)
- ✅ MCP server task management (tasks loaded from MCP server)
- ✅ Fixed JSON schema validation (array items properly defined)
- ✅ Result collection and aggregation for batch evaluations
- ✅ Output file saving for batch results (JSON format)
- Customize white agent reasoning strategies
- Add more FHIR tools to MCP server
- Add support for different LLM backends (currently uses GPT-5)
- Generate summary statistics and reports from batch results

## License

MIT
