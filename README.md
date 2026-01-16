# MedAgentBench Evaluation Framework

An implementation of MedAgentBench using the A2A (Agent-to-Agent) protocol and MCP (Model Context Protocol) for evaluating medical AI agents on clinical reasoning tasks.

## Overview

This framework evaluates medical AI agents using FHIR (Fast Healthcare Interoperability Resources) servers. It uses a green-agent/white-agent architecture:

- **Green agent**: Assessment manager that coordinates evaluations and grades results
- **White agent**: The AI agent being tested
- **MCP server**: Provides 9 FHIR tools for medical data access and serves evaluation tasks
- **FHIR server**: Contains medical records and patient data

## Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Green     │  Task + MCP URL    │   White     │
│   Agent     │───────────────────>│   Agent     │
│ (Evaluator) │                    │  (LLM)      │
└─────────────┘                    └─────────────┘
   :9001                              │
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
├── main.py                              # CLI entry point
├── pyproject.toml                       # Project dependencies
├── .env                                 # API keys (create this)
├── src/
│   ├── green_agent/
│   │   ├── medagent.py                  # Green agent (evaluator)
│   │   ├── medagent_green_agent.toml    # Agent card config
│   │   └── eval_resources/
│   │       ├── eval.py                  # Evaluation orchestrator
│   │       ├── refsol.py                # Reference solutions & grading
│   │       └── utils.py                 # HTTP utilities
│   ├── white_agent/
│   │   └── medagent.py                  # White agent (LLM + MCP tools)
│   ├── mcp/
│   │   ├── server.py                    # FastMCP server (9 FHIR tools)
│   │   └── resources/tasks/tasks.json   # Evaluation tasks
│   ├── medagent_launcher.py             # Evaluation launcher
│   ├── typings/                         # Type definitions
│   ├── utils/                           # Utility functions
│   └── my_util/
│       ├── my_a2a.py                    # A2A protocol utilities
│       └── logging_config.py            # Logging setup
└── outputs/                             # Evaluation results
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

The server starts on port 8002 and logs the number of loaded tasks.

### 5. Run Evaluation (Terminal 3)

```bash
# Run single task
uv run python main.py launch

# Run batch evaluation
uv run python main.py batch --task-indices "0,1,2"
```

## Usage

### Single Task Evaluation

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

```bash
# Run all tasks
uv run python main.py batch

# Run specific tasks
uv run python main.py batch --task-indices "0,1,2"

# Save results to file
uv run python main.py batch \
  --task-indices "0,1,2" \
  --output-file results.json
```

### Run Individual Components

For debugging or development:

```bash
# Start green agent only
uv run python main.py green

# Start white agent only
uv run python main.py white
```

## CLI Commands

### `launch` - Single Task Evaluation

| Option         | Type | Default             | Description                |
| -------------- | ---- | ------------------- | -------------------------- |
| `--task-index` | INT  | 0                   | Task index to run          |
| `--mcp-server` | STR  | http://0.0.0.0:8002 | MCP server URL             |
| `--max-rounds` | INT  | 8                   | Maximum interaction rounds |

### `batch` - Batch Evaluation

| Option           | Type | Default             | Description                             |
| ---------------- | ---- | ------------------- | --------------------------------------- |
| `--task-indices` | STR  | None (all)          | Comma-separated indices (e.g., "0,1,2") |
| `--mcp-server`   | STR  | http://0.0.0.0:8002 | MCP server URL                          |
| `--max-rounds`   | INT  | 8                   | Maximum interaction rounds              |
| `--output-file`  | STR  | None                | Path to save results JSON               |

### `green` / `white` - Start Individual Agents

Start agents separately for debugging. Ports are configured via environment variables or defaults.

## How It Works

### Evaluation Flow

1. **Launcher** loads tasks from MCP server and starts both agents
2. **Green agent** sends task + MCP server URL to white agent
3. **White agent** discovers available tools from MCP server
4. **White agent** uses LLM with function calling to solve the task
5. **White agent** invokes FHIR tools via MCP server
6. **White agent** returns answer using `FINISH([answer])` format
7. **Green agent** evaluates using task-specific grading logic
8. **Green agent** writes results to `outputs/` directory

### Green Agent (Assessment Manager)

**File**: `src/green_agent/medagent.py`

The green agent:

- Receives task configuration with MCP server URL
- Sends task prompt to white agent
- Receives final answer in `FINISH([answer])` format
- Evaluates answer using task-specific grading functions in `eval_resources/refsol.py`
- Writes results to `runs.jsonl` and computes `overall.json` metrics

### White Agent (Agent Under Test)

**File**: `src/white_agent/medagent.py`

The white agent:

- Uses LiteLLM for LLM calls (configurable model)
- Discovers tools from MCP server at startup
- Converts MCP tool descriptors to OpenAI function format
- Executes tool calls via MCP server
- Returns final answer via `FINISH([answer])`

### MCP Server

**File**: `src/mcp/server.py`

Built with FastMCP, provides:

**Task Management**

- Loads tasks from `src/mcp/resources/tasks/tasks.json`
- Serves tasks via `medagentbench://tasks` resource

**9 FHIR Tools**

1. `search_patients` - Search patients by name, DOB, identifier
2. `list_patient_problems` - Get patient conditions/problem list
3. `list_lab_observations` - Get laboratory results
4. `list_vital_signs` - Get vital sign observations
5. `record_vital_observation` - Create a vital sign observation
6. `list_medication_requests` - Get medication orders
7. `create_medication_request` - Create a medication order
8. `list_patient_procedures` - Get completed procedures
9. `create_service_request` - Create lab/imaging/consult orders

## Task Types

The benchmark includes 10 task types with multiple instances each:

| Task   | Description                    | Requires POST        |
| ------ | ------------------------------ | -------------------- |
| task1  | Patient lookup (MRN)           | No                   |
| task2  | Age calculation                | No                   |
| task3  | Record vital signs             | Yes (Observation)    |
| task4  | Lab value lookup (24h window)  | No                   |
| task5  | Magnesium replacement protocol | Conditional          |
| task6  | Glucose average calculation    | No                   |
| task7  | Latest glucose lookup          | No                   |
| task8  | Orthopedic consult request     | Yes (ServiceRequest) |
| task9  | Potassium replacement protocol | Conditional          |
| task10 | HbA1C check and order if stale | Conditional          |

## Output Files

Results are written to `outputs/<agent_name>/<task_name>/`:

- `runs.jsonl` - Individual task results (one JSON per line)
- `error.jsonl` - Failed task records
- `overall.json` - Aggregate metrics (success rate, history stats)

## Programmatic Usage

```python
import asyncio
from src.medagent_launcher import launch_medagent_evaluation, launch_medagent_batch_evaluation

# Run a single task
result = asyncio.run(launch_medagent_evaluation(
    task_index=0,
    mcp_server_url="http://0.0.0.0:8002",
    max_rounds=8
))
print(result)  # Returns dict with task_index, task_id, response

# Run batch evaluation
results = asyncio.run(launch_medagent_batch_evaluation(
    task_indices=[0, 1, 2],
    mcp_server_url="http://0.0.0.0:8002",
    max_rounds=8,
    output_file="results.json"
))
print(f"Evaluated {len(results)} tasks")
```

## Technologies

| Technology   | Purpose                          |
| ------------ | -------------------------------- |
| A2A Protocol | Agent-to-agent communication     |
| MCP          | Model Context Protocol for tools |
| FastMCP      | MCP server framework             |
| FHIR         | Healthcare data standard         |
| LiteLLM      | Universal LLM API wrapper        |
| Typer        | CLI framework                    |
| Pydantic     | Data validation                  |
| Uvicorn      | ASGI server                      |
| Docker       | FHIR server containerization     |

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

### MCP Server Issues

```bash
# Restart MCP server
uv run python -m src.mcp.server
```

### Port Conflicts

```bash
# Check port usage
lsof -i :9001  # Green agent (launcher)
lsof -i :9002  # White agent
lsof -i :8002  # MCP server
lsof -i :8080  # FHIR server

# Kill process
kill -9 <PID>
```

### API Errors

- Verify `OPENAI_API_KEY` in `.env` file
- Check API quota at https://platform.openai.com/account/usage

### Tasks Not Loading

- Verify `src/mcp/resources/tasks/tasks.json` exists
- Check MCP server startup logs for task count

## Key Files

| File                                       | Description                |
| ------------------------------------------ | -------------------------- |
| `main.py`                                  | CLI entry point            |
| `src/medagent_launcher.py`                 | Evaluation orchestration   |
| `src/green_agent/medagent.py`              | Green agent implementation |
| `src/green_agent/eval_resources/refsol.py` | Task grading functions     |
| `src/white_agent/medagent.py`              | White agent with MCP tools |
| `src/mcp/server.py`                        | FastMCP server             |
| `src/mcp/resources/tasks/tasks.json`       | Evaluation tasks           |

## License

MIT
