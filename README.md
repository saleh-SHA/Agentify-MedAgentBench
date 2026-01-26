# MedAgentBench for AgentBeats

A medical AI agent evaluation framework compatible with the [AgentBeats](https://github.com/RDI-Foundation/agentbeats-tutorial) platform. This benchmark evaluates AI agents on clinical reasoning tasks using FHIR (Fast Healthcare Interoperability Resources) servers.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running Evaluations](#running-evaluations)
- [Configuration](#configuration)
- [Task Types](#task-types)
- [FHIR Tools](#fhir-tools-via-mcp-server)
- [Environment Variables](#environment-variables)
- [Building Docker Images](#building-docker-images)
- [Technologies](#technologies)

---

## Overview

MedAgentBench evaluates medical AI agents on their ability to:

- Query patient information from FHIR servers
- Perform clinical calculations (e.g., patient age, average lab values)
- Create medical orders based on clinical conditions
- Follow medical protocols (e.g., potassium replacement)

## Architecture

MedAgentBench implements a **green-agent/purple-agent architecture**:

| Agent            | Role        | Description                                                     |
| ---------------- | ----------- | --------------------------------------------------------------- |
| **Green Agent**  | Evaluator   | Orchestrates evaluations, sends tasks, validates answers        |
| **Purple Agent** | Participant | The AI agent being tested - uses FHIR tools to answer questions |

```
┌──────────────┐      task       ┌──────────────┐      tools      ┌────────────┐
│ GREEN AGENT  │ ──────────────> │ PURPLE AGENT │ <─────────────> │ MCP SERVER │
│  (Evaluator) │ <────────────── │ (Participant)│                 │   :8002    │
│    :9009     │    FINISH()     │    :9019     │                 └─────┬──────┘
└──────────────┘                 └──────────────┘                       │
                                                                        │ FHIR API
                                                                        v
                                                                 ┌────────────┐
                                                                 │FHIR SERVER │
                                                                 │   :8080    │
                                                                 └────────────┘
```

The evaluation flow:

1. Green agent sends a medical question to the purple agent
2. Purple agent uses FHIR tools (via MCP server) to query/modify patient records
3. Purple agent returns answer in `FINISH([answer1, answer2, ...])` format
4. Green agent validates using task-specific grading functions

---

## Project Structure

```
MedAgentBench/
├── run.sh                          # Main entry point
├── main.py                         # CLI for standalone execution
├── fhir_launcher.sh                # Docker FHIR server launcher
├── sample.env                      # Environment configuration template
├── pyproject.toml                  # Project dependencies
│
├── scenarios/medagentbench/        # AgentBeats scenario implementation
│   ├── scenario.toml               # Scenario configuration
│   ├── evaluator/src/              # Green agent (evaluator)
│   │   ├── server.py               # A2A server entry point
│   │   ├── agent.py                # Evaluation logic + task graders
│   │   ├── executor.py             # AgentExecutor implementation
│   │   └── messenger.py            # A2A client helper
│   ├── agent/src/                  # Purple agent (participant)
│   │   ├── server.py               # A2A server entry point
│   │   ├── agent.py                # LLM + MCP tool calling logic
│   │   └── executor.py             # AgentExecutor implementation
│   ├── Dockerfile.medagentbench-evaluator
│   └── Dockerfile.medagentbench-agent
│
├── src/
│   ├── agentbeats/                 # AgentBeats runner + A2A client helpers
│   ├── mcp/                        # MCP server with FHIR tools
│   │   ├── server.py               # FastMCP server implementation
│   │   └── resources/tasks/tasks.json  # Evaluation tasks
│   ├── green_agent/                # Original green agent implementation
│   ├── white_agent/                # Original participant agent (white = purple)
│   └── medagent_launcher.py        # Original evaluation launcher
│
├── outputs/                        # Evaluation results
└── logs/                           # Agent logs
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for FHIR server)
- OpenAI API key

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp sample.env .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start External Services

MedAgentBench requires two services running:

**Terminal 1 - FHIR Server (Docker):**

```bash
./fhir_launcher.sh
# Or manually:
# docker run -p 8080:8080 jyxsu6/medagentbench:latest
```

Wait for: `Started Application in XXX seconds`

**Terminal 2 - MCP Server:**

```bash
uv run python -m src.mcp.server
```

### 4. Run Evaluation

**Terminal 3 - Using run.sh (Recommended):**

```bash
# Run batch evaluation on all tasks
./run.sh batch

# Run specific tasks
./run.sh batch --task-indices "0,1,2"

# Run single task
./run.sh launch --task-index 0
```

---

## Running Evaluations

### Using `run.sh` (Main Entry Point)

The `run.sh` script is the primary way to run MedAgentBench:

```bash
# Start green agent server (for AgentBeats controller)
./run.sh

# Run batch evaluation on ALL tasks
./run.sh batch

# Run batch evaluation on specific tasks
./run.sh batch --task-indices "0,1,2,3,4"

# Run single task evaluation
./run.sh launch --task-index 0
```

### Using AgentBeats Runner

For integration with the AgentBeats platform:

```bash
uv run agentbeats-run scenarios/medagentbench/scenario.toml
```

**Options:**

- `--show-logs`: Show agent stdout/stderr during assessment
- `--serve-only`: Start agents without running the assessment

### Using Python CLI Directly

```bash
# Start green agent only
uv run python main.py green

# Start white (purple) agent only
uv run python main.py white

# Run single task
uv run python main.py launch --task-index 0

# Run batch evaluation
uv run python main.py batch --task-indices "0,1,2"
```

---

## Configuration

### Scenario Configuration

Edit `scenarios/medagentbench/scenario.toml`:

```toml
[green_agent]
endpoint = "http://localhost:9009"
cmd = "python scenarios/medagentbench/evaluator/src/server.py --host 0.0.0.0 --port 9009"

[[participants]]
role = "agent"
endpoint = "http://localhost:9019"
cmd = "python scenarios/medagentbench/agent/src/server.py --host 0.0.0.0 --port 9019"

[config]
num_tasks = 5                                # Number of tasks to run
max_rounds = 10                              # Max tool-calling rounds per task
mcp_server_url = "http://localhost:8002"     # MCP server URL
fhir_api_base = "http://localhost:8080/fhir/" # FHIR server URL
```

### Assessment Request Format

The green agent receives a JSON assessment request:

```json
{
  "participants": { "agent": "http://localhost:9019" },
  "config": {
    "num_tasks": 5,
    "max_rounds": 10,
    "domain": "medagentbench",
    "mcp_server_url": "http://localhost:8002",
    "fhir_api_base": "http://localhost:8080/fhir/"
  }
}
```

---

## Task Types

MedAgentBench includes **10 task types** covering various clinical scenarios:

| Task   | Type        | Description                                      |
| ------ | ----------- | ------------------------------------------------ |
| task1  | Query       | Patient lookup by demographics (name, DOB)       |
| task2  | Query       | Calculate patient age                            |
| task3  | Write       | Record vital sign observation (blood pressure)   |
| task4  | Query       | Recent lab value lookup (within 24h)             |
| task5  | Conditional | Check magnesium level & order replacement if low |
| task6  | Query       | Average glucose value (within 24h)               |
| task7  | Query       | Latest glucose value                             |
| task8  | Write       | Create orthopedic consultation request           |
| task9  | Conditional | Potassium replacement protocol                   |
| task10 | Conditional | HbA1C check & order lab if >1 year old           |

### Task Descriptions

**Query Tasks** - Read-only operations that retrieve patient data:

- **task1**: Find patient MRN by searching with name and date of birth. Return "Patient not found" if no match.
- **task2**: Calculate a patient's current age from their birth date (as of 2023-11-13).
- **task4**: Retrieve the most recent magnesium (MG) lab value recorded within the last 24 hours. Return -1 if unavailable.
- **task6**: Calculate the average of all glucose (GLU) readings within the last 24 hours. Return -1 if no readings.
- **task7**: Retrieve the most recent glucose (GLU) value from the patient's chart. Return -1 if unavailable.

**Write Tasks** - Create new FHIR resources:

- **task3**: Record a blood pressure vital sign observation (118/77 mmHg) for a patient with proper FHIR Observation structure.
- **task8**: Create an orthopedic consultation ServiceRequest for a patient with ACL tear, including SBAR-formatted clinical notes.

**Conditional Tasks** - Clinical decision-making with protocol-based actions:

- **task5**: Magnesium replacement protocol - Check serum magnesium level within 24h. If Mg <= 1.9 mEq/L, order IV magnesium sulfate with dose based on severity (4g if <1.0, 2g if <1.5, 1g otherwise).
- **task9**: Potassium replacement protocol - Check serum potassium level. If K < 3.5 mEq/L, order oral potassium chloride (dose = (3.5-K)/0.1 \* 10 mEq) and schedule a follow-up serum potassium lab for the next morning.
- **task10**: HbA1C monitoring - Retrieve the last HbA1C value and date. If more than 1 year old or unavailable, order a new HbA1C lab test (LOINC 4548-4).

---

## FHIR Tools (via MCP Server)

The MCP server provides **9 FHIR tools** for interacting with patient data:

| Tool                        | Type | Description                              |
| --------------------------- | ---- | ---------------------------------------- |
| `search_patients`           | GET  | Search patients by name, DOB, identifier |
| `list_patient_problems`     | GET  | Get patient conditions/problem list      |
| `list_lab_observations`     | GET  | Get laboratory results by code           |
| `list_vital_signs`          | GET  | Get vital sign observations              |
| `record_vital_observation`  | POST | Create a new vital sign observation      |
| `list_medication_requests`  | GET  | Get medication orders                    |
| `create_medication_request` | POST | Create a new medication order            |
| `list_patient_procedures`   | GET  | Get completed procedures                 |
| `create_service_request`    | POST | Create lab/imaging/consult orders        |

---

## Environment Variables

| Variable                   | Default                              | Description                                         |
| -------------------------- | ------------------------------------ | --------------------------------------------------- |
| `OPENAI_API_KEY`           | (required)                           | OpenAI API key for the purple agent                 |
| `MEDAGENT_LLM_MODEL`       | `openai/gpt-4o-mini`                 | LLM model to use (litellm format)                   |
| `MEDAGENT_LLM_PROVIDER`    | (auto-detect)                        | LLM provider override (e.g., `anthropic`, `google`) |
| `MCP_SERVER_URL`           | `http://localhost:8002`              | MCP server URL                                      |
| `MCP_FHIR_API_BASE`        | `http://localhost:8080/fhir/`        | FHIR server base URL                                |
| `MEDAGENTBENCH_TASKS_FILE` | `src/mcp/resources/tasks/tasks.json` | Path to tasks file                                  |
| `MEDAGENT_OUTPUT_DIR`      | `outputs/medagentbench`              | Output directory for results                        |
| `HOST`                     | `0.0.0.0`                            | Host for agent servers                              |
| `AGENT_PORT`               | `9009`                               | Port for green agent server                         |

---

## Example Output

```
================================================================================
                    MedAgentBench Green Agent
================================================================================
Host: 0.0.0.0
Port: 9009
================================================================================
Running evaluation mode with arguments: batch
Command: uv run python main.py batch
================================================================================

[Status: working]
Starting MedAgentBench assessment.

[Status: working]
Evaluating 5 tasks in medagentbench domain

[Status: working]
Task task1_1: ✓ (2.3s)

[Status: working]
Task task2_1: ✓ (3.1s)

...

MedAgentBench Results
Domain: medagentbench
Tasks: 5
Pass Rate: 80.0% (4/5)
Time: 45.2s

Task Results:
  task1_1: ✓ (completed)
  task2_1: ✓ (completed)
  task3_1: ✓ (completed)
  task4_1: ✗ (completed)
  task5_1: ✓ (completed)
```

---

## Building Docker Images

```bash
# Build evaluator (green agent)
docker build --platform linux/amd64 \
  -f scenarios/medagentbench/Dockerfile.medagentbench-evaluator \
  -t ghcr.io/yourusername/medagentbench-evaluator:v1.0 .

# Build agent (purple agent)
docker build --platform linux/amd64 \
  -f scenarios/medagentbench/Dockerfile.medagentbench-agent \
  -t ghcr.io/yourusername/medagentbench-agent:v1.0 .
```

---

## Technologies

| Technology       | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| **A2A Protocol** | Agent-to-agent communication (a2a-sdk)              |
| **MCP**          | Model Context Protocol for tool management          |
| **FHIR**         | Fast Healthcare Interoperability Resources standard |
| **LiteLLM**      | Multi-provider LLM interface                        |
| **FastMCP**      | MCP server implementation                           |
| **Docker**       | FHIR server containerization                        |
| **uv**           | Python package management                           |

---

## License

MIT
