# MedAgentBench for AgentBeats

A medical AI agent evaluation framework compatible with the [AgentBeats](https://github.com/RDI-Foundation/agentbeats-tutorial) platform. This benchmark evaluates AI agents on clinical reasoning tasks using FHIR (Fast Healthcare Interoperability Resources) servers.

## Overview

MedAgentBench implements a green-agent/purple-agent architecture where:

- **Green Agent (Evaluator)**: Orchestrates evaluations, sends tasks to purple agents, and validates answers using reference solutions
- **Purple Agent (Participant)**: The AI agent being evaluated - uses FHIR tools via MCP server to answer medical questions

## Project Structure

```
scenarios/
└─ medagentbench/
   ├─ evaluator/src/           # green agent (evaluator)
   │  ├─ server.py             # A2A server entry point
   │  ├─ agent.py              # evaluation logic + task graders
   │  ├─ executor.py           # AgentExecutor implementation
   │  └─ messenger.py          # A2A client helper
   ├─ agent/src/               # purple agent (participant)
   │  ├─ server.py             # A2A server entry point
   │  ├─ agent.py              # LLM + MCP tool calling logic
   │  └─ executor.py           # AgentExecutor implementation
   ├─ Dockerfile.medagentbench-evaluator
   ├─ Dockerfile.medagentbench-agent
   └─ scenario.toml            # scenario configuration

src/
├─ agentbeats/                 # AgentBeats runner + A2A client helpers
├─ mcp/                        # MCP server with FHIR tools
│  ├─ server.py
│  └─ resources/tasks/tasks.json
└─ ...                         # original implementation (for reference)
```

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Environment Variables

```bash
cp sample.env .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start Required Services

MedAgentBench requires two external services to be running:

**Terminal 1 - FHIR Server (Docker):**

```bash
docker pull jyxsu6/medagentbench:latest
docker run -p 8080:8080 jyxsu6/medagentbench:latest
```

Wait for: `Started Application in XXX seconds`

**Terminal 2 - MCP Server:**

```bash
uv run python -m src.mcp.server
```

### 4. Run the Evaluation

**Terminal 3:**

```bash
uv run agentbeats-run scenarios/medagentbench/scenario.toml
```

This command will:

- Start the evaluator (green agent) and participant (purple agent) servers
- Construct an `assessment_request` message with participant endpoints and config
- Send the request to the green agent and print streamed responses

**Options:**

- `--show-logs`: Show agent stdout/stderr during assessment
- `--serve-only`: Start agents without running the assessment (useful for debugging)

## Configuration

Edit `scenarios/medagentbench/scenario.toml` to customize the evaluation:

```toml
[green_agent]
endpoint = "https://medagentbench.ddns.net:9009"
cmd = "python scenarios/medagentbench/evaluator/src/server.py --host 0.0.0.0 --port 9009"

[[participants]]
role = "agent"
endpoint = "https://medagentbench.ddns.net:9019"
cmd = "python scenarios/medagentbench/agent/src/server.py --host 0.0.0.0 --port 9019"

[config]
num_tasks = 5                                    # Number of tasks to run
max_rounds = 10                                  # Max tool-calling rounds per task
mcp_server_url = "https://medagentbench.ddns.net:8002"         # MCP server URL
fhir_api_base = "https://medagentbench.ddns.net:8080/fhir/"    # FHIR server URL
```

## Assessment Flow

1. **Assessment Request**: The green agent receives a JSON message:

   ```json
   {
     "participants": { "agent": "https://medagentbench.ddns.net:9019" },
     "config": {
       "num_tasks": 5,
       "max_rounds": 10,
       "mcp_server_url": "https://medagentbench.ddns.net:8002",
       "fhir_api_base": "https://medagentbench.ddns.net:8080/fhir/"
     }
   }
   ```

2. **Task Execution**: For each task, the green agent:

   - Sends the medical question to the purple agent
   - The purple agent uses FHIR tools via MCP to gather information
   - The purple agent returns `FINISH([answer1, answer2, ...])` format
   - The green agent validates using task-specific grading functions

3. **Results**: The green agent produces an artifact with:
   - Pass rate and correct count
   - Per-task results (correct/incorrect, time, status)

## Task Types

MedAgentBench includes 10 task types covering various clinical scenarios:

| Task   | Type        | Description                     |
| ------ | ----------- | ------------------------------- |
| task1  | Query       | Patient lookup by demographics  |
| task2  | Query       | Calculate patient age           |
| task3  | Write       | Record vital sign observation   |
| task4  | Query       | Recent lab value lookup (24h)   |
| task5  | Conditional | Check magnesium & order if low  |
| task6  | Query       | Average glucose (24h)           |
| task7  | Query       | Latest glucose value            |
| task8  | Write       | Create orthopedic consultation  |
| task9  | Conditional | Potassium replacement protocol  |
| task10 | Conditional | HbA1C check & order if >1yr old |

## FHIR Tools (via MCP Server)

The MCP server provides 9 FHIR tools:

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

## Environment Variables

| Variable                   | Default                                     | Description                         |
| -------------------------- | ------------------------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`           | (required)                                  | OpenAI API key for the purple agent |
| `MEDAGENT_LLM_MODEL`       | `openai/gpt-4o`                             | LLM model to use (litellm format)   |
| `MCP_SERVER_URL`           | `https://medagentbench.ddns.net:8002`       | MCP server URL                      |
| `MCP_FHIR_API_BASE`        | `https://medagentbench.ddns.net:8080/fhir/` | FHIR server base URL                |
| `MEDAGENTBENCH_TASKS_FILE` | `src/mcp/resources/tasks/tasks.json`        | Path to tasks file                  |

## Example Output

```
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

## Original Implementation

The repository also includes the original MedAgentBench implementation (in `src/`) which can be run using:

```bash
# Single task
uv run python main.py launch --task-index 0

# Batch evaluation
uv run python main.py batch --task-indices "0,1,2"
```

See the original CLI commands:

- `green`: Start green agent only
- `white`: Start white agent only
- `launch`: Run single task evaluation
- `batch`: Run batch evaluation

## Technologies

- **A2A Protocol**: Agent-to-agent communication (a2a-sdk)
- **MCP**: Model Context Protocol for tool management
- **FHIR**: Fast Healthcare Interoperability Resources standard
- **LiteLLM**: Multi-provider LLM interface
- **Docker**: FHIR server containerization

## License

MIT
