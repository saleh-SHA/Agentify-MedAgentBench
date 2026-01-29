# Agentify-MedAgentBench

This project extends [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench), a benchmark for evaluating LLM-based medical agents on clinical reasoning tasks using FHIR (Fast Healthcare Interoperability Resources). It adopts two key protocols: the Agent-to-Agent (A2A) protocol for standardized inter-agent communication, and the Model Context Protocol (MCP) for dynamic tool discovery. Together, these enable modular, interoperable evaluation of medical AI agents.

## Table of Contents

- [Overview](#overview)
- [Extensions to MedAgentBench](#extensions-to-medagentbench)
- [What Does This Benchmark Evaluate?](#what-does-this-benchmark-evaluate)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
  - [A2A Protocol Communication](#a2a-protocol-communication)
- [Quick Start Guide](#quick-start-guide)
- [Configuration](#configuration)
- [Benchmark Tasks](#benchmark-tasks)
- [Understanding Results](#understanding-results)
- [Development](#development)

---

## Overview

Agentify-MedAgentBench is an evaluation framework designed to test whether AI agents can perform real clinical tasks in a simulated electronic health record (EHR) environment. Unlike traditional medical AI benchmarks that focus on question-answering, this benchmark challenges agents to actually _do things_: query patient records, interpret lab values, and create clinical orders when appropriate.

The framework builds upon the original [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) benchmark developed by Stanford ML Group, which provides 300 clinically-derived tasks across 10 categories written by physicians, along with realistic patient profiles in a FHIR-compliant environment.

**What makes this version different?** This "agentified" implementation introduces two key architectural improvements:

1. **A2A Protocol Integration**: Instead of tightly-coupled components, the evaluator and agent communicate through the standardized Agent-to-Agent protocol. This means you can swap in different agent implementations without changing the evaluation infrastructure.

2. **MCP Tool Discovery**: Rather than hardcoding available tools, agents dynamically discover what actions they can take through Model Context Protocol. This mirrors how real-world agents would interact with unfamiliar systems.

---

## Extensions to MedAgentBench

Beyond the A2A and MCP protocol integrations, this implementation introduces significant enhancements to the evaluation framework. The original MedAgentBench used a pass/fail system; our version transforms it into a **comprehensive evaluation framework** with diagnostic capabilities, failure taxonomy, refined task prompts, and new tools with validated schema.

### Structured Failure Tracking

Instead of Pass/Fail metric, every failure now includes:

- **Primary failure category**: For aggregate statistics across evaluation runs
- **Detailed failure reasons**: For task-level diagnostics and debugging

This two-level failure classification enables:

- Aggregate failure analysis across runs
- Identification of systematic agent weaknesses (Is the task inherently difficult? Is the agent not following the specified format?)
- Better debugging of why specific tasks fail

### Comprehensive Failure Taxonomy

The evaluator categorizes failures into distinct types:

| Category                   | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `answer_mismatch`          | Returned value doesn't match expected answer    |
| `readonly_violation`       | Agent made POST request on read-only task       |
| `wrong_post_count`         | Incorrect number of POST requests               |
| `payload_validation_error` | POST payload has incorrect fields or values     |
| `max_rounds_reached`       | Agent didn't finish within iteration limit      |
| `invalid_finish_format`    | Response not in required `FINISH([...])` format |

### Robust POST Request Extraction

- **Original method**: Fragile string parsing of agent responses
- **New method**: Structured data extraction from MCP server responses returned via DataPart message (A2A)

The MCP server now returns FHIR operation metadata in a structured format, making POST request validation more reliable.

### Comprehensive Payload Validation

A single evaluation run reveals **all issues** with a payload, not just the first one. For example, if a MedicationRequest has both an incorrect dose and wrong route, both failures are reported. This is much more useful for debugging and agent improvement.

### Flexible Answer Comparison

- **Original method**: Strict exact match only
- **New method**: Handles common variations

Agents that return `["191 mg/dL"]` instead of `[191]` are now correctly evaluated as passing, reducing false failures while maintaining evaluation integrity.

### Comprehensive Metrics

The evaluation output includes:

- **Failure breakdown percentages**: Distribution of failure types across failed tasks
- **Round statistics**: Min, max, and average tool-calling rounds per task
- **Tool call counts**: Average number of tool invocations per task

These metrics help identify whether agents are efficient in their reasoning or taking unnecessary steps.

### MCP Server Integration

- **Original method**: Tasks and prompts hardcoded or file-based
- **New method**: Domain-specific MCP server for healthcare tasks, prompts, and tools

Benefits of the MCP-based approach:

- **Schema-strict tools**: Pydantic-typed tool definitions with validation
- **Single source of truth**: Prompts, tasks, and tools served from one location
- **Easy updates**: Modify tasks or prompts without touching evaluator code
- **Standardized protocol**: Agents discover tools dynamically via MCP

### Agentified Evaluation Benchmark

The evaluation follows a true agent-to-agent architecture:

- **Assessor (Green Agent)** and **Assessee (Purple Agent)** communicate via native A2A protocol
- The Assessor provides the Assessee with task, prompt, and MCP configuration in a structured way
- This separation enables plug-and-play agent evaluation—swap in any A2A-compliant assessee agent

---

## What Does This Benchmark Evaluate?

The benchmark tests an AI agent's ability to perform clinical reasoning tasks that a medical professional might encounter when working with an EHR system. These tasks fall into three categories:

### Read-Only Tasks

The agent must query patient records and return specific information. For example:

- _"What is the MRN of the patient named Peter Stafford with DOB 1932-12-29?"_
- _"Calculate the average glucose level for patient S1234567 over the past 24 hours."_

These tasks test whether the agent can correctly use search tools, interpret FHIR resources, and extract the relevant data.

### Action Tasks

The agent must create new clinical records. For example:

- _"Record a blood pressure of 118/77 mmHg for patient S6534835."_
- _"Order an orthopedic consultation for patient S2703270 with an ACL tear."_

These tasks test whether the agent can construct valid FHIR resources with the correct codes, values, and references.

### Conditional Action Tasks

The agent must first assess a clinical situation, then decide whether action is needed. For example:

- _"Check the patient's magnesium level. If it's below 1.9 mg/dL, order IV magnesium replacement with appropriate dosing."_
- _"Review the patient's most recent A1C. If it's more than a year old, order a new A1C test."_

These are the most challenging tasks because they require clinical reasoning: the agent must understand thresholds, calculate appropriate doses, and avoid unnecessary interventions when lab values are normal.

### What Gets Measured

For each task, the framework evaluates:

- **Correctness**: Did the agent arrive at the right answer or take the right action?
- **Appropriateness**: Did the agent avoid actions when they weren't needed (e.g., not ordering medication for normal lab values)?
- **Payload Validity**: For POST operations, were all required FHIR fields present and correct?
- **Efficiency**: How many tool-calling rounds did the agent need?

---

## How It Works

When you run the benchmark, here's what happens behind the scenes:

### Step 1: Infrastructure Startup

The scenario runner reads the configuration file and starts three services:

- A **FHIR server** (Docker container) containing synthetic patient data
- An **MCP server** that exposes FHIR operations as discoverable tools
- The **agent under test (Purple Agent)** that will attempt to complete the clinical tasks
- The **assessor (Green Agent)** that orchestrates the evaluation and validates results of the purple agent

### Step 2: Evaluation Begins

The evaluator (called the "Green Agent") receives a list of tasks to run. For each task, it:

1. **Retrieves** the prompt and tasks definitions from the MCP server
2. **Sends the prompt** to the agent via A2A protocol, along with configuration telling the agent where to find the MCP server
3. **Waits for a response** in the format `FINISH([answer1, answer2, ...])`

### Step 3: Agent Execution

When the agent (called the "Purple Agent") receives a task, it:

1. **Connects to the MCP server** and discovers available tools (search_patients, create_medication_request, etc.)
2. **Enters a tool-calling loop**: it sends the prompt to an LLM, which may request tool calls
3. **Executes tool calls** through the MCP server, which queries or mockingly modifies the FHIR server
4. **Iterates** until the LLM produces a final answer or hits the maximum iteration limit
5. **Returns the answer** along with metadata about what tools were called

### Step 4: Validation

The evaluator receives the agent's response and validates it against the expected solution. Depending on the task type, this might involve:

- Comparing the returned value to a reference answer
- Checking that the correct FHIR endpoint was called
- Validating that POST payloads contain all required fields with correct values
- Validating the number of POST requests made
- Ensuring read-only tasks didn't trigger any write operations

### Step 5: Results

After all tasks complete, the framework writes detailed results showing pass/fail status, failure reasons, and aggregate metrics.

---

## Architecture

The system is organized into four layers, each with a specific responsibility:

<div align="center">

```
       ┌─────────────────────────────────────────┐
       │         ORCHESTRATION LAYER             │
       ├─────────────────────────────────────────┤
       │  ┌─────────────────────────────────┐    │
       │  │  Scenario Runner (agentbeats)   │    │
       │  └───────────────┬─────────────────┘    │
       │                  ▼                      │
       │  ┌─────────────────────────────────┐    │
       │  │  Client CLI (sends EvalRequest) │    │
       │  └───────────────┬─────────────────┘    │
       └──────────────────┼──────────────────────┘
                          │ A2A
                          ▼
       ┌─────────────────────────────────────────┐
       │           A2A AGENT LAYER               │
       ├─────────────────────────────────────────┤
       │  ┌─────────────────────────────────┐    │
       │  │   EVALUATOR (Green Agent)       │    │
       │  │   :9009                          │    │
       │  │   • Orchestrates tasks          │    │
       │  │   • Validates responses         │    │
       │  └───────────────┬─────────────────┘    │
       │                  │ A2A                  │
       │                  ▼                      │
       │  ┌─────────────────────────────────┐    │
       │  │   AGENT (Purple Agent)          │    │
       │  │   :9019+                         │    │
       │  │   • LLM + tool-calling loop     │    │
       │  │   • Returns FINISH([...])       │    │
       │  └───────────────┬─────────────────┘    │
       └──────────────────┼──────────────────────┘
                          │ MCP
                          ▼
       ┌─────────────────────────────────────────┐
       │          MCP SERVER LAYER               │
       ├─────────────────────────────────────────┤
       │  ┌─────────────────────────────────┐    │
       │  │   FastMCP Server :8002          │    │
       │  │   • FHIR tools (search, create) │    │
       │  │   • Resources (tasks, prompts)  │    │
       │  └───────────────┬─────────────────┘    │
       └──────────────────┼──────────────────────┘
                          │ FHIR REST
                          ▼
       ┌─────────────────────────────────────────┐
       │            DATA LAYER                   │
       ├─────────────────────────────────────────┤
       │  ┌─────────────────────────────────┐    │
       │  │   FHIR Server (Docker) :8080    │    │
       │  │   • Patient, Observation        │    │
       │  │   • MedicationRequest, etc.     │    │
       │  └─────────────────────────────────┘    │
       └─────────────────────────────────────────┘
```

</div>

### Layer Descriptions

**Orchestration Layer**: The entry point. The scenario runner parses configuration, starts all services as subprocesses, waits for health checks, and then triggers the evaluation through the client CLI.

**A2A Agent Layer**: Where the evaluation logic lives. The evaluator orchestrates tasks and validates results. The agent under test receives prompts and produces answers. Both communicate using the standardized A2A protocol, which means you could replace either component with a different implementation.

**MCP Server Layer**: The tool provider. Instead of the agent having hardcoded knowledge of available operations, it discovers tools dynamically through MCP. This layer also hosts benchmark resources like task definitions and prompt templates.

**Data Layer**: The simulated EHR. A FHIR server running in Docker contains synthetic patient records that the agent queries and modifies during evaluation.

### A2A Protocol Communication

The Agent-to-Agent (A2A) protocol enables standardized communication between the Evaluator (Green Agent) and the Medical AI Agent (Purple Agent). Here's how they interact:

#### Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GREEN AGENT (Evaluator) :9009                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
      1. GET /.well-known/agent.json│  ◄─── Agent Discovery
                                    ▼
                              AgentCard
                    (capabilities, skills, endpoint)
                                    │
      2. POST / (A2A Message)       │  ◄─── Task Request
         ├─ TextPart: prompt +      ▼
         │  instructions
         └─ DataPart: config
            (mcp_server_url, etc.)
                                    │
      3. TaskStatusUpdateEvent      │  ◄─── Progress Updates
         (state: working)           ▼
                                    │
      4. Task + Artifacts           │  ◄─── Final Response
         ├─ TextPart: FINISH([...]) ▼
         └─ DataPart: metadata
            (tool call history, etc.)
                                    │
┌───────────────────────────────────┴─────────────────────────────────────────┐
│                     PURPLE AGENT (Medical AI) :9019+                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Protocol Concepts

**Agent Discovery**: Before communication, the Evaluator fetches the agent's `AgentCard` from `/.well-known/agent.json`. This describes the agent's capabilities, supported input/output modes, and available skills.

**Multi-Part Messages**: A2A messages contain multiple parts that separate human-readable content from machine-readable data:

- **TextPart**: Contains the task prompt (system instructions + clinical question)
- **DataPart**: Contains structured configuration (MCP server URL, max iterations)

**Task Lifecycle**: Each task progresses through states: `submitted` → `working` → `completed/failed`. The agent sends status updates during processing and attaches artifacts containing the final response.

**Response Structure**: The agent returns both the answer and evaluation metadata:

- **TextPart**: The answer in `FINISH([...])` format
- **DataPart**: Metadata including tool call history, FHIR operations performed, and iteration count

This separation allows the Evaluator to assess not just the final answer, but also how the agent arrived at it—enabling detailed failure analysis, detailed evaluation and performance metrics.

---

## Quick Start Guide

This section walks you through running your first evaluation end-to-end.

### Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- Docker installed and running
- An OpenAI API key (or key for another LLM provider)
- The [uv](https://github.com/astral-sh/uv) package manager

### Step 1: Clone and Install

```bash
git clone https://github.com/your-repo/Agentify-MedAgentBench.git
cd Agentify-MedAgentBench
uv sync
```

### Step 2: Configure Your LLM

Create a `.env` file in the project root with your API key:

```bash
OPENAI_API_KEY=your-api-key-here
MEDAGENT_LLM_MODEL=openai/gpt-4o-mini
```

The framework uses LiteLLM, so you can use other providers by changing the model string (e.g., `anthropic/claude-3.5-sonnet`).

### Step 3: Run the Evaluation

The `run.sh` script handles everything automatically - starting the FHIR server, MCP server, and running the evaluation:

```bash
./run.sh
```

This will:

1. Start the FHIR server (Docker container with synthetic patient data) on port 8080
2. Start the MCP server (Docker container with FHIR tools) on port 8002
3. Run the MedAgentBench evaluation with the `--show-logs` flag

> **Note:** The FHIR server (HAPI FHIR) takes couple of minutes to initialize on first startup. Please be patient during this process. Subsequent runs will be faster if the container is already running.

The `--show-logs` flag displays agent output so you can watch the evaluation progress. You'll see the agent receiving tasks, making tool calls, and producing answers.

### Step 4: Review Results

When the evaluation completes, results are written to:

```
outputs/medagentbench/default_agent/medagentbench/
├── runs.jsonl      # Detailed per-task results
├── error.jsonl     # Any tasks that errored
└── overall.json    # Summary metrics
```

Open `overall.json` to see your agent's pass rate and failure breakdown.

### Additional Commands

```bash
# Start servers only (without running evaluation)
./run.sh --servers-only

# Stop all running containers
./run.sh --cleanup

# Start servers manually (alternative to run.sh)
./fhir_mcp_launcher.sh

# Stop servers manually
./fhir_mcp_launcher.sh --cleanup
```

### Manual Server Setup (Alternative)

If you prefer to start servers manually without Docker for the MCP server:

```bash
# Terminal 1: Start FHIR server
docker run -d -p 8080:8080 --name medagentbench-fhir jyxsu6/medagentbench:latest

# Terminal 2: Start MCP server (local Python)
uv run python -m src.mcp.server

# Terminal 3: Run evaluation
uv run agentbeats-run scenarios/medagentbench/scenario.toml --show-logs
```

---

## Configuration

The evaluation is configured through `scenarios/medagentbench/scenario.toml`. This file defines the agent endpoints, server configurations, and evaluation parameters.

| Parameter        | Default                         | Description                                                          |
| ---------------- | ------------------------------- | -------------------------------------------------------------------- |
| `num_tasks`      | `30`                            | Number of tasks to run. Set to `null` or remove to run all 300 tasks |
| `task_ids`       | (optional)                      | Specific task IDs to run. Overrides `num_tasks` if provided          |
| `max_rounds`     | `8`                             | Maximum tool-calling iterations per task before timeout              |
| `domain`         | `"medagentbench"`               | Domain identifier for the evaluation                                 |
| `mcp_server_url` | `"http://localhost:8002"`       | URL of the MCP server providing FHIR tools                           |
| `fhir_api_base`  | `"http://localhost:8080/fhir/"` | Base URL of the FHIR server                                          |
| `num_workers`    | `10`                            | Number of parallel agent workers for concurrent task execution       |

### Selecting Tasks

By default, the benchmark runs a subset of tasks based on `num_tasks`. To run specific tasks, use `task_ids`:

```toml
[config]
# Run specific tasks by ID
task_ids = ["task1_1", "task1_2", "task3_1", "task5_1"]

# Or run tasks from a specific category
task_ids = ["task5_1", "task5_2", "task5_3", "task5_4", "task5_5", "task5_6", "task5_7", "task5_8", "task5_9", "task5_10"]
```

Task IDs follow the pattern `task{category}_{number}`. For example, `task5_3` is the third instance of task category 5 (magnesium replacement).

### Adjusting Iteration Limits

The `max_rounds` setting controls how many tool-calling iterations the agent can perform before timing out:

```toml
[config]
max_rounds = 8
```

Some complex tasks may require more iterations. If you see many `max_rounds_reached` failures, try increasing this value while maintaining this control variable for all the assessed agents for fair evaluation.

### Parallel Execution

The `num_workers` setting enables parallel task execution:

```toml
[config]
# Sequential execution (default)
num_workers = 1

# Moderate parallelism (5-10 workers)
num_workers = 10

# High parallelism (requires more resources)
num_workers = 20
```

Each worker runs as a separate process on a different port (9019, 9020, etc.). Make sure the ports are available and accessible.

### Environment Variables

| Variable             | Default                       | Description                          |
| -------------------- | ----------------------------- | ------------------------------------ |
| `OPENAI_API_KEY`     | -                             | API key for your LLM provider        |
| `MEDAGENT_LLM_MODEL` | `openai/gpt-4o-mini`          | LiteLLM model identifier             |
| `MCP_FHIR_API_BASE`  | `http://localhost:8080/fhir/` | FHIR server URL (used by MCP server) |
| `MCP_SERVER_URL`     | `http://localhost:8002`       | MCP server URL (used by agent)       |

---

## Benchmark Tasks

The benchmark includes 330 tasks across 11 categories. Each category tests different clinical reasoning capabilities:

| Category    | Type           | What It Tests                                      |
| ----------- | -------------- | -------------------------------------------------- |
| **Task 1**  | Read-only      | Basic patient lookup by name and date of birth     |
| **Task 2**  | Read-only      | Age calculation from patient demographics          |
| **Task 3**  | POST           | Creating vital signs observations (blood pressure) |
| **Task 4**  | Read-only      | Retrieving recent lab values (magnesium)           |
| **Task 5**  | Conditional    | Clinical decision-making for magnesium replacement |
| **Task 6**  | Read-only      | Calculating aggregate values (average glucose)     |
| **Task 7**  | Read-only      | Finding most recent lab values (glucose)           |
| **Task 8**  | POST           | Creating specialist consultation requests          |
| **Task 9**  | Conditional    | Multi-step intervention for potassium deficiency   |
| **Task 10** | Conditional    | Time-based decision-making for A1C testing         |
| **Task 11** | Trend Analysis | Blood pressure trend analysis for hypertension     |

### Example Task (Conditional)

Here's what a conditional task looks like:

```json
{
  "id": "task5_3",
  "instruction": "Check the last serum magnesium level within the past 24 hours for patient S1234567. If the magnesium level is low (below 1.9 mg/dL), order IV magnesium sulfate replacement based on the following protocol: Mg 1.5-1.9: 1g over 1h, Mg 1.0-1.5: 2g over 2h, Mg <1.0: 4g over 4h",
  "context": "Current time: 2023-11-13T10:15:00+00:00",
  "sol": [1.3],
  "eval_MRN": "S1234567"
}
```

For this task, the agent must:

1. Query the patient's magnesium observations
2. Find the most recent value within 24 hours
3. Determine it's 1.3 mg/dL (moderate deficiency)
4. Create a MedicationRequest for 2g IV magnesium over 2 hours
5. Return the magnesium value in the expected format

### Example Task (Trend Analysis)

Task 11 introduces trend analysis, requiring the agent to analyze patterns over time:

```json
{
  "id": "task11_9",
  "instruction": "Does patient S1733937 have a hypertension alert based on their blood pressure readings from the past 7 days (reference date: 2022-05-06T00:00:00+00:00)? A hypertension alert is true if >=50% of readings have systolic >= 140 OR diastolic >= 90.",
  "context": "Return FINISH([hypertension_alert, reading_count]) where hypertension_alert is true/false and reading_count is the number of BP readings found.",
  "eval_MRN": "S1733937",
  "eval_ref_date": "2022-05-06T00:00:00+00:00"
}
```

For this task, the agent must:

1. Discover and use the `analyze_blood_pressure_trend` tool
2. Pass the correct parameters: patient ID, reference date, and 7-day window
3. Analyze the tool's output for hypertension alert status
4. Return both the alert (true/false) and exact reading count

**Evaluation criteria:**

- No POST requests (read-only task)
- Tool called with correct `patient`, `reference_date`, and `days_back=7`
- Hypertension alert matches reference (computed from FHIR data)
- Reading count is an exact match

### Response Format

Agents must respond using the `FINISH([...])` format:

```
FINISH(["S6534835"])           # Single value (e.g., patient MRN)
FINISH([191.5])                # Numeric value (e.g., glucose level)
FINISH([])                     # No return value (action-only task)
```

---

## Understanding Results

After running an evaluation, the framework produces detailed output to help you understand how your agent performed.

### Summary Metrics (`overall.json`)

```json
{
  "domain": "medagentbench",
  "total_tasks": 10,
  "correct_count": 8,
  "pass_rate": 0.8,
  "failure_breakdown": {
    "answer_mismatch": 0.1,
    "payload_validation_error": 0.1
  },
  "avg_rounds": 2.5
}
```

The `pass_rate` is the primary metric. The `failure_breakdown` shows what types of errors occurred, which helps identify systematic issues with your agent.

### Per-Task Results (`runs.jsonl`)

Each line contains detailed information about one task:

```json
{
  "index": "task5_3",
  "output": {
    "correct": false,
    "result": [1.3],
    "expected": [1.3],
    "primary_failure": "payload_validation_error",
    "failure_details": ["wrong_dose_value", "wrong_rate_unit"]
  }
}
```

This tells you the agent found the correct magnesium value but made errors in the medication order payload.

### Common Failure Types

| Failure                    | What It Means                                              |
| -------------------------- | ---------------------------------------------------------- |
| `answer_mismatch`          | The returned value didn't match the expected answer        |
| `readonly_violation`       | The agent made a POST request on a read-only task          |
| `wrong_post_count`         | The agent made too many or too few POST requests           |
| `payload_validation_error` | A POST payload had incorrect fields or values              |
| `max_rounds_reached`       | The agent didn't finish within the iteration limit         |
| `invalid_finish_format`    | The response wasn't in the required `FINISH([...])` format |

---

## Development

### Project Structure

```
Agentify-MedAgentBench/
├── run.sh                         # Main entry point - starts servers and runs evaluation
├── fhir_mcp_launcher.sh           # Launches FHIR and MCP server containers
├── scenarios/medagentbench/
│   ├── scenario.toml              # Evaluation configuration
│   ├── agent/src/                 # The agent under test
│   │   └── agent.py               # LLM + tool-calling logic
│   └── evaluator/src/             # The evaluation harness
│       └── agent.py               # Task validation logic
├── src/
│   ├── agentbeats/
│   │   └── run_scenario.py        # Entry point
│   └── mcp/
│       ├── Dockerfile             # Docker image for MCP server
│       ├── server.py              # MCP tool server
│       └── resources/
│           ├── tasks/tasks.json   # Task definitions
│           └── prompts/           # System prompt templates
└── outputs/                       # Evaluation results
```

### Implementing a Custom Agent

To test a different agent implementation, modify `scenarios/medagentbench/agent/src/agent.py`. The key method is `run()`, which receives a message and must return a response in the expected format.

Your agent must:

1. Connect to the MCP server URL provided in the message's `DataPart`
2. Discover available tools via MCP
3. Use an LLM to reason about the task and decide which tools to call
4. Return the final answer as `FINISH([...])` along with metadata in a `DataPart`

### Adding New Tasks

1. Add the task definition to `src/mcp/resources/tasks/tasks.json`
2. Implement a validation function in `scenarios/medagentbench/evaluator/src/agent.py`
3. Register the function in the `TASK_EVALUATORS` dictionary

---

## License

MIT
