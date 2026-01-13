"""Green agent implementation for MedAgentBench - manages assessment and evaluation."""

import os
import uvicorn
import tomllib
import dotenv
import json
import time
import datetime
from src.green_agent.eval_resources import eval
from typing import Dict, Any, List, Optional
from src.typings.output import TaskOutput
from src.typings.status import SampleStatus
from src.typings.general import ChatHistoryItem, SampleIndex
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a, logging_config

dotenv.load_dotenv()
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/")
OUTPUT_DIR = os.environ.get("MEDAGENT_OUTPUT_DIR", "outputs/medagentbench")

logger = logging_config.setup_logging("logs/green_agent.log", "green_agent")


def load_agent_card_toml(agent_name: str) -> dict:
    """Load agent card configuration from TOML file.

    Args:
        agent_name: Name of the agent configuration file (without .toml extension)

    Returns:
        Dictionary containing agent card configuration
    """
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def get_output_dir(agent_name: str, task_name: str) -> str:
    """Get the output directory path for a specific agent/task combination.
    
    Args:
        agent_name: Name of the agent being evaluated
        task_name: Name of the task
        
    Returns:
        Path to the output directory
    """
    return os.path.join(OUTPUT_DIR, agent_name, task_name)


def write_run_result(
    output_dir: str,
    index: SampleIndex,
    task_output: TaskOutput,
    error: Optional[str] = None,
    info: Optional[str] = None
) -> None:
    """Write a single task run result to the appropriate JSONL file.
    
    Args:
        output_dir: Directory to write the result to
        index: Task index
        task_output: The TaskOutput object containing results
        error: Error message if the task failed
        info: Additional info about the error
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_ms = int(time.time() * 1000)
    time_str = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    
    # Build the result record
    record = {
        "index": index,
        "error": error,
        "info": info,
        "output": task_output.model_dump() if task_output else None,
        "time": {"timestamp": timestamp_ms, "str": time_str},
    }
    
    # Write to appropriate file
    if error:
        target_file = os.path.join(output_dir, "error.jsonl")
    else:
        target_file = os.path.join(output_dir, "runs.jsonl")
    
    with open(target_file, "a+", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Result written to {target_file}")


def calculate_overall_metrics(
    results: List[TaskOutput],
    task_data_list: List[Dict[str, Any]],
    fhir_api_base: str
) -> Dict[str, Any]:
    """Calculate overall evaluation metrics from a list of task results.
    
    Args:
        results: List of TaskOutput objects
        task_data_list: List of task data dictionaries (for evaluation)
        fhir_api_base: Base URL for FHIR API
        
    Returns:
        Dictionary containing overall metrics including:
        - total: Total number of results
        - validation: Status distribution and history length statistics
        - custom: Task-specific metrics (success rate, raw results)
    """
    total = len(results)
    
    # Calculate status distribution
    status_counts: Dict[str, int] = {}
    for result in results:
        status = result.status or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Convert to percentages
    status_distribution = {k: v / total for k, v in status_counts.items()}
    
    # Calculate history length statistics
    history_lengths = [
        len(result.history) if result.history else 0 
        for result in results
    ]
    
    validation = {
        **status_distribution,
        "average_history_length": sum(history_lengths) / total if total > 0 else 0,
        "max_history_length": max(history_lengths) if history_lengths else 0,
        "min_history_length": min(history_lengths) if history_lengths else 0,
    }
    
    # Calculate task-specific metrics (success rate)
    correct_count = 0
    evaluated_results = []
    
    for i, result in enumerate(results):
        if result.result is not None and i < len(task_data_list):
            task_data = task_data_list[i]
            try:
                is_correct = eval.eval(task_data, result, fhir_api_base) is True
            except Exception as e:
                logger.error(f"Evaluation error for task {result.index}: {e}")
                is_correct = False
            
            if is_correct:
                correct_count += 1
                result.status = (result.status or "") + " Correct"
            else:
                result.status = (result.status or "") + " Incorrect"
        
        evaluated_results.append(result.model_dump())
    
    success_rate = correct_count / total if total > 0 else 0
    
    custom = {
        "success_rate": success_rate,
        "correct_count": correct_count,
        "total_evaluated": total,
        "raw_results": evaluated_results,
    }
    
    return {
        "total": total,
        "validation": validation,
        "custom": custom,
    }


def write_overall_metrics(
    output_dir: str,
    results: List[TaskOutput],
    task_data_list: List[Dict[str, Any]],
    fhir_api_base: str
) -> Dict[str, Any]:
    """Calculate and write overall metrics to overall.json.
    
    Args:
        output_dir: Directory to write the metrics to
        results: List of TaskOutput objects
        task_data_list: List of task data dictionaries
        fhir_api_base: Base URL for FHIR API
        
    Returns:
        The calculated overall metrics dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    overall = calculate_overall_metrics(results, task_data_list, fhir_api_base)
    
    output_file = os.path.join(output_dir, "overall.json")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(overall, indent=4, ensure_ascii=False))
    
    logger.info(f"Overall metrics written to {output_file}")
    return overall


async def ask_agent_to_solve(
    white_agent_url: str,
    task_data: Dict[str, Any],
    mcp_server_url: str,
    max_num_steps: int = 8
) -> Dict[str, Any]:
    """Execute a single MedAgentBench task with the white agent.

    This function coordinates a multi-turn conversation with the white agent to solve
    a medical task. The white agent will discover and use tools from the MCP server
    to interact with the FHIR server and answer medical questions.

    Args:
        white_agent_url: URL of the white agent to test
        task_data: Task configuration containing:
            - id: Task identifier
            - instruction: The medical question to answer
            - context: Additional context for the task
            - sol: Expected solution (for evaluation)
        mcp_server_url: URL of the MCP server providing FHIR tools (e.g., "http://0.0.0.0:8002")
        max_num_steps: Maximum number of interaction rounds (default: 8)

    Returns:
        Dictionary containing:
            - status: Task completion status
            - result: Agent's final answer (if completed)
            - history: Conversation history
            - correct: Whether the answer was correct (if completed)
    """

    # Build the initial task prompt with MCP server information
    task_prompt = f"""You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Instructions:
* You have access to tools for querying and updating a FHIR (Fast Healthcare Interoperability Resources) server.
* First, discover available tools by querying: GET {mcp_server_url}/tools
* Use tools by calling the MCP server's tool invocation endpoint: POST {mcp_server_url}/tools/invoke


MCP Server URL: {mcp_server_url}

Task Context: {task_data.get('context', 'N/A')}

Question: {task_data['instruction']}
"""

    context_id = None
    history = []

    logger.info(f"Starting MedAgentBench task {task_data['id']}...")

    # Send initial task prompt with MCP server URL
    logger.info(f"Sending task to white agent...")
    logger.info(f"MCP Server URL: {mcp_server_url}")
    logger.info(f"Task: {task_data['instruction'][:100]}...")

    white_agent_response = await my_a2a.send_message(
        white_agent_url, task_prompt, context_id=context_id
    )

    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(res_result, Message)

    if context_id is None:
        context_id = res_result.context_id

    history.append({"role": "user", "content": task_prompt})

    # Multi-turn interaction loop
    # The white agent will interact with the MCP server directly
    for round_num in range(max_num_steps):
        # Get agent's response
        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
        agent_response = text_parts[0].strip()

        logger.info(f"White agent response (round {round_num + 1}):\n{agent_response[:200]}...")
        history.append({"role": "assistant", "content": agent_response})

        # Check if agent has finished
        if agent_response.startswith('FINISH(') and agent_response.endswith(')'):
            # Agent has finished - extract answer
            answer = agent_response[7:-1]  # Remove "FINISH(" and ")"
            logger.info(f"Task completed with answer: {answer}")

            # Check if answer is correct
            # Use flexible matching: check if any expected solution appears in the answer
            expected_solutions = task_data.get('sol', [])
            is_correct = False
            if expected_solutions:
                # Try exact match first
                if answer in expected_solutions:
                    is_correct = True
                else:
                    # Try substring match (check if expected answer is contained in agent's response)
                    for expected in expected_solutions:
                        if expected in answer:
                            is_correct = True
                            break

            return {
                "status": "completed",
                "result": answer,
                "history": history,
                "correct": is_correct,
                "rounds": round_num + 1
            }

        # If not finished, the white agent is continuing to work
        # Send acknowledgment to allow it to continue
        logger.info(f"White agent is continuing work...")

        feedback = "Continue with your analysis and tool usage."

        white_agent_response = await my_a2a.send_message(
            white_agent_url, feedback, context_id=context_id
        )
        res_root = white_agent_response.root
        assert isinstance(res_root, SendMessageSuccessResponse)
        res_result = res_root.result
        assert isinstance(res_result, Message)

        history.append({"role": "user", "content": feedback})

    # Max rounds reached
    logger.info(f"Maximum rounds ({max_num_steps}) reached without completion")
    return {
        "status": "max_rounds_reached",
        "result": None,
        "history": history,
        "correct": False,
        "rounds": max_num_steps
    }


class MedAgentGreenExecutor(AgentExecutor):
    """Agent executor for MedAgentBench green agent.

    This executor handles incoming task requests, sets up the MedAgentBench
    evaluation environment, coordinates with the white agent, and reports results.
    """

    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a MedAgentBench evaluation task.

        Args:
            context: Request context containing the task configuration
            event_queue: Queue for sending response events
        """
        # Parse the task configuration
        logger.info("Received a MedAgentBench task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        white_agent_url = tags["white_agent_url"]
        medagent_config_str = tags["medagent_config"]
        medagent_config = json.loads(medagent_config_str)

        # Extract configuration
        logger.info("Setting up MedAgentBench environment...")
        mcp_server_url = medagent_config.get("mcp_server_url", "http://0.0.0.0:8002")
        max_rounds = medagent_config.get("max_rounds", 8)
        task_data = medagent_config["task_data"]
        
        # Optional: agent name and task name for output directory
        agent_name = medagent_config.get("agent_name", "default_agent")
        task_name = medagent_config.get("task_name", "medagentbench")
        output_dir = get_output_dir(agent_name, task_name)

        logger.info(f"Task ID: {task_data['id']}")
        logger.info(f"MCP Server: {mcp_server_url}")
        logger.info(f"Max rounds: {max_rounds}")
        logger.info(f"Output directory: {output_dir}")

        # Run the evaluation
        logger.info("Starting evaluation...")
        timestamp_started = time.time()
        error_msg = None
        error_info = None

        try:
            result = await ask_agent_to_solve(
                white_agent_url=white_agent_url,
                task_data=task_data,
                mcp_server_url=mcp_server_url,
                max_num_steps=max_rounds
            )
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            error_msg = "EXECUTION_ERROR"
            error_info = str(e)
            result = {
                "status": "error",
                "result": None,
                "history": [],
                "correct": False,
                "rounds": 0
            }

        # Transform history to match ChatHistoryItem schema
        # Convert "assistant" role to "agent" to match the evaluation script's format (refsol.py)
        history = result.get("history", [])
        transformed_history: List[ChatHistoryItem] = []
        for item in history:
            role = item.get("role", "user")
            if role == "assistant":
                role = "agent"
            # Create proper ChatHistoryItem objects
            transformed_history.append(ChatHistoryItem(
                role=role,
                content=item.get("content", "")
            ))

        # Create TaskOutput object with proper ChatHistoryItem objects
        task_output = TaskOutput(
            index=task_data["id"],
            status=result["status"],
            result=result.get("result"),
            history=transformed_history,
            rounds=result.get("rounds"),
        )
        
        # Evaluate using the TaskOutput object (not the raw dict)
        # This ensures the eval functions can properly access history items with .role and .content attributes
        try:
            eval_result = eval.eval(task_data, task_output, FHIR_API_BASE)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            eval_result = False

        # Determine success (Main metric for evaluation in MedAgentBench)
        is_correct = eval_result is True
        success = task_output.status == "completed" and is_correct
        result_emoji = "✅" if success else "❌"

        # Calculate comprehensive metrics
        time_used = time.time() - timestamp_started
        history_length = len(transformed_history)
        
        metrics = {
            "time_used": time_used,
            "status": task_output.status,
            "rounds_used": task_output.rounds,
            "correct": is_correct,
            "history_length": history_length,
            "task_id": task_data["id"],
            "timestamp": int(time.time() * 1000),
        }

        # Update task_output status with correctness suffix (matching MegAgentBench format)
        if is_correct:
            task_output.status = (task_output.status or "") + " Correct"
        else:
            task_output.status = (task_output.status or "") + " Incorrect"

        # Write result to runs.jsonl or error.jsonl
        write_run_result(
            output_dir=output_dir,
            index=task_data["id"],
            task_output=task_output,
            error=error_msg,
            info=error_info
        )

        # Prepare detailed report
        report = f"""MedAgentBench Evaluation Complete {result_emoji}

Task ID: {task_data['id']}
Question: {task_data['instruction'][:200]}{'...' if len(task_data['instruction']) > 200 else ''}
Expected Answer: {task_data.get('sol', 'N/A')}
Agent Answer: {result.get('result', 'N/A')}

Status: {task_output.status}
Correct: {is_correct}
Rounds Used: {task_output.rounds}
History Length: {history_length}
Time: {time_used:.2f}s

Output Directory: {output_dir}

Metrics:
{json.dumps(metrics, indent=2)}
"""

        logger.info("Evaluation complete.")
        logger.info(report)
        
        # Print to console for visibility (matching MegAgentBench behavior)
        print(f"\n{'='*60}")
        print(f"Task {task_data['id']}: {result_emoji} {'Correct' if is_correct else 'Incorrect'}")
        print(f"Status: {task_output.status}")
        print(f"Time: {time_used:.2f}s | Rounds: {task_output.rounds}")
        print(f"{'='*60}\n", flush=True)

        await event_queue.enqueue_event(
            new_agent_text_message(report)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current task execution.

        Args:
            context: Request context
            event_queue: Event queue
        """
        raise NotImplementedError("Task cancellation not implemented")


def start_medagent_green(
    agent_name: str = "medagent_green_agent",
    host: str = "localhost",
    port: int = 9001
) -> None:
    """Start the MedAgentBench green agent server.

    Args:
        agent_name: Name of the agent configuration file (default: "medagent_green_agent")
        host: Host to bind the server to (default: "localhost")
        port: Port to bind the server to (default: 9001)
    """
    logger.info("Starting MedAgentBench green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # Complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=MedAgentGreenExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=host, port=port)
