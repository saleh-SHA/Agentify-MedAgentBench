"""Green agent implementation for MedAgentBench - manages assessment and evaluation."""

import os
import uvicorn
import tomllib
import dotenv
import json
import time
import datetime
from src.green_agent.eval_resources import eval
from src.green_agent.eval_resources.utils import send_get_request, send_post_request
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

# MedAgentBench prompt template with MCP server instructions
MEDAGENT_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.




Instructions:
* Read the arguments carefully for each tool before deciding which are the suitable arguments to use.

* Some arguments are too general and some are more relevant to the question. You should use the more specific relevant arguments to the question.

* If you decide to invoke a GET function, you MUST put it in the format of
GET {mcp_server_url}/tools

* If you decide to invoke a POST function, you MUST put it in the format of 
POST {mcp_server_url}/tools/invoke
Payload format:
{{
  "tool_name": "tool_name",
  "arguments": {{
    "argument_name": "argument_value"
  }}
}}

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Task Context: {context}

Question: {question}"""

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


def update_overall_from_runs(output_dir: str) -> Dict[str, Any]:
    """Read all results from runs.jsonl and update overall.json.
    
    This function reads all completed task results from runs.jsonl,
    calculates overall metrics, and writes them to overall.json.
    Called after every task to keep overall.json up to date.
    
    Args:
        output_dir: Directory containing runs.jsonl and where overall.json will be written
        
    Returns:
        The calculated overall metrics dictionary, or empty dict if no results
    """
    runs_file = os.path.join(output_dir, "runs.jsonl")
    
    if not os.path.exists(runs_file):
        logger.warning(f"No runs.jsonl found at {runs_file}")
        return {}
    
    # Read all results from runs.jsonl
    results = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("output"):
                    results.append(record["output"])
    
    if not results:
        logger.warning("No results found in runs.jsonl")
        return {}
    
    # Calculate simple overall metrics (without re-evaluation)
    total = len(results)
    
    # Count by status
    status_counts: Dict[str, int] = {}
    correct_count = 0
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        # Check if marked as correct (status contains "Correct")
        if "Correct" in status:
            correct_count += 1
    
    # Calculate history length statistics
    history_lengths = [
        len(result.get("history", [])) 
        for result in results
    ]
    
    validation = {
        **{k: v / total for k, v in status_counts.items()},
        "average_history_length": sum(history_lengths) / total if total > 0 else 0,
        "max_history_length": max(history_lengths) if history_lengths else 0,
        "min_history_length": min(history_lengths) if history_lengths else 0,
    }
    
    success_rate = correct_count / total if total > 0 else 0
    
    overall = {
        "total": total,
        "validation": validation,
        "custom": {
            "success_rate": success_rate,
            "correct_count": correct_count,
            "total_evaluated": total,
            "raw_results": results,
        },
    }
    
    # Write to overall.json
    output_file = os.path.join(output_dir, "overall.json")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(overall, indent=4, ensure_ascii=False))
    
    logger.info(f"Overall metrics updated: {correct_count}/{total} correct ({success_rate:.2%})")
    return overall


async def ask_agent_to_solve(
    white_agent_url: str,
    task_data: Dict[str, Any],
    mcp_server_url: str,
    max_num_steps: int = 9
) -> Dict[str, Any]:
    """Execute a single MedAgentBench task with the white agent.

    This function coordinates a multi-turn conversation with the white agent to solve
    a medical task. The green agent receives GET/POST commands from the white agent,
    executes them against the FHIR/MCP server, and returns the results in the conversation.

    Args:
        white_agent_url: URL of the white agent to test
        task_data: Task configuration containing:
            - id: Task identifier
            - instruction: The medical question to answer
            - context: Additional context for the task
            - sol: Expected solution (for evaluation)
        mcp_server_url: URL of the MCP server providing FHIR tools (e.g., "http://0.0.0.0:8002")
        max_num_steps: Maximum number of interaction rounds (default: 9)

    Returns:
        Dictionary containing:
            - status: Task completion status
            - result: Agent's final answer (if completed)
            - history: Conversation history
            - correct: Whether the answer was correct (if completed)
    """

    # Build the initial task prompt with MCP server information
    task_prompt = MEDAGENT_PROMPT.format(
        mcp_server_url=mcp_server_url,
        context=task_data.get('context', 'N/A'),
        question=task_data['instruction']
    )

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
    # Parse agent responses and execute GET/POST requests, then inject results
    for round_num in range(max_num_steps):
        # Get agent's response
        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
        agent_response = text_parts[0].strip()
        
        # Clean up response (remove markdown code blocks if present - common with some models)
        agent_response = agent_response.replace('```tool_code', '').replace('```', '').strip()

        logger.info(f"White agent response (round {round_num + 1}):\n{agent_response[:200]}...")
        history.append({"role": "assistant", "content": agent_response})

        # Parse agent response and determine action
        if agent_response.startswith('GET'):
            # Handle GET request
            url = agent_response[3:].strip()
            # Add JSON format parameter if not already present
            if '?' in url:
                url = url + '&_format=json'
            else:
                url = url + '?_format=json'
            
            logger.info(f"Executing GET request: {url}")
            get_res = send_get_request(url)
            
            if "data" in get_res:
                feedback = f"Here is the response from the GET request:\n{get_res['data']}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
            else:
                feedback = f"Error in sending the GET request: {get_res['error']}"
            
            logger.info(f"GET response received, sending back to agent...")

        elif agent_response.startswith('POST'):
            # Handle POST request
            try:
                # Parse the URL from the first line and payload from subsequent lines
                lines = agent_response.split('\n')
                url = lines[0][4:].strip()  # Remove 'POST' prefix
                payload_str = '\n'.join(lines[1:])
                payload = json.loads(payload_str)
                
                logger.info(f"Executing POST request to: {url}")
                logger.info(f"Payload: {json.dumps(payload)[:200]}...")
                
                # Execute the POST request and return the result
                post_res = send_post_request(url, payload)
                
                if "data" in post_res:
                    feedback = f"Here is the response from the POST request:\n{post_res['data']}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
                else:
                    feedback = f"Error in sending the POST request: {post_res['error']}"
                
                logger.info(f"POST response received, sending back to agent...")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON payload in POST request: {e}")
                feedback = f"Invalid POST request: Could not parse JSON payload - {e}"
            except Exception as e:
                logger.error(f"Invalid POST request: {e}")
                feedback = f"Invalid POST request: {e}"

        elif agent_response.startswith('FINISH(') and agent_response.endswith(')'):
            # Agent has finished - extract answer
            answer = agent_response[len('FINISH('):-1]  # Remove "FINISH(" and ")"
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
                        if str(expected) in answer:
                            is_correct = True
                            break

            return {
                "status": "completed",
                "result": answer,
                "history": history,
                "correct": is_correct,
                "rounds": round_num + 1
            }

        else:
            # Invalid action - agent didn't follow the required format
            # The agent MUST respond with GET, POST, or FINISH only
            logger.warning(f"Agent returned invalid action: {agent_response[:100]}...")
            return {
                "status": "agent_invalid_action",
                "result": None,
                "history": history,
                "correct": False,
                "rounds": round_num + 1
            }

        # Send feedback back to the agent
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
        "status": "task_limit_reached",
        "result": None,
        "history": history,
        "correct": False,
        "rounds": max_num_steps
    }


class MedAgentGreenExecutor(AgentExecutor):
    """Agent executor for MedAgentBench green agent.

    This executor handles incoming task requests, sets up the MedAgentBench
    evaluation environment, coordinates with the white agent, and reports results.
    
    Overall metrics (overall.json) are automatically updated after every task
    by reading all results from runs.jsonl.
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
        max_rounds = medagent_config.get("max_rounds", 9)
        task_data = medagent_config["task_data"]
        
        # Optional: agent name and task name for output directory
        agent_name = medagent_config.get("agent_name", "default_agent")
        task_name = medagent_config.get("task_name", "medagentbench")
        output_dir = get_output_dir(agent_name, task_name)

        logger.info(f"Task ID: {task_data['id']}")
        logger.info(f"MCP Server URL: {mcp_server_url}")
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
        
        # Update overall metrics from all runs (reads from runs.jsonl)
        update_overall_from_runs(output_dir)

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
