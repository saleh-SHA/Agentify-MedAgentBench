"""Green agent implementation for MedAgentBench - manages assessment and evaluation."""

import os
import uvicorn
import tomllib
import dotenv
import json
import time
import datetime
from src.green_agent.eval_resources import eval
from src.green_agent.eval_resources.eval import calculate_overall_metrics
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

print(f"*Host*: {os.environ.get('HOST')}, *port*: {os.environ.get('AGENT_PORT')}")

# MedAgentBench prompt template with MCP server instructions
# This prompt contains ALL instructions for the white agent (no system prompt in white agent)
MEDAGENT_PROMPT = """You are an expert medical AI assistant that uses FHIR functions to assist medical professionals. You are given a question and a set of available FHIR tools. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

You have access to FHIR tools via the MCP server at: {mcp_server_url}

Instructions:
1. Use the provided FHIR tools to query patient data, retrieve medical records, create orders, and perform other medical record operations as needed.

2. Read the arguments carefully for each tool before deciding which are the suitable arguments to use. Some arguments are too general and some are more relevant to the question.

3. Make as many tool calls as needed to gather the information required to answer the question.

4. When you have gathered all necessary information and have the final answer(s), you MUST respond with ONLY the finish format (make sure the list is JSON loadable):
FINISH([answer1, answer2, ...])

IMPORTANT: Your final response MUST be in the FINISH format with no other text. The list inside FINISH() must be valid JSON.

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


def sanitize_task_data(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a sanitized copy of task_data without evaluation fields.
    
    Removes 'sol' (solution) and 'eval_MRN' fields which should not be
    exposed to the agent being evaluated.
    
    Args:
        task_data: Original task data
        
    Returns:
        Sanitized task data safe to send to the white agent
    """
    # Fields that should not be sent to the white agent
    evaluation_fields = {'sol', 'eval_MRN'}
    return {k: v for k, v in task_data.items() if k not in evaluation_fields}


def extract_fhir_operations(response: str) -> tuple[str, List[Dict[str, Any]]]:
    """Extract FHIR operations metadata from white agent response.
    Tracks the POST operations, if any, that the white agent made to the FHIR server which will be used for evaluation.
    
    The white agent embeds FHIR POST operations in the response:
    <fhir_operations>
    [{"fhir_url": "...", "payload": {...}, "accepted": true}, ...]
    </fhir_operations>
    
    Args:
        response: Raw response from white agent
        
    Returns:
        Tuple of (clean_response, fhir_operations_list)
    """
    import re
    
    fhir_ops = []
    clean_response = response
    
    # Extract FHIR operations section
    pattern = r'<fhir_operations>\s*(.*?)\s*</fhir_operations>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        try:
            fhir_ops = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse FHIR operations metadata")
        
        # Remove the metadata section from the response
        clean_response = re.sub(pattern, '', response, flags=re.DOTALL).strip()
    
    return clean_response, fhir_ops


def add_fhir_posts_to_history(history: List[Dict[str, str]], fhir_ops: List[Dict[str, Any]]) -> None:
    """Add FHIR POST operations to history for evaluation.
    
    Adds entries in the format expected by refsol.py:
    - {"role": "agent", "content": "POST <url>\n<payload_json>"}
    - {"role": "user", "content": "POST request accepted"} (if accepted)
    
    Args:
        history: History list to modify (in place)
        fhir_ops: List of FHIR operations from white agent
    """
    for op in fhir_ops:
        fhir_url = op.get("fhir_url", "")
        payload = op.get("payload", {})
        accepted = op.get("accepted", False)
        
        # Add the POST request entry (format expected by refsol.py)
        post_content = f"POST {fhir_url}\n{json.dumps(payload)}"
        history.append({"role": "agent", "content": post_content})
        
        # Add acceptance response if the POST was successful
        if accepted:
            history.append({"role": "user", "content": "POST request accepted"})
        else:
            history.append({"role": "user", "content": "POST request failed"})


async def ask_agent_to_solve(
    white_agent_url: str,
    task_data: Dict[str, Any],
    mcp_server_url: str,
    max_num_steps: int = 9
) -> Dict[str, Any]:
    """Execute a single MedAgentBench task with the white agent.

    This function sends the task prompt to the white agent which handles tool discovery
    and invocation directly via the MCP server. The white agent performs multiple
    internal rounds of tool calling and returns the final answer in FINISH format.

    Args:
        white_agent_url: URL of the white agent to test
        task_data: Task configuration containing:
            - id: Task identifier
            - instruction: The medical question to answer
            - context: Additional context for the task
            - sol: Expected solution (for evaluation, NOT sent to agent)
            - eval_MRN: Evaluation MRN (for evaluation, NOT sent to agent)
        mcp_server_url: URL of the MCP server providing FHIR tools (e.g., "http://0.0.0.0:8002")
        max_num_steps: Maximum number of tool-calling rounds (passed to white agent context)

    Returns:
        Dictionary containing:
            - status: Task completion status
            - result: Agent's final answer (if completed)
            - history: Conversation history
            - correct: Whether the answer was correct (if completed)
    """
    # Sanitize task_data to remove evaluation fields before using in prompts
    safe_task_data = sanitize_task_data(task_data)

    # Build the task prompt with MCP server information
    task_prompt = MEDAGENT_PROMPT.format(
        mcp_server_url=mcp_server_url,
        context=safe_task_data.get('context', 'N/A'),
        question=safe_task_data['instruction']
    )

    history = []

    instruction_preview = task_data['instruction'][:100] + "..." if len(task_data['instruction']) > 100 else task_data['instruction']
    logger.info(f"Starting task {task_data['id']}: {instruction_preview}")

    # Send the task prompt to white agent - it will handle tool discovery and invocation internally
    white_agent_response = await my_a2a.send_message(
        white_agent_url, task_prompt, context_id=None
    )

    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(res_result, Message)

    history.append({"role": "user", "content": task_prompt})

    # Get agent's final response
    text_parts = get_text_parts(res_result.parts)
    assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
    raw_response = text_parts[0].strip()
    
    # Clean up response (remove markdown code blocks if present - common with some models)
    raw_response = raw_response.replace('```tool_code', '').replace('```', '').strip()
    
    # Extract FHIR operations metadata and clean response
    agent_response, fhir_ops = extract_fhir_operations(raw_response)
    
    # Add FHIR POST operations to history (for evaluation by refsol.py)
    if fhir_ops:
        logger.info(f"Extracted {len(fhir_ops)} FHIR POST operations for evaluation")
        add_fhir_posts_to_history(history, fhir_ops)

    response_preview = agent_response[:500] + "..." if len(agent_response) > 500 else agent_response
    logger.info(f"White agent response: {response_preview}")
    history.append({"role": "assistant", "content": agent_response})

    # Parse the FINISH format from the response
    if agent_response.startswith('FINISH(') and agent_response.endswith(')'):
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

        # Calculate rounds as number of agent responses in history
        rounds = sum(1 for h in history if h.get("role") in ("agent", "assistant"))
        return {
            "status": "completed",
            "result": answer,
            "history": history,
            "correct": is_correct,
            "rounds": rounds
        }
    
    # Check if FINISH is somewhere in the response (agent may have added extra text)
    finish_match = None
    if 'FINISH(' in agent_response and ')' in agent_response:
        start_idx = agent_response.find('FINISH(')
        # Find the matching closing parenthesis
        paren_count = 0
        for i, char in enumerate(agent_response[start_idx:]):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    finish_match = agent_response[start_idx:start_idx + i + 1]
                    break
    
    if finish_match:
        answer = finish_match[len('FINISH('):-1]
        logger.info(f"Task completed with extracted answer: {answer}")

        expected_solutions = task_data.get('sol', [])
        is_correct = False
        if expected_solutions:
            if answer in expected_solutions:
                is_correct = True
            else:
                for expected in expected_solutions:
                    if str(expected) in answer:
                        is_correct = True
                        break

        # Calculate rounds as number of agent responses in history
        rounds = sum(1 for h in history if h.get("role") in ("agent", "assistant"))
        return {
            "status": "completed",
            "result": answer,
            "history": history,
            "correct": is_correct,
            "rounds": rounds
        }

    # Agent didn't return FINISH format
    warning_preview = agent_response[:200] + "..." if len(agent_response) > 200 else agent_response
    logger.warning(f"Agent did not return FINISH format: {warning_preview}")
    # Calculate rounds as number of agent responses in history
    rounds = sum(1 for h in history if h.get("role") in ("agent", "assistant"))
    return {
        "status": "agent_invalid_action",
        "result": agent_response,
        "history": history,
        "correct": False,
        "rounds": rounds
    }


class MedAgentGreenExecutor(AgentExecutor):
    """Agent executor for MedAgentBench green agent.

    This executor handles incoming task requests, sets up the MedAgentBench
    evaluation environment, coordinates with the white agent, and reports results.
    
    The executor tracks all task results and writes overall metrics when
    finalize() is called after all tasks are processed.
    """

    def __init__(self):
        # Track all results and task data for overall metrics calculation
        self.all_results: List[TaskOutput] = []
        self.all_task_data: List[Dict[str, Any]] = []
        self.current_output_dir: Optional[str] = None
        self.current_fhir_api_base: str = FHIR_API_BASE

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
        mcp_server_url = medagent_config.get("mcp_server_url", "http://0.0.0.0:8002")
        max_rounds = medagent_config.get("max_rounds", 9)
        task_data = medagent_config["task_data"]
        
        # Optional: agent name and task name for output directory
        agent_name = medagent_config.get("agent_name", "default_agent")
        task_name = medagent_config.get("task_name", "medagentbench")
        output_dir = get_output_dir(agent_name, task_name)
        
        # Store output directory and fhir_api_base for overall metrics
        self.current_output_dir = output_dir
        self.current_fhir_api_base = medagent_config.get("fhir_api_base", FHIR_API_BASE)

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
        
        # Store results for overall metrics calculation (will be written when finalize() is called)
        self.all_results.append(task_output)
        self.all_task_data.append(task_data)

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

        logger.info(f"Task {task_data['id']} completed: {'Correct' if is_correct else 'Incorrect'} in {time_used:.2f}s")
        
        # Print summary to console for visibility
        print(f"\n{'='*60}")
        print(f"Task {task_data['id']}: {result_emoji} {'Correct' if is_correct else 'Incorrect'}")
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

    def finalize(self) -> Dict[str, Any]:
        """Finalize evaluation and write overall metrics to overall.json.
        
        This method should be called after all tasks have been executed.
        It calculates overall metrics from all stored results and writes to overall.json.
        
        Returns:
            The calculated overall metrics dictionary
        """
        if not self.all_results:
            logger.warning("No results to finalize")
            return {}
        
        if not self.current_output_dir:
            logger.error("No output directory set")
            return {}
        
        logger.info(f"Finalizing evaluation with {len(self.all_results)} results...")
        
        overall_metrics = write_overall_metrics(
            output_dir=self.current_output_dir,
            results=self.all_results,
            task_data_list=self.all_task_data,
            fhir_api_base=self.current_fhir_api_base
        )
        
        logger.info(f"Overall success rate: {overall_metrics['custom']['success_rate']:.2%}")
        logger.info(f"Correct: {overall_metrics['custom']['correct_count']}/{overall_metrics['total']}")
        
        # Reset for next batch
        self.all_results = []
        self.all_task_data = []
        
        return overall_metrics


def start_medagent_green(
    agent_name: str = "medagent_green_agent",
    host: str = None,
    port: int = None
) -> None:
    """Start the MedAgentBench green agent server.

    Args:
        agent_name: Name of the agent configuration file (default: "medagent_green_agent")
        host: Host to bind the server to (reads from HOST env var, default: "localhost")
        port: Port to bind the server to (reads from AGENT_PORT env var, default: 9001)
    """
    # Read from environment variables if not provided (for AgentBeats controller compatibility)
    if host is None:
        host = os.environ.get("HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("AGENT_PORT", "9001"))

    logger.info(f"Starting MedAgentBench green agent on {host}:{port}...")
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
