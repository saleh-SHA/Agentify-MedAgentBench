"""Launcher module for MedAgentBench - initiates and coordinates the evaluation process."""

import multiprocessing
import json
import os
import subprocess
import time
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from src.green_agent.medagent import start_medagent_green, OUTPUT_DIR
from src.white_agent.medagent import start_medagent_white
from src.my_util import my_a2a, logging_config
from src.typings import *
from src.typings.output import TaskOutput
from src.typings.general import ChatHistoryItem

logger = logging_config.setup_logging("logs/medagent.log", "medagent_launcher")

# Import FHIR_API_BASE for evaluation
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/")


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open and accepting connections.

    Args:
        host: Host to check
        port: Port to check
        timeout: Connection timeout in seconds

    Returns:
        True if port is open, False otherwise
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def wait_for_service(host: str, port: int, service_name: str, max_wait: int = 60, check_interval: float = 2.0) -> bool:
    """Wait for a service to become available on a specific port.

    Args:
        host: Host where service is running
        port: Port where service should be listening
        service_name: Name of the service (for logging)
        max_wait: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if service is available, False if timeout
    """
    logger.info(f"Waiting for {service_name} on {host}:{port}...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if is_port_open(host, port):
            logger.info(f"{service_name} is ready on {host}:{port}")
            return True
        time.sleep(check_interval)

    logger.error(f"{service_name} did not become ready within {max_wait} seconds")
    return False


def calculate_overall_from_runs(
    output_dir: str,
    task_data_list: List[Dict[str, Any]],
    fhir_api_base: str = FHIR_API_BASE
) -> Dict[str, Any]:
    """Calculate overall metrics from runs.jsonl file.
    
    This function reads the runs.jsonl file generated during batch evaluation
    and computes aggregate metrics similar to Customized MegAgentBench's overall.json.
    
    Args:
        output_dir: Directory containing runs.jsonl
        task_data_list: List of task data dictionaries (for reference)
        fhir_api_base: Base URL for FHIR API
        
    Returns:
        Dictionary containing overall metrics:
        - total: Total number of results
        - validation: Status distribution and history length statistics
        - custom: Task-specific metrics (success rate, raw results)
    """
    runs_file = os.path.join(output_dir, "runs.jsonl")
    
    if not os.path.exists(runs_file):
        logger.error(f"runs.jsonl not found at {runs_file}")
        return {}
    
    # Read all results from runs.jsonl
    results = []
    with open(runs_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if record.get("output"):
                        results.append(record["output"])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in runs.jsonl: {e}")
    
    if not results:
        logger.warning("No valid results found in runs.jsonl")
        return {"total": 0, "validation": {}, "custom": {"success_rate": 0, "raw_results": []}}
    
    total = len(results)
    
    # Calculate status distribution (matching MegAgentBench format)
    # Map to SampleStatus values for consistent reporting
    status_mapping = {
        "running": SampleStatus.RUNNING.value,
        "completed": SampleStatus.COMPLETED.value,
        "agent context limit": SampleStatus.AGENT_CONTEXT_LIMIT.value,
        "agent validation failed": SampleStatus.AGENT_VALIDATION_FAILED.value,
        "agent invalid action": SampleStatus.AGENT_INVALID_ACTION.value,
        "task limit reached": SampleStatus.TASK_LIMIT_REACHED.value,
        "unknown": SampleStatus.UNKNOWN.value,
        "task error": SampleStatus.TASK_ERROR.value,
    }
    
    # Initialize status counts for all known statuses
    status_counts: Dict[str, float] = {s.value: 0.0 for s in SampleStatus}
    
    for result in results:
        status = result.get("status", "unknown")
        # Extract base status (before "Correct" or "Incorrect" suffix)
        base_status = status.split()[0].lower() if status else "unknown"
        
        # Map to known status or keep as unknown
        if base_status in status_mapping:
            status_counts[status_mapping[base_status]] += 1
        else:
            status_counts[SampleStatus.UNKNOWN.value] += 1
    
    # Convert to percentages
    for key in status_counts:
        status_counts[key] /= total
    
    # Calculate history length statistics
    history_lengths = []
    for result in results:
        history = result.get("history", [])
        history_lengths.append(len(history) if history else 0)
    
    validation = {
        **status_counts,
        "average_history_length": sum(history_lengths) / total if total > 0 else 0,
        "max_history_length": max(history_lengths) if history_lengths else 0,
        "min_history_length": min(history_lengths) if history_lengths else 0,
    }
    
    # Calculate success rate from the status (already evaluated during run)
    correct_count = 0
    for result in results:
        status = result.get("status", "")
        if "Correct" in status:
            correct_count += 1
    
    success_rate = correct_count / total if total > 0 else 0
    
    custom = {
        "success rate": success_rate,  # Match MegAgentBench key name with space
        "correct_count": correct_count,
        "total_evaluated": total,
        "raw_results": results,
    }
    
    return {
        "total": total,
        "validation": validation,
        "custom": custom,
    }


def write_overall_json(
    output_dir: str,
    task_data_list: List[Dict[str, Any]],
    fhir_api_base: str = FHIR_API_BASE
) -> Dict[str, Any]:
    """Calculate and write overall.json metrics file.
    
    This creates an overall.json file matching the format of Customized MegAgentBench.
    
    Args:
        output_dir: Directory to write overall.json to (and read runs.jsonl from)
        task_data_list: List of task data dictionaries
        fhir_api_base: Base URL for FHIR API
        
    Returns:
        The calculated overall metrics dictionary
    """
    overall = calculate_overall_from_runs(output_dir, task_data_list, fhir_api_base)
    
    if not overall:
        logger.warning("No metrics to write to overall.json")
        return {}
    
    output_file = os.path.join(output_dir, "overall.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Overall metrics written to {output_file}")
    
    # Print summary
    if overall.get("custom"):
        success_rate = overall["custom"].get("success rate", 0)
        correct = overall["custom"].get("correct_count", 0)
        total = overall.get("total", 0)
        logger.info(f"Success rate: {success_rate:.2%} ({correct}/{total})")
    
    if overall.get("validation"):
        avg_history = overall["validation"].get("average_history_length", 0)
        max_history = overall["validation"].get("max_history_length", 0)
        min_history = overall["validation"].get("min_history_length", 0)
        logger.info(f"History length - avg: {avg_history:.2f}, max: {max_history}, min: {min_history}")
    
    return overall


async def load_medagent_tasks(mcp_server_url: str):
    """Load MedAgentBench tasks from MCP server via SSE.

    Args:
        mcp_server_url: URL of the MCP server

    Returns:
        List of MedAgentBench Task Inputs
    """
    sse_url = f"{mcp_server_url}/sse"
    logger.info(f"Connecting to MCP server at {sse_url} to load tasks...")
    
    try:
        async with sse_client(sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Read the tasks resource
                result = await session.read_resource("medagentbench://tasks")
                
                # Extract content from the result
                if result.contents:
                    for content in result.contents:
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)
                            tasks = data.get("tasks", [])
                            logger.info(f"Loaded {len(tasks)} tasks from MCP server")
                            return tasks
                
                logger.error("No task content found in MCP response")
                return []
    except Exception as e:
        logger.error(f"Error loading MedAgentBench tasks from MCP server: {e}")
        raise Exception(f"Error loading MedAgentBench tasks from MCP server: {e}")


async def launch_medagent_evaluation(
    task_index: int = 0,
    mcp_server_url: str = "http://0.0.0.0:8002",
    max_rounds: int = 9,
    green_port: int = 9001,
    white_port: int = 9002,
    agent_name: str = "default_agent",
    task_name: str = "medagentbench"
):
    """Launch a MedAgentBench evaluation.

    Args:
        task_index: Index of the task to run from test data
        mcp_server_url: URL of the MCP server providing FHIR tools
        max_rounds: Maximum number of interaction rounds. Original MedAgentBench uses 8. We use 9 to allow for MCP tool discovery.
        green_port: Port for the green agent
        white_port: Port for the white agent
        agent_name: Name of the agent being evaluated (for output directory)
        task_name: Name of the task (for output directory)

    Returns:
        Dictionary containing task_id, task_index, and evaluation response
    """

    # Step 1: Launch FHIR server using fhir_launcher.sh (if not already running)
    logger.info("=" * 80)
    logger.info("CHECKING FHIR SERVER")
    logger.info("=" * 80)

    fhir_process = None
    if is_port_open("localhost", 8080):
        logger.info("FHIR server is already running on port 8080.")
    else:
        logger.info("FHIR server is not running. Launching it now...")
        fhir_launcher_script = os.path.join(os.getcwd(), "fhir_launcher.sh")
        if not os.path.exists(fhir_launcher_script):
            logger.error(f"FHIR launcher script not found at {fhir_launcher_script}")
            return

        try:
            # Launch FHIR server in background
            logger.info(f"Starting FHIR server using {fhir_launcher_script}...")
            fhir_process = subprocess.Popen(
                ["bash", fhir_launcher_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"FHIR server process started (PID: {fhir_process.pid})")

            # Wait for FHIR server to be ready
            fhir_ready = wait_for_service("localhost", 8080, "FHIR server", max_wait=300)
            assert fhir_ready, "FHIR server failed to start in time"
            logger.info("FHIR server is ready and accepting connections.")

        except Exception as e:
            logger.error(f"Failed to start FHIR server: {e}")
            raise

    # Step 2: Launch MCP server (if not already running)
    logger.info("=" * 80)
    logger.info("CHECKING MCP SERVER")
    logger.info("=" * 80)

    mcp_process = None
    if is_port_open("localhost", 8002):
        logger.info("MCP server is already running on port 8002.")
    else:
        logger.info("MCP server is not running. Launching it now...")
        try:
            # Launch MCP server in background
            logger.info("Starting MCP server (src.mcp.server)...")
            mcp_process = subprocess.Popen(
                ["python", "-m", "src.mcp.server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"MCP server process started (PID: {mcp_process.pid})")

            # Wait for MCP server to be ready
            mcp_ready = wait_for_service("localhost", 8002, "MCP server", max_wait=60)
            assert mcp_ready, "MCP server failed to start in time"
            logger.info("MCP server is ready and accepting connections.")

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            # Cleanup FHIR server if we started it
            if fhir_process:
                fhir_process.terminate()
            raise

    # Step 3: Load test data
    logger.info("=" * 80)
    logger.info("LOADING TEST DATA")
    logger.info("=" * 80)
    logger.info("Loading MedAgentBench test data...")
    tasks = await load_medagent_tasks(mcp_server_url)
    if task_index >= len(tasks):
        logger.error(f"Error: Task index {task_index} out of range (max: {len(tasks) - 1})")
        # Cleanup services only if we started them
        if mcp_process is not None:
            mcp_process.terminate()
        if fhir_process is not None:
            fhir_process.terminate()
        return

    task_data = tasks[task_index]
    logger.info(f"Selected task: {task_data['id']}")
    instruction_preview = task_data['instruction'][:150] + "..." if len(task_data['instruction']) > 150 else task_data['instruction']
    logger.info(f"Question: {instruction_preview}")

    # Start green agent
    logger.info("Launching MedAgentBench green agent...")
    green_address = ("localhost", green_port)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_medagent_green,
        args=("medagent_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url, timeout=30), "Green agent not ready in time"
    logger.info("Green agent is ready.")

    # Start MedAgentBench white agent
    logger.info("Launching MedAgentBench white agent...")
    white_address = ("localhost", white_port)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_medagent_white,
        args=("medagent_white_agent", *white_address)
    )
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url, timeout=30), "White agent not ready in time"
    logger.info("MedAgentBench white agent is ready.")

    # Prepare task configuration
    logger.info("Preparing task configuration...")
    medagent_config = {
        "mcp_server_url": mcp_server_url,
        "max_rounds": max_rounds,
        "task_data": task_data,
        "agent_name": agent_name,
        "task_name": task_name,
        "fhir_api_base": FHIR_API_BASE
    }

    task_text = f"""Your task is to instantiate MedAgentBench to test the agent located at:
            <white_agent_url>
            {white_url}/
            </white_agent_url>

            You should use the following configuration:
            <medagent_config>
            {json.dumps(medagent_config, indent=2)}
            </medagent_config>
        """

    logger.info(f"Prepared task configuration for task {task_data['id']}")

    # Send task to green agent
    logger.info("Sending task to green agent...")
    response = await my_a2a.send_message(green_url, task_text)

    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Extract text from A2A response structure
    response_text = None
    try:
        if hasattr(response, 'result') and hasattr(response.result, 'parts'):
            # Extract text from SendMessageSuccessResponse
            for part in response.result.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    response_text = part.root.text
                    logger.info(response_text)
        elif isinstance(response, str):
            response_text = response
            # If it's already a string, try to parse as JSON for formatting
            try:
                response_obj = json.loads(response)
                logger.info(json.dumps(response_obj, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                logger.info(response)
        else:
            # Fall back to string representation
            response_text = str(response)
            logger.info(response_text)
    except Exception as e:
        logger.warning(f"Could not format response: {e}")
        response_text = str(response)
        logger.info(response_text)

    logger.info("=" * 80)

    # Cleanup
    logger.info("Evaluation complete. Terminating processes...")

    # Terminate agents
    logger.info("Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("Agents terminated.")

    # Terminate MCP server only if we started it
    if mcp_process is not None:
        logger.info("Terminating MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=10)
            logger.info("MCP server terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("MCP server did not terminate gracefully, killing it...")
            mcp_process.kill()
    else:
        logger.info("MCP server was already running, leaving it active.")

    # Terminate FHIR server only if we started it
    if fhir_process is not None:
        logger.info("Terminating FHIR server...")
        fhir_process.terminate()
        try:
            fhir_process.wait(timeout=10)
            logger.info("FHIR server terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("FHIR server did not terminate gracefully, killing it...")
            fhir_process.kill()
    else:
        logger.info("FHIR server was already running, leaving it active.")

    logger.info("Cleanup complete.")

    # Return result for batch processing with JSON-serializable response
    return {
        "task_index": task_index,
        "task_id": task_data["id"],
        "response": response_text
    }


async def launch_medagent_batch_evaluation(
    task_indices: list = None,
    mcp_server_url: str = "http://0.0.0.0:8002",
    max_rounds: int = 8,
    agent_name: str = "default_agent",
    task_name: str = "medagentbench",
    clear_previous_runs: bool = True
):
    """Launch batch MedAgentBench evaluations.

    This function runs multiple MedAgentBench tasks in sequence and generates
    aggregate metrics similar to Customized MegAgentBench, including:
    - runs.jsonl: Individual task results (appended during execution)
    - overall.json: Aggregate metrics (generated at the end)

    Args:
        task_indices: List of task indices to run. If None, runs all tasks.
        mcp_server_url: URL of the MCP server
        max_rounds: Maximum rounds per task
        agent_name: Name of the agent being evaluated (for output directory)
        task_name: Name of the task (for output directory)
        clear_previous_runs: Whether to clear previous runs.jsonl before starting
        
    Returns:
        List of task results and writes overall.json to output directory
    """

    # Step 1: Launch FHIR server (if not already running)
    logger.info("=" * 80)
    logger.info("CHECKING FHIR SERVER FOR BATCH EVALUATION")
    logger.info("=" * 80)

    fhir_process = None
    if is_port_open("localhost", 8080):
        logger.info("FHIR server is already running on port 8080.")
    else:
        logger.info("FHIR server is not running. Launching it now...")
        fhir_launcher_script = os.path.join(os.getcwd(), "fhir_launcher.sh")
        if not os.path.exists(fhir_launcher_script):
            logger.error(f"FHIR launcher script not found at {fhir_launcher_script}")
            return []

        try:
            logger.info(f"Starting FHIR server using {fhir_launcher_script}...")
            fhir_process = subprocess.Popen(
                ["bash", fhir_launcher_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"FHIR server process started (PID: {fhir_process.pid})")

            fhir_ready = wait_for_service("localhost", 8080, "FHIR server", max_wait=300)
            assert fhir_ready, "FHIR server failed to start in time"
            logger.info("FHIR server is ready and accepting connections.")

        except Exception as e:
            logger.error(f"Failed to start FHIR server: {e}")
            raise

    # Step 2: Launch MCP server (if not already running)
    logger.info("=" * 80)
    logger.info("CHECKING MCP SERVER FOR BATCH EVALUATION")
    logger.info("=" * 80)

    mcp_process = None
    if is_port_open("localhost", 8002):
        logger.info("MCP server is already running on port 8002.")
    else:
        logger.info("MCP server is not running. Launching it now...")
        try:
            logger.info("Starting MCP server (src.mcp.server)...")
            mcp_process = subprocess.Popen(
                ["python", "-m", "src.mcp.server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"MCP server process started (PID: {mcp_process.pid})")

            mcp_ready = wait_for_service("localhost", 8002, "MCP server", max_wait=60)
            assert mcp_ready, "MCP server failed to start in time"
            logger.info("MCP server is ready and accepting connections.")

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            if fhir_process:
                fhir_process.terminate()
            raise

    # Step 3: Load test data
    logger.info("=" * 80)
    logger.info("LOADING TEST DATA FOR BATCH EVALUATION")
    logger.info("=" * 80)
    logger.info("Loading MedAgentBench test data...")
    tasks = await load_medagent_tasks(mcp_server_url)

    if task_indices is None:
        task_indices = list(range(len(tasks)))
    
    # Prepare output directory (matching green agent's path)
    output_dir = os.path.join(OUTPUT_DIR, agent_name, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear previous runs if requested
    if clear_previous_runs:
        runs_file = os.path.join(output_dir, "runs.jsonl")
        if os.path.exists(runs_file):
            os.remove(runs_file)
        
        error_file = os.path.join(output_dir, "error.jsonl")
        if os.path.exists(error_file):
            os.remove(error_file)
        
        overall_file = os.path.join(output_dir, "overall.json")
        if os.path.exists(overall_file):
            os.remove(overall_file)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Running {len(task_indices)} tasks...")

    results = []
    task_data_list = []  # Store task data for metrics calculation

    for i, idx in enumerate(task_indices):
        logger.info(f"\n{'='*80}")
        logger.info(f"Running task {i + 1}/{len(task_indices)} (index: {idx})")
        logger.info(f"{'='*80}\n")

        # Store task data for overall metrics calculation
        task_data_list.append(tasks[idx])

        result = await launch_medagent_evaluation(
            task_index=idx,
            mcp_server_url=mcp_server_url,
            max_rounds=max_rounds,
            agent_name=agent_name,
            task_name=task_name
        )
        results.append(result)

    logger.info("\n" + "="*80)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total tasks evaluated: {len(results)}")
    
    # Calculate and write overall metrics (matching MegAgentBench format)
    logger.info("Calculating overall metrics...")
    overall = write_overall_json(
        output_dir=output_dir,
        task_data_list=task_data_list,
        fhir_api_base=FHIR_API_BASE
    )
    
    # Print summary statistics
    if overall:
        print("\n" + "="*60)
        print("OVERALL METRICS")
        print("="*60)
        print(f"Total tasks: {overall.get('total', 0)}")
        if overall.get('custom'):
            success_rate = overall['custom'].get('success rate', 0)
            correct = overall['custom'].get('correct_count', 0)
            print(f"Success rate: {success_rate:.2%} ({correct}/{overall.get('total', 0)})")
        if overall.get('validation'):
            print(f"Average history length: {overall['validation'].get('average_history_length', 0):.2f}")
            print(f"Max history length: {overall['validation'].get('max_history_length', 0)}")
            print(f"Min history length: {overall['validation'].get('min_history_length', 0)}")
        print(f"Results saved to: {output_dir}")
        print("="*60 + "\n")

    # Cleanup services only if we started them
    logger.info("=" * 80)
    logger.info("CLEANING UP BATCH EVALUATION SERVICES")
    logger.info("=" * 80)

    if mcp_process is not None:
        logger.info("Terminating MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=10)
            logger.info("MCP server terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("MCP server did not terminate gracefully, killing it...")
            mcp_process.kill()
    else:
        logger.info("MCP server was already running, leaving it active.")

    if fhir_process is not None:
        logger.info("Terminating FHIR server...")
        fhir_process.terminate()
        try:
            fhir_process.wait(timeout=10)
            logger.info("FHIR server terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("FHIR server did not terminate gracefully, killing it...")
            fhir_process.kill()
    else:
        logger.info("FHIR server was already running, leaving it active.")

    logger.info("Batch evaluation cleanup complete.")

    return results
