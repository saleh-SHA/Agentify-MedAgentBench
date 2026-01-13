"""Launcher module for MedAgentBench - initiates and coordinates the evaluation process."""

import multiprocessing
import json
from pathlib import Path

import requests
from src.green_agent.medagent import start_medagent_green
from src.white_agent.medagent import start_medagent_white
from src.my_util import my_a2a, logging_config
from src.typings import *

logger = logging_config.setup_logging("logs/medagent.log", "medagent_launcher")

def load_medagent_tasks(mcp_server_url: str):
    """Load MedAgentBench tasks from MCP server.

    Args:
        mcp_server_url: URL of the MCP server

    Returns:
        List of MedAgentBench Task Inputs
    """
    try:
        response = requests.get(f"{mcp_server_url}/resources/medagentbench_tasks", timeout=10.0)
        response.raise_for_status()
        response_json = response.json()
        tasks = response_json["data"]["tasks"]
        return tasks
    except requests.RequestException as e:
        raise Exception(reason="Error loading MedAgentBench tasks from MCP server", detail=str(e))


async def launch_medagent_evaluation(
    task_index: int = 0,
    mcp_server_url: str = "http://0.0.0.0:8002",
    max_rounds: int = 9,
    green_port: int = 9001,
    white_port: int = 9002
):
    """Launch a MedAgentBench evaluation.

    Args:
        task_index: Index of the task to run from test data
        mcp_server_url: URL of the MCP server providing FHIR tools
        max_rounds: Maximum number of interaction rounds. Original MedAgentBench uses 8. We use 9 to allow for MCP tool discovery.
        green_port: Port for the green agent
        white_port: Port for the white agent

    Returns:
        Dictionary containing task_id, task_index, and evaluation response
    """

    # Load test data
    logger.info("Loading MedAgentBench test data...")
    tasks = load_medagent_tasks(mcp_server_url)
    if task_index >= len(tasks):
        logger.error(f"Error: Task index {task_index} out of range (max: {len(tasks) - 1})")
        return

    task_data = tasks[task_index]
    logger.info(f"Selected task: {task_data["id"]}")
    logger.info(f"Question: {task_data["instruction"]}")
    logger.info(f"Expected answer: {task_data.get("sol", [])}")

    # Start green agent
    logger.info("Launching MedAgentBench green agent...")
    green_address = ("localhost", green_port)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_medagent_green,
        args=("medagent_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
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
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready in time"
    logger.info("MedAgentBench white agent is ready.")

    # Prepare task configuration
    logger.info("Preparing task configuration...")
    medagent_config = {
        "mcp_server_url": mcp_server_url,
        "max_rounds": max_rounds,
        "task_data": task_data
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

    logger.info("Task configuration:")
    logger.info(task_text)

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
    logger.info("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("Agents terminated.")

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
    output_file: str = None
):
    """Launch batch MedAgentBench evaluations.

    Args:
        task_indices: List of task indices to run. If None, runs all tasks.
        mcp_server_url: URL of the MCP server
        max_rounds: Maximum rounds per task
        output_file: Path to save results. If None, prints to console.
    """
    logger.info("Loading MedAgentBench test data...")
    tasks = load_medagent_tasks(mcp_server_url)

    if task_indices is None:
        task_indices = list(range(len(tasks)))

    results = []

    for idx in task_indices:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running task {idx + 1}/{len(task_indices)}")
        logger.info(f"{'='*80}\n")

        result = await launch_medagent_evaluation(
            task_index=idx,
            mcp_server_url=mcp_server_url,
            max_rounds=max_rounds
        )
        results.append(result)

    logger.info("\n" + "="*80)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total tasks evaluated: {len(results)}")

    # Save results to file if specified
    if output_file:
        try:
            # Remove existing file if it exists
            output_path = Path(output_file)
            if output_path.exists():
                output_path.unlink()
                logger.info(f"Removed existing results file: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")

    return results