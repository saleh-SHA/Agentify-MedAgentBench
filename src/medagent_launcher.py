"""Launcher module for MedAgentBench - initiates and coordinates the evaluation process."""

import multiprocessing
import json
from pathlib import Path
from src.green_agent.medagent import start_medagent_green
from src.white_agent.medagent import start_medagent_white
from src.my_util import my_a2a, logging_config

logger = logging_config.setup_logging("logs/medagent.log", "medagent_launcher")


def load_medagent_test_data(data_file: str = None):
    """Load MedAgentBench test data.

    Args:
        data_file: Path to the test data JSON file. If None, uses default location.

    Returns:
        List of test cases
    """
    if data_file is None:
        # Default location
        project_root = Path(__file__).parent.parent
        data_file = project_root / "MedAgentBench" / "data" / "medagentbench" / "test_data_v2.json"

    with open(data_file, 'r') as f:
        return json.load(f)


async def launch_medagent_evaluation(
    task_index: int = 0,
    mcp_server_url: str = "http://0.0.0.0:8002",
    max_rounds: int = 8,
    green_port: int = 9001,
    white_port: int = 9002
):
    """Launch a MedAgentBench evaluation.

    Args:
        task_index: Index of the task to run from test data
        mcp_server_url: URL of the MCP server providing FHIR tools
        max_rounds: Maximum number of interaction rounds
        green_port: Port for the green agent
        white_port: Port for the white agent
    """

    # Load test data
    logger.info("Loading MedAgentBench test data...")
    test_data = load_medagent_test_data()

    if task_index >= len(test_data):
        logger.error(f"Error: Task index {task_index} out of range (max: {len(test_data) - 1})")
        return

    task_data = test_data[task_index]
    logger.info(f"Selected task: {task_data['id']}")
    logger.info(f"Question: {task_data['instruction']}")
    logger.info(f"Expected answer: {task_data['sol']}")

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
    logger.info(response)
    logger.info("=" * 80)

    # Cleanup
    logger.info("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("Agents terminated.")


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
    test_data = load_medagent_test_data()

    if task_indices is None:
        task_indices = list(range(len(test_data)))

    results = []

    for idx in task_indices:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running task {idx + 1}/{len(task_indices)}")
        logger.info(f"{'='*80}\n")

        await launch_medagent_evaluation(
            task_index=idx,
            mcp_server_url=mcp_server_url,
            max_rounds=max_rounds
        )

    logger.info("\n" + "="*80)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("="*80)