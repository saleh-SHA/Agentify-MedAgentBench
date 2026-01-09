"""Green agent implementation for MedAgentBench - manages assessment and evaluation."""

import uvicorn
import tomllib
import dotenv
import json
import time
from typing import Dict, Any
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a, logging_config

dotenv.load_dotenv()

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
    task_prompt = f"""You are a medical AI assistant with access to tools via an MCP (Model Context Protocol) server.

MCP Server URL: {mcp_server_url}

Instructions:
1. You have access to tools for querying and updating a FHIR (Fast Healthcare Interoperability Resources) server.
2. First, discover available tools by querying: GET {mcp_server_url}/tools
3. Use tools by calling the MCP server's tool invocation endpoint: POST {mcp_server_url}/tools/invoke
4. The white agent should handle all interactions with the MCP server and FHIR server directly.
5. When you have the final answer, respond with: FINISH(answer)
   Example: FINISH(Patient MRN is S6534835)

Task Context: {task_data.get('context', 'N/A')}

Question: {task_data['instruction']}

Please start by discovering available tools from the MCP server and then use them to answer the question.
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

        logger.info(f"Task ID: {task_data['id']}")
        logger.info(f"MCP Server: {mcp_server_url}")
        logger.info(f"Max rounds: {max_rounds}")

        # Run the evaluation
        logger.info("Starting evaluation...")
        timestamp_started = time.time()

        result = await ask_agent_to_solve(
            white_agent_url=white_agent_url,
            task_data=task_data,
            mcp_server_url=mcp_server_url,
            max_num_steps=max_rounds
        )

        # Calculate metrics
        metrics = {
            "time_used": time.time() - timestamp_started,
            "status": result["status"],
            "rounds_used": result["rounds"],
            "correct": result.get("correct"),
        }

        # Determine success
        success = result["status"] == "completed" and result.get("correct") is True
        result_emoji = "✅" if success else "❌"

        # Prepare report
        report = f"""MedAgentBench Evaluation Complete {result_emoji}

            Task ID: {task_data['id']}
            Question: {task_data['instruction']}
            Expected Answer: {task_data.get('sol', 'N/A')}
            Agent Answer: {result.get('result', 'N/A')}
            Status: {result['status']}
            Correct: {result.get('correct', 'N/A')}
            Rounds Used: {result['rounds']}
            Time: {metrics['time_used']:.2f}s

            Metrics: {json.dumps(metrics, indent=2)}
            """

        logger.info("Evaluation complete.")
        logger.info(report)

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
