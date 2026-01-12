"""MedAgentBench white agent - supports MCP server tool calling."""

import uvicorn
import dotenv
import json
import httpx
from typing import Dict, Any, List
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion
from src.my_util import logging_config

dotenv.load_dotenv()

logger = logging_config.setup_logging("logs/white_agent.log", "white_agent")


def prepare_medagent_white_card(url):
    """Prepare agent card for MedAgentBench white agent."""
    skill = AgentSkill(
        id="medical_task_fulfillment",
        name="Medical Task Fulfillment",
        description="Handles medical tasks using MCP server tools to query FHIR servers",
        tags=["medical", "fhir", "mcp"],
        examples=[],
    )
    card = AgentCard(
        name="medagent_white_agent",
        description="Medical AI agent with MCP server tool calling capability",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class MedAgentWhiteExecutor(AgentExecutor):
    """White agent executor with MCP server tool calling support."""

    def __init__(self):
        self.ctx_id_to_messages = {}
        self.ctx_id_to_tools = {}

    async def discover_tools(self, mcp_server_url: str) -> List[Dict[str, Any]]:
        """Discover available tools from MCP server.

        Args:
            mcp_server_url: URL of the MCP server

        Returns:
            List of tool descriptors
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{mcp_server_url}/tools", timeout=10.0)
                response.raise_for_status()
                tools = response.json()
                return tools
        except Exception as e:
            logger.error(f"Error discovering tools from MCP server: {e}")
            return []

    async def invoke_tool(
        self, mcp_server_url: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a tool via MCP server.

        Args:
            mcp_server_url: URL of the MCP server
            tool_name: Name of the tool to invoke
            arguments: Tool arguments

        Returns:
            Tool invocation result
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{mcp_server_url}/tools/invoke",
                    json={"tool_name": tool_name, "arguments": arguments},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "tool_name": tool_name
            }

    def convert_tools_to_openai_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tool descriptors to OpenAI function calling format.

        Args:
            mcp_tools: List of MCP tool descriptors

        Returns:
            List of OpenAI function definitions
        """
        openai_tools = []
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the white agent task with MCP tool support.

        Args:
            context: Request context
            event_queue: Event queue for responses
        """
        user_input = context.get_user_input()

        # Initialize conversation history for this context
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # Check if this is the first message and contains MCP server URL
        if len(messages) == 1 and "MCP Server URL:" in user_input:
            # Extract MCP server URL
            mcp_url = None
            for line in user_input.split('\n'):
                if line.startswith('MCP Server URL:'):
                    mcp_url = line.split('MCP Server URL:')[1].strip()
                    break

            if mcp_url:
                # Discover tools from MCP server
                logger.info(f"Discovering tools from MCP server: {mcp_url}")
                mcp_tools = await self.discover_tools(mcp_url)
                self.ctx_id_to_tools[context.context_id] = {
                    "mcp_url": mcp_url,
                    "tools": mcp_tools,
                    "openai_tools": self.convert_tools_to_openai_format(mcp_tools)
                }
                logger.info(f"Discovered {len(mcp_tools)} tools")

        # Get tools for this context
        tool_config = self.ctx_id_to_tools.get(context.context_id)

        # Call LLM with or without tools
        max_iterations = 10  # Prevent infinite loops
        for iteration in range(max_iterations):
            if tool_config:
                # Use function calling
                logger.info(f"Calling GPT-5 with {len(tool_config['openai_tools'])} tools (iteration {iteration + 1})")
                response = completion(
                    messages=messages,
                    model="openai/gpt-5",
                    custom_llm_provider="openai",
                    tools=tool_config["openai_tools"],
                    tool_choice="auto"
                )
            else:
                # Regular completion without tools
                logger.info(f"Calling GPT-5 without tools")
                response = completion(
                    messages=messages,
                    model="openai/gpt-5",
                    custom_llm_provider="openai",
                )

            message = response.choices[0].message.model_dump()
            messages.append(message)

            # Check if the model wants to call a tool
            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.info(f"Model requested {len(tool_calls)} tool calls")

                # Execute each tool call
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Invoking tool '{function_name}' with args: {function_args}")

                    # Invoke tool via MCP server
                    tool_result = await self.invoke_tool(
                        tool_config["mcp_url"],
                        function_name,
                        function_args
                    )
                    logger.info(f"Tool result received")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })

                # Continue loop to get next response from model
                continue

            # No tool calls, check if we have a final answer
            content = message.get("content")
            if content:
                logger.info(f"Sending response to green agent")
                messages.append({
                    "role": "assistant",
                    "content": content
                })

                await event_queue.enqueue_event(
                    new_agent_text_message(content, context_id=context.context_id)
                )
                break

        # If we exhausted iterations
        if iteration == max_iterations - 1:
            error_msg = "Maximum iterations reached without completing task"
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        """Cancel execution."""
        raise NotImplementedError


def start_medagent_white(
    agent_name: str = "medagent_white_agent",
    host: str = "localhost",
    port: int = 9002
):
    """Start the MedAgentBench white agent server.

    Args:
        agent_name: Name of the agent
        host: Host to bind to
        port: Port to bind to
    """
    logger.info("Starting MedAgentBench white agent...")
    url = f"http://{host}:{port}"
    card = prepare_medagent_white_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=MedAgentWhiteExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
