"""MedAgentBench white agent - supports MCP server tool calling."""

import os
import uvicorn
import dotenv
import json
from typing import Dict, Any, List, Optional
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion
from mcp import ClientSession
from mcp.client.sse import sse_client
from src.my_util import logging_config

dotenv.load_dotenv()

# MCP server URL from environment (default to localhost:8002 for local development)
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8002")

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
    """White agent executor with MCP server tool calling support.
    
    The agent receives all instructions from the green agent's prompt.
    MCP server URL is configured via MCP_SERVER_URL environment variable.
    """

    def __init__(self):
        self.ctx_id_to_messages = {}
        self.mcp_server_url = MCP_SERVER_URL

    async def discover_tools(self, session: ClientSession) -> List[Dict[str, Any]]:
        """Discover available tools from MCP server.

        Args:
            session: MCP client session

        Returns:
            List of tool descriptors
        """
        try:
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                })
            return tools
        except Exception as e:
            logger.error(f"Error discovering tools from MCP server: {e}")
            return []

    async def invoke_tool(
        self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a tool via MCP server.

        Args:
            session: MCP client session
            tool_name: Name of the tool to invoke
            arguments: Tool arguments

        Returns:
            Tool invocation result
        """
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            # Extract content from the result
            if result.content:
                # MCP returns content as a list of content items
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    elif hasattr(item, 'data'):
                        content_parts.append(str(item.data))
                
                # Try to parse as JSON if it looks like JSON
                combined = "\n".join(content_parts)
                try:
                    return json.loads(combined)
                except json.JSONDecodeError:
                    return {"result": combined}
            return {"result": str(result)}
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
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def _run_with_mcp(
        self,
        messages: List[Dict[str, Any]],
        session: ClientSession,
        openai_tools: List[Dict[str, Any]],
        context_id: str
    ) -> str:
        """Run the LLM loop with MCP tools.

        Args:
            messages: Conversation history
            session: MCP client session
            openai_tools: Tools in OpenAI format
            context_id: Context ID for logging

        Returns:
            Final response content with FHIR operations metadata
        """
        max_iterations = 10  # Prevent infinite loops
        fhir_posts: List[Dict[str, Any]] = []  # Track FHIR POST operations for evaluation
        
        for iteration in range(max_iterations):
            logger.info(f"Calling GPT-5 with tools (iteration {iteration + 1})")
            response = completion(
                messages=messages,
                model="openai/gpt-5",
                custom_llm_provider="openai",
                tools=openai_tools,
                tool_choice="auto"
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
                        session,
                        function_name,
                        function_args
                    )
                    
                    # Check if this is a successful POST operation
                    is_error = isinstance(tool_result, dict) and "error" in tool_result
                    logger.info(f"Tool '{function_name}' returned {'error' if is_error else 'success'}")

                    # Track FHIR POST operations for evaluation (from MCP server response)
                    if isinstance(tool_result, dict) and "fhir_post" in tool_result:
                        fhir_post = tool_result["fhir_post"]
                        fhir_posts.append(fhir_post)
                        logger.info(f"Tracked FHIR POST to {fhir_post['fhir_url']}")

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
                # Append FHIR operations metadata if any POST requests were made
                if fhir_posts:
                    content = f"{content}\n<fhir_operations>\n{json.dumps(fhir_posts)}\n</fhir_operations>"
                return content

        # If we exhausted iterations
        logger.warning(f"Maximum iterations ({max_iterations}) reached")
        result = "FINISH([\"Unable to complete task within maximum iterations\"])"
        if fhir_posts:
            result = f"{result}\n<fhir_operations>\n{json.dumps(fhir_posts)}\n</fhir_operations>"
        return result

    async def _run_without_tools(self, messages: List[Dict[str, Any]]) -> str:
        """Run LLM without tools.

        Args:
            messages: Conversation history

        Returns:
            Response content
        """
        logger.info("Calling GPT-5 without tools")
        response = completion(
            messages=messages,
            model="openai/gpt-5",
            custom_llm_provider="openai",
        )
        message = response.choices[0].message.model_dump()
        messages.append(message)
        return message.get("content", "")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the white agent task with MCP tool support.

        The agent receives all instructions from the green agent's prompt.
        MCP server URL is configured via environment variable.

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

        # Connect to MCP server and run the entire tool-calling loop within the context
        sse_url = f"{self.mcp_server_url}/sse"
        logger.info(f"Connecting to MCP server at {sse_url}")

        final_content: Optional[str] = None

        try:
            # Use proper async with blocks to keep the connection alive
            async with sse_client(sse_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("MCP session initialized")

                    # Discover tools
                    mcp_tools = await self.discover_tools(session)
                    logger.info(f"Discovered {len(mcp_tools)} tools")

                    if mcp_tools:
                        openai_tools = self.convert_tools_to_openai_format(mcp_tools)
                        # Run the LLM loop with tools
                        final_content = await self._run_with_mcp(
                            messages, session, openai_tools, context.context_id
                        )
                    else:
                        # No tools available, run without
                        final_content = await self._run_without_tools(messages)

        except Exception as e:
            logger.error(f"Failed to connect to MCP server or execute: {e}")
            # Fall back to running without tools
            final_content = await self._run_without_tools(messages)

        # Send final response
        if final_content:
            preview = final_content[:200] + "..." if len(final_content) > 200 else final_content
            logger.info(f"Sending final response to green agent: {preview}")
            await event_queue.enqueue_event(
                new_agent_text_message(final_content, context_id=context.context_id)
            )
        else:
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "FINISH([\"No response generated\"])",
                    context_id=context.context_id
                )
            )

    async def cancel(self, context, event_queue) -> None:
        """Cancel execution."""
        raise NotImplementedError


def start_medagent_white(
    agent_name: str = "medagent_white_agent",
    host: str = None,
    port: int = None
):
    """Start the MedAgentBench white agent server.

    Args:
        agent_name: Name of the agent
        host: Host to bind to (reads from HOST env var, default: "0.0.0.0")
        port: Port to bind to (reads from AGENT_PORT env var, default: 9019)
    """
    # Read from environment variables if not provided (for AgentBeats controller compatibility)
    if host is None:
        host = os.environ.get("HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("AGENT_PORT", "9019"))

    logger.info(f"Starting MedAgentBench white agent on {host}:{port}...")
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
