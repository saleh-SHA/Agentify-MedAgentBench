"""MedAgentBench purple agent - medical AI agent with MCP tool calling support."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from litellm import completion
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medagentbench_agent")

# Configuration from environment (required)
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")  # Can be overridden via DataPart config

# LLM configuration (required)
# LLM_MODEL: The model identifier in LiteLLM format (e.g., "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet")
LLM_MODEL = "openai/gpt-4o-mini"

# LLM_PROVIDER: Optional provider override (e.g., "openai", "anthropic", "google", "azure")
# Set to None (or leave empty) to let LiteLLM auto-detect from the model string prefix
LLM_PROVIDER = os.environ.get("MEDAGENT_LLM_PROVIDER", None) or None  # Convert empty string to None


def extract_config_from_message(message: Message) -> dict[str, Any]:
    """Extract structured config from message DataPart.
    
    The evaluator sends config (e.g., mcp_server_url) via DataPart alongside
    the text prompt. This function extracts that config.
    
    Returns:
        Config dict, or empty dict if no DataPart found.
    """
    for part in message.parts:
        if isinstance(part.root, DataPart):
            data = part.root.data
            if isinstance(data, dict):
                return data
    return {}


class Agent:
    def __init__(self):
        self.mcp_server_url = MCP_SERVER_URL  # Can be overridden via DataPart config
        self.model = LLM_MODEL
        self.llm_provider = LLM_PROVIDER  # None means auto-detect from model string
        self.max_iterations: int | None = None  # Must be provided via config

    async def discover_tools(self, session: ClientSession) -> list[dict[str, Any]]:
        """Discover available tools from MCP server."""
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
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke a tool via MCP server."""
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            if result.content:
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    elif hasattr(item, 'data'):
                        content_parts.append(str(item.data))
                
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

    def convert_tools_to_openai_format(self, mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tool descriptors to OpenAI function calling format."""
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
        messages: list[dict[str, Any]],
        session: ClientSession,
        openai_tools: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Run the LLM loop with MCP tools.
        
        Returns:
            Tuple of (response_text, metadata_dict) where metadata contains
            tool_history and fhir_operations for evaluation.
        """
        fhir_posts: list[dict[str, Any]] = []
        tool_history: list[dict[str, Any]] = []  # Track all tool calls
        
        for iteration in range(self.max_iterations):
            logger.info(f"Calling {self.model} with tools (iteration {iteration + 1})")
            # Build completion kwargs - only include custom_llm_provider if explicitly set
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "tools": openai_tools,
                "tool_choice": "auto",
            }
            if self.llm_provider:
                completion_kwargs["custom_llm_provider"] = self.llm_provider
            response = completion(**completion_kwargs)

            message = response.choices[0].message.model_dump()
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.info("-" * 60)
                logger.info(f"TOOL CALLS: Model requested {len(tool_calls)} tool call(s)")

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"  CALLING TOOL: {function_name}")
                    logger.info(f"  ARGUMENTS: {json.dumps(function_args, indent=2)}")

                    tool_result = await self.invoke_tool(
                        session,
                        function_name,
                        function_args
                    )
                    
                    is_error = isinstance(tool_result, dict) and "error" in tool_result
                    logger.info(f"  TOOL RESULT: {'ERROR' if is_error else 'SUCCESS'}")
                    result_preview = json.dumps(tool_result)
                    logger.debug(f"  RESULT DATA: {result_preview}")

                    # Track tool call for history
                    tool_history.append({
                        "tool_name": function_name,
                        "arguments": function_args,
                        "result": tool_result,
                        "is_error": is_error,
                    })

                    # Track FHIR POST operations (from MCP server response)
                    if isinstance(tool_result, dict) and "fhir_post" in tool_result:
                        fhir_post = tool_result["fhir_post"]
                        fhir_posts.append(fhir_post)
                        logger.debug(f"Tracked FHIR POST to {fhir_post['fhir_url']}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })

                continue

            content = message.get("content")
            if content:
                logger.debug("-" * 60)
                logger.debug("MODEL RESPONSE (no tool calls):")
                logger.debug(f"  {content[:500]}{'...' if len(content) > 500 else ''}")
                logger.debug("-" * 60)
                # Return content and metadata separately (no XML embedding)
                metadata = {
                    "tool_history": tool_history,
                    "fhir_operations": fhir_posts,
                    "rounds": iteration + 1,
                }
                return content, metadata

        logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
        result = "FINISH([\"Unable to complete task within maximum iterations\"])"
        metadata = {
            "tool_history": tool_history,
            "fhir_operations": fhir_posts,
            "rounds": self.max_iterations,
            "max_rounds_reached": True,  # Flag for evaluator to detect early termination
        }
        return result, metadata

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Execute the agent task with MCP tool support."""
        user_input = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing request..."),
        )

        # Extract config from DataPart (required)
        config = extract_config_from_message(message)
        
        if "mcp_server_url" in config:
            self.mcp_server_url = config["mcp_server_url"]
            logger.info(f"Extracted MCP server URL from DataPart config: {self.mcp_server_url}")
        
        if not self.mcp_server_url:
            raise RuntimeError("mcp_server_url must be provided via DataPart config or MCP_SERVER_URL environment variable")
        
        if "max_iterations" not in config:
            raise RuntimeError("max_iterations must be provided via DataPart config")
        self.max_iterations = int(config["max_iterations"])
        logger.info(f"Extracted max_iterations from DataPart config: {self.max_iterations}")

        messages = [{"role": "user", "content": user_input}]

        # Connect to MCP server
        mcp_url = f"{self.mcp_server_url}/mcp"
        logger.info(f"Connecting to MCP server at {mcp_url}")

        final_content: str | None = None
        metadata: dict[str, Any] = {"tool_history": [], "fhir_operations": []}

        try:
            logger.info(f"Attempting streamable-http connection to {mcp_url}...")
            async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
                logger.info("MCP connection established, initializing session...")
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")

                    mcp_tools = await self.discover_tools(session)
                    logger.info("=" * 60)
                    logger.info(f"TOOL DISCOVERY: Found {len(mcp_tools)} tools")
                    for t in mcp_tools:
                        logger.info(f"  - {t['name']}")
                    logger.info("=" * 60)

                    if mcp_tools:
                        openai_tools = self.convert_tools_to_openai_format(mcp_tools)
                        # Log full tool schemas for debugging
                        logger.debug("TOOL SCHEMAS SENT TO LLM:")
                        for tool in openai_tools:
                            func = tool.get("function", {})
                            logger.debug(f"  {func.get('name')}:")
                            params = func.get("parameters", {})
                            required = params.get("required", [])
                            props = params.get("properties", {})
                            logger.debug(f"    Required: {required}")
                            for prop_name, prop_schema in props.items():
                                logger.debug(f"    - {prop_name}: {prop_schema.get('type', 'unknown')} - {prop_schema.get('description', 'no desc')}")
                        logger.debug("=" * 60)
                        final_content, metadata = await self._run_with_mcp(
                            messages, session, openai_tools
                        )
                    else:
                        # No tools discovered - return error instead of fallback
                        logger.error("=" * 60)
                        logger.error("NO TOOLS DISCOVERED from MCP server")
                        logger.error("Cannot proceed without tools - returning error")
                        logger.error("=" * 60)
                        final_content = "FINISH([\"ERROR: no_tools_discovered - MCP server returned zero tools\"])"
                        metadata = {
                            "tool_history": [],
                            "fhir_operations": [],
                            "error_type": "no_tools_discovered",
                            "error_message": "MCP server returned zero tools",
                        }

        except Exception as e:
            # MCP connection failed - return error instead of fallback
            logger.error("=" * 60)
            logger.error(f"MCP CONNECTION FAILED: {mcp_url}")
            logger.error(f"ERROR: {e}")
            logger.error("Cannot proceed without MCP connection - returning error")
            logger.error("=" * 60)
            final_content = f"FINISH([\"ERROR: mcp_connection_failed - {str(e)}\"])"
            metadata = {
                "tool_history": [],
                "fhir_operations": [],
                "error_type": "mcp_connection_failed",
                "error_message": str(e),
            }

        if final_content:
            preview = final_content[:200] + "..." if len(final_content) > 200 else final_content
            logger.info(f"Sending response: {preview}")
            # Return both TextPart (answer) and DataPart (metadata) for clean extraction
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=final_content)),
                    Part(root=DataPart(data=metadata)),
                ],
                name="Response",
            )
        else:
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text="FINISH([\"No response generated\"])")),
                    Part(root=DataPart(data=metadata)),
                ],
                name="Response",
            )

