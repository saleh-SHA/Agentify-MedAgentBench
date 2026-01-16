"""MedAgentBench purple agent - medical AI agent with MCP tool calling support."""

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from litellm import completion
from mcp import ClientSession
from mcp.client.sse import sse_client

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medagentbench_agent")

# Configuration from environment (defaults for local development)
DEFAULT_MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8002")
DEFAULT_FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/").rstrip("/")
LLM_MODEL = os.environ.get("MEDAGENT_LLM_MODEL", "openai/gpt-5")


def extract_mcp_server_url(text: str) -> str | None:
    """Extract MCP server URL from task prompt.
    
    Looks for pattern: 'MCP server at: <url>'
    """
    pattern = r'MCP server at:\s*(https?://[^\s]+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1).rstrip('/')
    return None


# Mapping of MCP tool names to FHIR POST operations for tracking
FHIR_POST_TOOLS = {
    "record_vital_observation": {
        "endpoint": "/Observation",
        "body_builder": lambda args: {
            "resourceType": args.get("resourceType", "Observation"),
            "category": args.get("category", []),
            "code": args.get("code", {}),
            "effectiveDateTime": args.get("effectiveDateTime", ""),
            "status": args.get("status", ""),
            "valueString": args.get("valueString", ""),
            "subject": args.get("subject", {}),
        }
    },
    "create_medication_request": {
        "endpoint": "/MedicationRequest",
        "body_builder": lambda args: {
            "resourceType": args.get("resourceType", "MedicationRequest"),
            "medicationCodeableConcept": args.get("medicationCodeableConcept", {}),
            "authoredOn": args.get("authoredOn", ""),
            "dosageInstruction": args.get("dosageInstruction", []),
            "status": args.get("status", ""),
            "intent": args.get("intent", ""),
            "subject": args.get("subject", {}),
        }
    },
    "create_service_request": {
        "endpoint": "/ServiceRequest",
        "body_builder": lambda args: {
            k: v for k, v in {
                "resourceType": args.get("resourceType", "ServiceRequest"),
                "code": args.get("code", {}),
                "authoredOn": args.get("authoredOn", ""),
                "status": args.get("status", ""),
                "intent": args.get("intent", ""),
                "priority": args.get("priority", ""),
                "subject": args.get("subject", {}),
                "occurrenceDateTime": args.get("occurrenceDateTime"),
                "note": args.get("note"),
            }.items() if v is not None
        }
    },
}


class Agent:
    def __init__(self):
        self.mcp_server_url = DEFAULT_MCP_SERVER_URL
        self.fhir_api_base = DEFAULT_FHIR_API_BASE
        self.model = LLM_MODEL

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
    ) -> str:
        """Run the LLM loop with MCP tools."""
        max_iterations = 10
        fhir_posts: list[dict[str, Any]] = []
        
        for iteration in range(max_iterations):
            logger.info(f"Calling {self.model} with tools (iteration {iteration + 1})")
            response = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider="openai",
                tools=openai_tools,
                tool_choice="auto"
            )

            message = response.choices[0].message.model_dump()
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.info(f"Model requested {len(tool_calls)} tool calls")

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Invoking tool '{function_name}'")

                    tool_result = await self.invoke_tool(
                        session,
                        function_name,
                        function_args
                    )
                    
                    is_error = isinstance(tool_result, dict) and "error" in tool_result
                    logger.info(f"Tool '{function_name}' returned {'error' if is_error else 'success'}")

                    # Track FHIR POST operations
                    if isinstance(tool_result, dict) and "fhir_post" in tool_result:
                        fhir_post = tool_result["fhir_post"]
                        fhir_posts.append(fhir_post)
                        logger.info(f"Tracked FHIR POST to {fhir_post['fhir_url']}")
                    elif function_name in FHIR_POST_TOOLS and not is_error:
                        tool_config = FHIR_POST_TOOLS[function_name]
                        fhir_url = f"{self.fhir_api_base}{tool_config['endpoint']}"
                        payload = tool_config["body_builder"](function_args)
                        
                        accepted = False
                        if isinstance(tool_result, dict):
                            status_code = tool_result.get("status_code", 0)
                            accepted = status_code in (200, 201)
                        
                        fhir_post = {
                            "fhir_url": fhir_url,
                            "payload": payload,
                            "accepted": accepted
                        }
                        fhir_posts.append(fhir_post)
                        logger.info(f"Tracked FHIR POST to {fhir_url}, accepted={accepted}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })

                continue

            content = message.get("content")
            if content:
                if fhir_posts:
                    content = f"{content}\n<fhir_operations>\n{json.dumps(fhir_posts)}\n</fhir_operations>"
                return content

        logger.warning(f"Maximum iterations ({max_iterations}) reached")
        result = "FINISH([\"Unable to complete task within maximum iterations\"])"
        if fhir_posts:
            result = f"{result}\n<fhir_operations>\n{json.dumps(fhir_posts)}\n</fhir_operations>"
        return result

    async def _run_without_tools(self, messages: list[dict[str, Any]]) -> str:
        """Run LLM without tools."""
        logger.info(f"Calling {self.model} without tools")
        response = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider="openai",
        )
        message = response.choices[0].message.model_dump()
        messages.append(message)
        return message.get("content", "")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Execute the agent task with MCP tool support."""
        user_input = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing request..."),
        )

        # Extract MCP server URL from the task prompt (sent by green agent)
        mcp_url = extract_mcp_server_url(user_input)
        if mcp_url:
            self.mcp_server_url = mcp_url
            logger.info(f"Extracted MCP server URL from prompt: {mcp_url}")
        else:
            logger.info(f"Using default MCP server URL: {self.mcp_server_url}")

        messages = [{"role": "user", "content": user_input}]

        # Connect to MCP server
        sse_url = f"{self.mcp_server_url}/sse"
        logger.info(f"Connecting to MCP server at {sse_url}")

        final_content: str | None = None

        try:
            logger.info(f"Attempting SSE connection to {sse_url}...")
            async with sse_client(sse_url) as (read_stream, write_stream):
                logger.info("SSE connection established, initializing MCP session...")
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")

                    mcp_tools = await self.discover_tools(session)
                    logger.info(f"Discovered {len(mcp_tools)} tools: {[t['name'] for t in mcp_tools]}")

                    if mcp_tools:
                        openai_tools = self.convert_tools_to_openai_format(mcp_tools)
                        final_content = await self._run_with_mcp(
                            messages, session, openai_tools
                        )
                    else:
                        logger.warning("No tools discovered from MCP server, running without tools")
                        final_content = await self._run_without_tools(messages)

        except Exception as e:
            logger.error(f"Failed to connect to MCP server at {sse_url}: {e}", exc_info=True)
            logger.warning("Falling back to running without tools")
            final_content = await self._run_without_tools(messages)

        if final_content:
            preview = final_content[:200] + "..." if len(final_content) > 200 else final_content
            logger.info(f"Sending response: {preview}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=final_content))],
                name="Response",
            )
        else:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="FINISH([\"No response generated\"])"))],
                name="Response",
            )

