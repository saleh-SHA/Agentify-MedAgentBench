import json
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    TextPart,
)


DEFAULT_TIMEOUT = 300


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> tuple[str, dict | None]:
    """Extract text and data parts separately.
    
    Returns:
        Tuple of (text_content, data_dict) where data_dict is the first DataPart found or None.
    """
    text_chunks: list[str] = []
    data_part: dict | None = None
    
    for part in parts:
        if isinstance(part.root, TextPart):
            text_chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            # Capture the first DataPart as structured data
            if data_part is None:
                data_part = part.root.data
    
    return "\n".join(text_chunks), data_part


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
):
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs: dict[str, object] = {"response": "", "context_id": None, "metadata": None}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                text, data = merge_parts(msg.parts)
                outputs["response"] = str(outputs["response"]) + text
                if data is not None:
                    outputs["metadata"] = data
            case (task, _update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                status_msg = task.status.message
                if status_msg:
                    text, data = merge_parts(status_msg.parts)
                    outputs["response"] = str(outputs["response"]) + text
                    if data is not None:
                        outputs["metadata"] = data
                if task.artifacts:
                    for artifact in task.artifacts:
                        text, data = merge_parts(artifact.parts)
                        outputs["response"] = str(outputs["response"]) + text
                        if data is not None:
                            outputs["metadata"] = data
            case _:
                pass

        return outputs


class AgentResponse:
    """Structured response from an agent containing text and metadata."""
    def __init__(self, text: str, metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}
    
    @property
    def tool_history(self) -> list:
        """Get tool call history from metadata."""
        return self.metadata.get("tool_history", [])
    
    @property
    def fhir_operations(self) -> list:
        """Get FHIR operations from metadata."""
        return self.metadata.get("fhir_operations", [])


class Messenger:
    def __init__(self):
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> AgentResponse:
        """Send message to agent and return structured response.
        
        Returns:
            AgentResponse containing text and structured metadata (tool_history, fhir_operations).
        """
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            timeout=timeout,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id")
        
        text = str(outputs["response"])
        metadata = outputs.get("metadata")
        return AgentResponse(text=text, metadata=metadata if isinstance(metadata, dict) else None)

    def reset(self) -> None:
        self._context_ids = {}

