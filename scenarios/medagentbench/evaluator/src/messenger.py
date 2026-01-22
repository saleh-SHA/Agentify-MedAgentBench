import asyncio
import json
import logging
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


DEFAULT_TIMEOUT = 600  # 10 minutes to handle longer LLM inference times
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts for transient errors
DEFAULT_RETRY_DELAY = 5  # Initial delay between retries (seconds)

logger = logging.getLogger("messenger")


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


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
        outputs: dict[str, object] = {"response": "", "context_id": None}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] = str(outputs["response"]) + merge_parts(msg.parts)
            case (task, _update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                status_msg = task.status.message
                if status_msg:
                    outputs["response"] = str(outputs["response"]) + merge_parts(status_msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] = str(outputs["response"]) + merge_parts(artifact.parts)
            case _:
                pass

        return outputs


class Messenger:
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: int = DEFAULT_RETRY_DELAY):
        self._context_ids: dict[str, str | None] = {}
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """Send message to agent with retry logic for transient errors.

        Retries on timeout and connection errors with exponential backoff.
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                outputs = await send_message(
                    message=message,
                    base_url=url,
                    context_id=None if new_conversation else self._context_ids.get(url),
                    timeout=timeout,
                )
                if outputs.get("status", "completed") != "completed":
                    raise RuntimeError(f"{url} responded with: {outputs}")
                self._context_ids[url] = outputs.get("context_id")
                return str(outputs["response"])

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff: delay * 2^attempt
                    wait_time = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self._max_retries} failed for {url}: {type(e).__name__}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self._max_retries} attempts failed for {url}: {e}")
                    raise
            except Exception as e:
                # Non-retryable errors - raise immediately
                logger.error(f"Non-retryable error for {url}: {e}")
                raise

        # Should not reach here, but just in case
        raise last_error or RuntimeError(f"Failed to communicate with {url}")

    def reset(self) -> None:
        self._context_ids = {}

