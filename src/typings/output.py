from typing import Union, List, Optional

from pydantic import BaseModel, model_validator

from .general import JSONSerializable, SampleIndex, ChatHistoryItem
from .status import AgentOutputStatus


class TaskOutput(BaseModel):
    index: Optional[SampleIndex] = None
    status: Optional[str] = None
    result: JSONSerializable = None
    history: Optional[List[ChatHistoryItem]] = None
    rounds: Optional[int] = None


class AgentOutput(BaseModel):
    status: AgentOutputStatus = AgentOutputStatus.NORMAL
    content: Optional[str] = None

    @model_validator(mode='after')
    def validate_content(self):
        if self.status is AgentOutputStatus.NORMAL and self.content is None:
            raise ValueError("If status is NORMAL, content should not be None")
        return self
