from typing import List, Dict, Union, Any, Literal

from pydantic import BaseModel

JSONSerializable = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]
SampleIndex = Union[int, str]


class ChatHistoryItem(BaseModel):
    role: Literal["user", "agent"]
    content: str
