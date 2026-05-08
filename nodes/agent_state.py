from typing import Annotated, TypedDict, Optional
from settings import Settings
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add

settings = Settings()

#Классичеcкие решения падают, если передается None
def add_response(left: list | None, right: list | None) -> list:
    left = left or []
    right = right or []
    return left + right

def add_messages_safe(left: list | None, right: list | None) -> list:
    left = left or []
    right = right or []
    return add_messages(left, right)

#add_messages только для list[BaseMessage]
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: Optional[str]
    reason: Optional[str]
    tenant: Optional[str]
    tenant_history: Optional[dict] # {"hemotest": "...", "ozon": "..."}
    has_work_question: Optional[bool]
    has_small_talk: Optional[bool]
    work_question: Optional[str]
    small_talk: Optional[str]
    work_questions: Annotated[list[BaseMessage], add_messages_safe]
    small_talks: Annotated[list[BaseMessage], add_messages_safe]
    use_assistant: Optional[bool]