from typing import Annotated, TypedDict, Optional
from langchain_openai import ChatOpenAI
from settings import Settings
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

settings = Settings()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: Optional[str]
    reason: Optional[str]
    tenant: Optional[str]
    tenant_history: Optional[dict] # {"hemotest": "...", "ozon": "..."}

llm = ChatOpenAI(
    model = settings.default_model,
    temperature = 0,
    base_url = settings.proxy_base_url
    )