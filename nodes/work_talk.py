from langgraph.types import Send
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from .agent_state import AgentState, llm
from pydantic import BaseModel, Field
from typing import Optional

class WorkQuestionDecision(BaseModel):
    has_work_question: bool = Field(
        description="True если в сообщении есть рабочий вопрос про аренду, договора, финансы или юридические вопросы арнеды"
    )
    has_small_talk: bool = Field(
        description="True если в сообщении есть small talk"
    )
    work_question: Optional[str] = Field(
        default=None,
        description="Extracted рабочий вопрос если он есть в сообщении, иначе None"
    )
    small_talk: Optional[str] = Field(
        default=None,
        description="Extracted часть сообщения, в которая является small talk если он есть, иначе None"
    )

work_detector_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Раздели сообщение пользователя на 2 части:
    1. Small talk - приветствие, вопросы как дела, погоду, шутки, отвлеченные темы
    2. Рабочий вопрос - все, что касается аренды, финансов, договоров, объектов недвидимости и законов в этой области

    Важно: сообщение может содержать как обе части, так и только одну.
    """),
    ("human", "Вопрос пользователя: {query}")
])

work_detector = work_detector_prompt | llm.with_structured_output(WorkQuestionDecision)

def work_small_talk_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    result = work_detector.invoke({"query": last_message})
    return {
        "has_work_question": result.has_work_question,
        "has_small_talk": result.has_small_talk,
        "work_question": result.work_question,
        "small_talk": result.small_talk
        }

def route_work_small_talk(state: AgentState) -> list[Send]:
    sends = []

    if state.get("has_work_question"):
        work_state = {**state, "work_question": [HumanMessage(content=state["work_question"])]}
        sends.append(Send("router_node", work_state))
    
    if state.get("has_small_talk"):
        work_state = {**state, "small_talk": [HumanMessage(content=state["small_talk"])]}
        sends.append(Send("assistant_node", work_state))

    if not sends:
        sends.append(Send("assistant_node", state))

    return sends