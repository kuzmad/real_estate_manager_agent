from langchain_core.prompts import ChatPromptTemplate
from .agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from settings import Settings
from pydantic import BaseModel, Field
from llm import llm
from utils import summarize_history

settings = Settings()

property_manager_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Ты менеджер коммерческой недвидимости.
    Ты ведешь историю взаимодейсвтия с арнедаторами.
    Запоминай детали: ремонты, поверки счетчиков, договоренности, жалобы.
    Отвечай на основе истории переписки
    """),
    ("system", "История общения: {history}"),
    ("human", "Вопрос пользователя: {query}")
])

class TenantDecision(BaseModel):
    tenant_id: str = Field(
        description=f"""Идентификатор арендатора. Возможные значения: {settings.tenants}
        Если не удалось определить арендатора или в запросе его нет, то верни general
    """
    )

tenant_extractor = ChatPromptTemplate.from_messages([
        ("system", 
     f"""
    Определи арендатора из запроса.
    Возможные значения: {settings.tenants}, а также general, когда арендатор не определен.
    Обрати внимание, что в тексте может быть написано на русском, например, hemotest = Гемотест,
    wildberries = вайлдбериз = вб = wb, ozon = озон
    """),
    ("human", "Вопрос пользователя: {query}")
]) | llm.with_structured_output(TenantDecision)

def tenant_extractor_node(state: AgentState) -> dict:
    messages = state["work_questions"]
    last_message = messages[-1].content
    result = tenant_extractor.invoke({"query": last_message})
    #print(result.tenant_id)
    return {"tenant": result.tenant_id}

manager = property_manager_prompt | llm

def update_tenant_history(
    tenant_history: dict,
    tenant_id: str,
    user_message: str,
    agent_response: str
) -> dict:
    updated = tenant_history.copy()
    
    if tenant_id not in updated:
        updated[tenant_id] = []

    updated[tenant_id].append(f"Human: {user_message}")
    updated[tenant_id].append(f"Agent: {agent_response}")

    #суммаризация если нужно
    if len(updated[tenant_id][:-2]) > settings.max_tenant_history:
        summary = summarize_history(updated[tenant_id][:-2])
        updated[tenant_id] = [f"Сводка: {summary}"] + updated[tenant_id][-2:]

    return updated


def property_manager_node(state: AgentState) -> dict:
    messages = state["work_questions"]
    last_message = messages[-1].content
    tenant_history = state.get("tenant_history") or {}
    tenant_id = state.get("tenant") or "general"

    history = "\n".join(tenant_history.get(tenant_id, []))

    result = manager.invoke({
        "history": history or "История пока пуста",
        "query": last_message
    })
    
    # Обновляем историю
    updated_history = update_tenant_history(
        tenant_history,
        tenant_id,
        last_message,
        result.content
    )

    return {
        "messages": [AIMessage(content=result.content)],
        "tenant_history": updated_history
    }