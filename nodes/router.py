from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .agent_state import AgentState, llm

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ты роутер запросов пользователя по упарвлению коммерческой недвижимостью.

        Тебе нужно определить, к какому маршруту относится запрос пользователя:

        1. lawyer - если пользователь спрашвиает про актуальное законодательство, прописанные в договорах с арнедаторами пункты
           Примеры: "Какая ставка налогообложения для ИП на УСН?", "Когда заканчивается договор с Ozon?"

        2. financial - сложные рассуждения об оптимизации налогов с точки зрения математики, расчет доходности объекта
            Примеры: "Рассчитай налог за квартал", "Сравни два сценария консолидации"

        3. property_manager - если пользователь просит уточнить историю взаимодейтсаия с арендаторами: дата ремонта, даты поверки счетчиков, общие договоренности
            также пользователь может попросить зафиксировать какие-то договоренности
           Примеры: "Когда очередная поверка счетчиков в помещении Ozon", "Когда планируется ремонт в помещении WB", "Запомни: вчера была поверка счетчиков у WB"
        
        4. assistant - small talk, общие вопросы, все что не входит выше (fallback)
            Примеры: "Как дела?", "Что такое ЕГРН?"

        Верни только структуру согласно схеме.
        """
    ),
    (
        "human",
        "Запрос пользователя: {query}"
    )
])

class RouterSteps(str, Enum):
    LAWYER = "lawyer"
    FINANCIAL = "financial"
    PROPERTY_MANAGER = "property_manager"
    ASSISTANT = "assistant"

class RouteDecision(BaseModel):
    next_step: RouterSteps = Field(
        description="Тип пользовательского запроса"
    )
    reason: str = Field(
        description="Краткое объяснение, почему выбран этот маршрут"
    )

router = router_prompt | llm.with_structured_output(RouteDecision)

def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    result = router.invoke({"query": last_message})
    return {
        "next_step": result.next_step.value,
        "reason": result.reason
    }

# =========================================================
# Routing function
# =========================================================
def route_after_router(state: AgentState) -> str:
    next_step = state["next_step"]
    if next_step is None:
        raise ValueError(f"Unext_step is None")
    print(f"{next_step}_node")
    return f"{next_step}_node"