from langchain.agents import create_agent
from .agent_state import AgentState
from langchain_core.tools import tool
from llm import llm

@tool
def calculate_quarterly_income(
        area: float,
        rate_per_sqm: float,
        months: int = 3
) -> str:
    """
    Рассчитывает доход от аренды за указанное количество месяцев
    area - площадь помещения в кв. метрах
    rate_per_sqm - ставка аренды в рублях за кавадратный метр в месяц
    months - количество месяц (по умолчанию 3 = квартал)
    """
    return f"Доход за {months} месяцев: {rate_per_sqm * area * months: ,.2f} рублей"

@tool
def calculate_usn_tax(
        income: float,
        rate: float = 6.0
) -> str:
    """
    Рассчитывает налог по УСН.
    income - доход в рублях
    rate - ставка УСН в процентах (по умолчанию 6%)
    """
    return f"Налог по УСН со ставкой {rate}% составит {income * rate / 100: ,.2f} рублей"

tools = [calculate_quarterly_income, calculate_usn_tax]

finance_aget_promt = """
Ты финансовый аналитик по коммерческой недвидимости
Используй доступные инструменты для точных расчетов
Показывай шаги рассуждений перед финальным ответом
"""

finance_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=finance_aget_promt
)

def financial_node(state: AgentState) -> dict:
    messages = state["work_questions"]
    last_message = messages[-1].content
    response = finance_agent.invoke({"messages": last_message})
    #print(response["messages"]) #провека вызовов тулов
    return {
       "messages": [response["messages"][-1]],
    }