from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .agent_state import AgentState, llm

assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Ты дрежулюбный ассистент по управлению коммерческой недвижимостью.
    Ты умеешь поддержать разговор на свободные темы и помочь с рабочими вопросами.
    Отвечай естественно и по-человечески.

    История диалога:
    {history}
    """),
    ("human", "Вопрос пользователя: {query}")
])

assistant = assistant_prompt | llm

def summarize_assistant_history(text: str) -> str:
    response = llm.invoke([
        SystemMessage(content="Сожми историю взаимодействия сохранив основную информаию и ключевые моменты"),
        HumanMessage(content=f"История:\n{text}")
    ])
    return response.content

def assistant_node(state: AgentState) -> dict:
    messages = state["small_talk"]
    last_message = messages[-1].content

    result = assistant.invoke({"history": messages[:-1], "query": last_message})

    return {
        "messages": [AIMessage(content=result.content)]
    }