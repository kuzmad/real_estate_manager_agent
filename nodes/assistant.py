from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .agent_state import AgentState
from llm import llm
from utils import summarize_history
from settings import Settings

settings = Settings()

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

def assistant_node(state: AgentState) -> dict:
    messages = state["small_talks"]
    last_message = messages[-1].content

    history_text = "\n".join([
        f"{'Human' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in messages[:-1]
    ])

    result = assistant.invoke({"history": history_text, "query": last_message})

    ai_response = AIMessage(content=result.content)
    updated_talks = list(messages) + [ai_response]

    #суммаризация если нужно
    if len(updated_talks[:-1]) > settings.max_history_messages:
        summary = summarize_history(updated_talks[:-1])
        updated_talks = [
            SystemMessage(content=f"Сводка: {summary}"),
            updated_talks[-1]  # только последний обмен
        ]

    return {
        "small_talks": updated_talks, #явно прокидываем, так как состояние передали через Send
        "messages": [ai_response],
        "use_assistant": True
    }
