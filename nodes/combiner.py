from langchain_core.prompts import ChatPromptTemplate
from .agent_state import AgentState, llm
from langchain_core.messages import AIMessage

combiner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Объедини несколько ответов в один связный естественный текст.
     Сохрани всю полезную информацию.
     Не упоминай что ответы были от разных источников.
     Пиши как один человек.
     """),
    ("human", "Объедини:\n\n{responses}")
])

combiner = combiner_prompt | llm

def combiner_node(state: AgentState) -> dict:
    partial_responses = state.get("partial_responses") or []

    if len(partial_responses) == 1:
        return {"messages": [AIMessage(content=partial_responses[0])]}

    responses_text = "\n\n---\n\n".join(partial_responses)
    result = combiner.invoke({"responses": responses_text})
    return {"messages": [AIMessage(content=result.content)]}