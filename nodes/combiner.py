from langchain_core.prompts import ChatPromptTemplate
from .agent_state import AgentState, llm
from langchain_core.messages import AIMessage

combiner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Объедини несколько ответов в один связный естественный текст.
     Сохрани всю полезную информацию, можешь сократить без потери качсетва.
     Не упоминай что ответы были от разных источников.
     Пиши как один человек. Сделай единый текст без переносов строк.
     """),
    ("human", "Объедини:\n\n{responses}")
])

combiner = combiner_prompt | llm

def combiner_node(state: AgentState) -> dict:
    number_of_messages = state["has_work_question"] + state["has_small_talk"]

    if number_of_messages == 0:
        return {"messages": [AIMessage(content="Жду новых вопросов")]}
    
    elif number_of_messages == 1:
        return {"messages": [AIMessage(content=state["messages"][-1].content)]}
    
    else:
        responses_text = "\n\n---\n\n".join([x.content for x in state["messages"][-2:]])
        result = combiner.invoke({"responses": responses_text})
        return {"messages": [AIMessage(content=result.content)]}