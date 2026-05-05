from langchain_core.prompts import ChatPromptTemplate
from .agent_state import AgentState, llm
from rag.loader import load_vectorstore
from langchain_core.messages import AIMessage

vectorstore = load_vectorstore()

lawyer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Ты юрист по коммерческой недвижимости.
    Отвечай строго на основе предоставленных договоров.
     Если в договорах информаци нет, то так и скажи.


    Контекст из договоров:
    {context}
    """),
    ("human", "Вопрос пользователя: {query}")
    ])

lawyer = lawyer_prompt | llm

def lawyer_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    docs = vectorstore.similarity_search(last_message, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = lawyer.invoke({"query": last_message,
                            "context": context})

    return {
        "messages": [AIMessage(content=result.content)]
    }