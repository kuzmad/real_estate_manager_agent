from langchain_core.messages import SystemMessage, HumanMessage
from llm import llm
from langchain_core.messages import BaseMessage

def summarize_history(messages: list) -> str:
    if len(messages) > 0:
        if isinstance(messages[0], BaseMessage):
                messages = [f"{"Human" if isinstance(message, HumanMessage) else "Agent"}: {message.content}"
                            for message in messages]
        all_text = "\n".join(messages)
        print(all_text)
        response = llm.invoke([
            SystemMessage(content="Сожми диалог, выделив ключевые события и договоренности"),
            HumanMessage(content=f"Суммаризируй:\n{all_text}")
        ])
        return response.content
    else:
        return messages