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
    
def determine_last_agent(values: dict) -> str:
    has_work = values.get("has_work_question", False)
    has_talk = values.get("has_small_talk", False)
    next_step = values.get("next_step")

    if has_work and has_talk:
        return f"combiner ({next_step} + assistant)"
    elif has_talk and not has_work:
        return "assistant"
    elif has_work and next_step:
        return next_step
    return "unknown"