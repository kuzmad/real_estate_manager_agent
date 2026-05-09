from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver #тут версия с async
from api.graph import build_graph
import json

from settings import Settings

settings = Settings()

app = FastAPI(title="Real Estate Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    async def generate():
        full_response = ""
        
        async with AsyncSqliteSaver.from_conn_string("memory.db") as checkpointer:
            graph = build_graph(checkpointer)
            config = {"configurable": {"thread_id": request.thread_id}}

            async for chunk, metadata in graph.astream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    # Всегда стримим combiner
                    if node == "combiner_node":
                        yield chunk.content
                    # Если combiner не генерировал — стримим финальный агент
                    elif node in {"lawyer_node", "financial_node",
                                "property_manager_node", "assistant_node"}:
                        full_response += chunk.content  # накапливаем но не стримим

            # Если combiner ничего не стримил — отдаём накопленное
            if full_response:
                yield full_response

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """Возвращает метаданные после завершения — агент, арендатор, история"""
    async with AsyncSqliteSaver.from_conn_string("memory.db") as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.get_state(config)
        
        if not state or not state.values:
            return {"error": "Стейт не найден"}
        
        values = state.values
        return {
            "last_route": values.get("next_step"),
            "tenant": values.get("tenant"),
            "tenant_history": values.get("tenant_history", {}),
            "messages": [
                {
                    "role": "human" if m.__class__.__name__ == "HumanMessage" else "ai",
                    "content": m.content
                }
                for m in values.get("messages", [])
            ]
        }