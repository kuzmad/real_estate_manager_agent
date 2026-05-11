from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from api.graph import build_graph
from settings import Settings

settings = Settings()

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

#Через lifespan загружаем один раз то, что будет постоянно в памяти
#до yield все, что будет в момент запуска приложения, после yield все, что после остановки
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string("memory.db") as checkpointer:
        app.state.graph = build_graph(checkpointer)
        print("✅ Граф инициализирован")
        yield
        print("🛑 Приложение остановлено")

app = FastAPI(title="Real Estate Agent API", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    graph = req.app.state.graph

    async def generate():
        config = {"configurable": {"thread_id": request.thread_id}}
        full_response = ""

        async for chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="messages"
        ):
            node = metadata.get("langgraph_node", "")
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if node == "combiner_node":
                    yield chunk.content
                elif node in {"lawyer_node", "financial_node",
                              "property_manager_node", "assistant_node"}:
                    full_response += chunk.content

        if full_response:
            yield full_response

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/state/{thread_id}")
async def get_state(thread_id: str, req: Request):
    graph = req.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        return {"error": "Стейт не найден"}

    values = state.values
    return {
        "last_route": determine_last_agent(values),
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