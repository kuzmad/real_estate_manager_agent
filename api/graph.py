from nodes.agent_state import AgentState
from nodes.router import router_node, route_after_router
from nodes.lawyer import lawyer_node
from nodes.finance import financial_node
from nodes.property import property_manager_node, tenant_extractor_node
from nodes.assistant import assistant_node
from nodes.work_talk import work_small_talk_node, route_work_small_talk
from nodes.combiner import combiner_node
from langgraph.graph import StateGraph, START, END
from settings import Settings
from llm import llm

settings = Settings()

def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("work_small_talk_node", work_small_talk_node)
    builder.add_node("router_node", router_node)
    builder.add_node("lawyer_node", lawyer_node)
    builder.add_node("financial_node", financial_node)
    builder.add_node("tenant_extractor_node", tenant_extractor_node)
    builder.add_node("property_manager_node", property_manager_node)
    builder.add_node("assistant_node", assistant_node)
    builder.add_node("combiner_node", combiner_node)

    builder.add_edge(START, "work_small_talk_node")

    # Send запускает ноды параллельно
    builder.add_conditional_edges(
        "work_small_talk_node",
        route_work_small_talk,
        {
            "router_node": "router_node",
            "assistant_node": "assistant_node"
        }
    )

    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "lawyer_node": "lawyer_node",
            "financial_node": "financial_node",
            "property_manager_node": "tenant_extractor_node",
        }
    )

    builder.add_edge("tenant_extractor_node", "property_manager_node")

    builder.add_edge("lawyer_node", "combiner_node")
    builder.add_edge("financial_node", "combiner_node")
    builder.add_edge("property_manager_node", "combiner_node")
    builder.add_edge("assistant_node", "combiner_node")
    builder.add_edge("combiner_node", END)

    return builder.compile(checkpointer=checkpointer)