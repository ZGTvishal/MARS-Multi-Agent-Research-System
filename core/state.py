from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query:str
    papers: list[dict]
    index_path: str
    chunks: list[str]
    retrieved_chunks: list[str]
    summary: str
    bertscore_f1: float
    reroute_count: int
    final_summary: str
    knowledge_graph: dict
    messages: Annotated[list, add_messages]