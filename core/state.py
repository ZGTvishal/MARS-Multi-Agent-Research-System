from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query:str
    papers: list[dict]
    index_path: str
    chunks: list[str]
    retrieved_chunks: dict[str, list[str]]
    summary: dict[str, str]
    bertscore_f1: dict[str, float]
    reroute_count: dict[str, int]
    final_summary: dict[str, str]
    knowledge_graph: dict
    messages: Annotated[list, add_messages]