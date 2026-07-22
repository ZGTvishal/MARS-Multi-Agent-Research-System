from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class AgentState(TypedDict):
    query:str
    papers: list[dict]
    index_path: str
    chunks: list[str]
    retrieved_chunks: Annotated[dict[str, list[str]], merge_dicts]
    summary: Annotated[dict[str, str], merge_dicts]
    bertscore_f1: Annotated[dict[str, float], merge_dicts]
    reroute_count: Annotated[dict[str, int], merge_dicts]
    errors: Annotated[dict[str, dict], merge_dicts]   
    final_summary: Annotated[dict[str, str], merge_dicts]
    knowledge_graph: dict
    messages: Annotated[list, add_messages]



