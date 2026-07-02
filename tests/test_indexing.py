import pytest
from core.state import AgentState
from agents.indexing import indexing_agent
import datetime

@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "transformer architecture attention mechanism",
        "papers": [],
        "index_path": "",
        "chunks": [],
        "retrieved_chunks": [],
        "summary": "",
        "bertscore_f1": 0.0,
        "reroute_count": 0,
        "final_summary": "",
        "knowledge_graph": {},
        "messages": []
    }

