import pytest
from core.state import AgentState
import agents.indexing as indexing_module
import numpy as np
import core.utils as utils_module



@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "",
        "papers": [],
        "chunks": [],
        "retrieved_chunks": {},
        "summary": {},
        "bertscore_f1": {},
        "reroute_count": {},
        "errors": {},
        "final_summary": {},
        "knowledge_graph": {},
        "messages": [],
    }

@pytest.fixture
def isolated_index_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module, "_part_dir", str(tmp_path))

@pytest.fixture
def mock_index(monkeypatch):
    def fake_encode(chunks: list[str], embedding_dim: int = 384, convert_to_numpy=True):
        return np.zeros((len(chunks), embedding_dim), dtype=np.float32)
        
    monkeypatch.setattr(
        indexing_module._model,
        "encode",
        fake_encode)
    