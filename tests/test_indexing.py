import pytest
from core.state import AgentState
import agents.indexing as indexing_module
import datetime
import os
import faiss


@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "transformer architecture attention mechanism",
        "papers": [{
            "title":"Example paper 1",
            "abstract":"example abstract of paper 1",
            "published": "2024/1/28",
            "authors": "Fake Author 1",
            "entry_id":f"http://arxiv.org/abs/2.00000"},
            {
            "title":"Example paper 2",
            "abstract":"example abstract of paper 2",
            "published": "2024/2/28",
            "authors": "Fake Author 2",
            "entry_id":f"http://arxiv.org/abs/3.00000"
            }],
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


@pytest.fixture
def isolated_index_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        indexing_module,
        "_part_dir",
        str(tmp_path)
    )


# @pytest.fixture
# def mock_index(monkeypatch)



def test_index_raises_on_empty_papers(base_state):
    state = {**base_state, "papers": []}
    with pytest.raises(ValueError, match="No papers found"):
        indexing_module.indexing_agent(state)



def test_index_state_keys(base_state, isolated_index_dir):
    required_keys = {"chunks", "index_path"}
    result = indexing_module.indexing_agent(base_state)
    assert required_keys == set(result.keys())



def test_index_chunk_vs_paper_length(base_state,isolated_index_dir):
    result = indexing_module.indexing_agent(base_state)
    nos_paper = len(base_state["papers"])
    assert nos_paper == len(result["chunks"])

# Droped test, due to redundancy
# def test_index_non_empty_chunks(base_state, isolated_index_dir):
#     result = indexing_module.indexing_agent(base_state)
#     assert len(result["chunks"]) != 0






def test_index_chunk_matches_specs(base_state, isolated_index_dir):
    result = indexing_module.indexing_agent(base_state)
    required_specs = ["Title:", "\nAbstract:"]
    for c in result["chunks"]:
        assert c.startswith(required_specs[0])
        assert required_specs[1] in c
    



def test_index_file_exits(base_state, isolated_index_dir):
    result = indexing_module.indexing_agent(base_state)
    assert os.path.exists(result["index_path"])
    



def test_index_indexfile_vs_len_of_chunks(base_state, isolated_index_dir):
    result = indexing_module.indexing_agent(base_state)
    i = faiss.read_index(result["index_path"])
    assert i.ntotal == len(result["chunks"])




def test_index_chunk_vs_paper_length(base_state, isolated_index_dir):
    pass