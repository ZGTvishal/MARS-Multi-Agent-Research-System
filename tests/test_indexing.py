import pytest
from core.state import AgentState
import agents.indexing as indexing_module
import os
import faiss
import numpy as np
import core.utils as utils_module


def test_index_raises_on_empty_papers(base_state, isolated_index_dir):
    state_i = {**base_state, "papers": []}
    with pytest.raises(ValueError, match="No papers found"):
        indexing_module.indexing_agent(state_i)



def test_index_state_keys(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "query":"transformer architecture attention mechanism", 
         "papers": [
            {
                "title": "Example paper 1",
                "abstract": "example abstract of paper 1",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": "example abstract of paper 2",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
        ]}
    required_keys = {"chunks"}
    result = indexing_module.indexing_agent(state)
    assert required_keys == set(result.keys())



def test_index_chunk_vs_paper_length(base_state,isolated_index_dir, mock_index):
    state = {**base_state, "query":"transformer architecture attention mechanism", 
         "papers": [
            {
                "title": "Example paper 1",
                "abstract": "example abstract of paper 1",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": "example abstract of paper 2",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
        ]}
    result = indexing_module.indexing_agent(state)
    nos_paper = len(state["papers"])
    assert nos_paper == len(result["chunks"])


def test_index_chunk_matches_specs(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "query":"transformer architecture attention mechanism", 
         "papers": [
            {
                "title": "Example paper 1",
                "abstract": "example abstract of paper 1",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": "example abstract of paper 2",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
        ]}
    result = indexing_module.indexing_agent(state)
    required_specs = ["Title:", "\nAbstract:"]
    for c in result["chunks"]:
        assert c.startswith(required_specs[0])
        assert required_specs[1] in c
    



def test_index_file_exits(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "query":"transformer architecture attention mechanism", 
         "papers": [
            {
                "title": "Example paper 1",
                "abstract": "example abstract of paper 1",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": "example abstract of paper 2",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
        ]}
    result = indexing_module.indexing_agent(state)
    expected_path = indexing_module.get_corpus_index_path(state["query"])
    assert os.path.exists(expected_path)
    

def test_index_indexfile_vs_len_of_chunks(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "query":"transformer architecture attention mechanism", 
         "papers": [
            {
                "title": "Example paper 1",
                "abstract": "example abstract of paper 1",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": "example abstract of paper 2",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
        ]}
    result = indexing_module.indexing_agent(state)
    expected_path = indexing_module.get_corpus_index_path(state["query"])
    i = faiss.read_index(expected_path)
    assert i.ntotal == len(result["chunks"])

