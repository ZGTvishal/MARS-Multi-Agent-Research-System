import pytest
from core.state import AgentState
import agents.indexing as indexing_module
import os
import faiss
import numpy as np
import core.utils as utils_module


def test_empty_paper_list(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "papers": []}
    with pytest.raises(ValueError, match="No papers found"):
            indexing_module.build_retrieval_indices(state)


def test_empty_abstract_in_one_paper(base_state, isolated_index_dir, mock_index):
     state = {**base_state, "query": "transformer architecture attention mechanism",

        "papers": [
            {
                "title": "Example paper 1",
                "abstract": 
                    "This study investigates attention mechanisms used within modern transformer architectures and evaluates their effectiveness for representing relationships between tokens in complex sequences. The research focuses on how self-attention assigns different importance values to elements of an input based on their contextual relationships. It further examines multi-head attention, positional information, and the interaction between encoder and decoder components. The authors discuss how these architectural features allow transformers to process sequences in parallel while retaining important contextual information. Results indicate that attention based architectures are capable of capturing both local and long-range dependencies and can achieve competitive performance on several language understanding and generation tasks.",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": 
                    "This paper examines the architecture of transformer models and the role of self-attention in processing sequential information. The study explains how attention allows a model to identify relationships between different tokens without relying entirely on recurrent processing. It discusses the use of query, key, and value representations to calculate attention weights and combine information from different positions in an input sequence. The paper also considers how multi-head attention enables transformer models to capture different types of relationships simultaneously. Experimental results demonstrate that transformer architectures can effectively model long-range dependencies and provide strong performance across a range of natural language processing tasks.",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
            {
            "title": "Paper with empty abstract",
            "abstract": "",
            "url": "http://example.com/empty",
            "authors": ["Author 3"],
            "year": 2024,
            "source": "arxiv",
        }
        ]}
     result = indexing_module.build_retrieval_indices(state)
     assert result["errors"] == {
        "http://example.com/empty": {
            "reason": "http://example.com/empty has an empty abstract"
        }
    }

def test_per_paper_index_exits(base_state, isolated_index_dir, mock_index):
    state = {**base_state, "query": "transformer architecture attention mechanism",

        "papers": [
            {
                "title": "Example paper 1",
                "abstract": 
                    "This study investigates attention mechanisms used within modern transformer architectures and evaluates their effectiveness for representing relationships between tokens in complex sequences. The research focuses on how self-attention assigns different importance values to elements of an input based on their contextual relationships. It further examines multi-head attention, positional information, and the interaction between encoder and decoder components. The authors discuss how these architectural features allow transformers to process sequences in parallel while retaining important contextual information. Results indicate that attention based architectures are capable of capturing both local and long-range dependencies and can achieve competitive performance on several language understanding and generation tasks.",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": 
                    "This paper examines the architecture of transformer models and the role of self-attention in processing sequential information. The study explains how attention allows a model to identify relationships between different tokens without relying entirely on recurrent processing. It discusses the use of query, key, and value representations to calculate attention weights and combine information from different positions in an input sequence. The paper also considers how multi-head attention enables transformer models to capture different types of relationships simultaneously. Experimental results demonstrate that transformer architectures can effectively model long-range dependencies and provide strong performance across a range of natural language processing tasks.",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
            {
            "title": "Paper with empty abstract",
            "abstract": "",
            "url": "http://example.com/empty",
            "authors": ["Author 3"],
            "year": 2024,
            "source": "arxiv",
        }
        ]}
    result = indexing_module.build_retrieval_indices(state)
    paper = state["papers"]

    for p in paper:
         if len(p['abstract']) != 0:
            expected_path = indexing_module.get_index_path(p['url'])
            assert os.path.exists(expected_path)
         
def test_index_vector_count(base_state, isolated_index_dir, mock_index):
     state = {**base_state, "query": "transformer architecture attention mechanism",

        "papers": [
            {
                "title": "Example paper 1",
                "abstract": 
                    "This study investigates attention mechanisms used within modern transformer architectures and evaluates their effectiveness for representing relationships between tokens in complex sequences. The research focuses on how self-attention assigns different importance values to elements of an input based on their contextual relationships. It further examines multi-head attention, positional information, and the interaction between encoder and decoder components. The authors discuss how these architectural features allow transformers to process sequences in parallel while retaining important contextual information. Results indicate that attention based architectures are capable of capturing both local and long-range dependencies and can achieve competitive performance on several language understanding and generation tasks.",
                "url": "http://arxiv.org/abs/2.00000",
                "authors": ["Fake Author 1"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example paper 2",
                "abstract": 
                    "This paper examines the architecture of transformer models and the role of self-attention in processing sequential information. The study explains how attention allows a model to identify relationships between different tokens without relying entirely on recurrent processing. It discusses the use of query, key, and value representations to calculate attention weights and combine information from different positions in an input sequence. The paper also considers how multi-head attention enables transformer models to capture different types of relationships simultaneously. Experimental results demonstrate that transformer architectures can effectively model long-range dependencies and provide strong performance across a range of natural language processing tasks.",
                "url": "http://arxiv.org/abs/3.00000",
                "authors": ["Fake Author 2"],
                "year": 2024,
                "source": "arxiv",
            },
            {
            "title": "Paper with empty abstract",
            "abstract": "",
            "url": "http://example.com/empty",
            "authors": ["Author 3"],
            "year": 2024,
            "source": "arxiv",
        }
        ]}
     result = indexing_module.build_retrieval_indices(state)
     paper = state["papers"]
     for p in paper:
        if len(p['abstract']) != 0:
            expected_path = indexing_module.get_index_path(p['url'])
            i = faiss.read_index(expected_path)
            assert i.ntotal == len(indexing_module.split_text(p["abstract"]))


        
            



    
    