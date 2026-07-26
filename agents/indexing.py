import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.state import AgentState
from core.utils import get_corpus_index_path

_model = SentenceTransformer("all-MiniLM-L6-v2")

def indexing_agent(state: AgentState) -> dict:
    papers = state["papers"]

    if len(papers) == 0:
        raise ValueError("No papers found")

    chunks = [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
   
    embedding = _model.encode(chunks, convert_to_numpy= True)
    dim = embedding.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding)
    index_path = get_corpus_index_path(state["query"])
    faiss.write_index(index, index_path)

    if not os.path.exists(index_path):
        raise RuntimeError(f"Failed to write FAISS index to {index_path}")
    return{
        "chunks": chunks,
    }