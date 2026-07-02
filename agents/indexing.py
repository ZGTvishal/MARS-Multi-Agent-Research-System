import os
import hashlib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.state import AgentState

def indexing_agent(state: AgentState) -> dict:
    papers = state["papers"]
    query = state["query"]

    chunks = [f"{p['title']}, {p['abstract']}" for p in papers]


    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(chunks, convert_to_numpy= True)


    dim = embedding.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding)

    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    index_dir = f"indexes/{query_hash}"
    os.makedirs(index_dir, exist_ok=True)
    index_path = f"{index_dir}/faiss.index"
    faiss.write_index(index, index_path)

    return{
        "chunks": chunks,
        "index_path": index_path
    }