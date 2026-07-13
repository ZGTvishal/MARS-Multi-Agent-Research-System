import os
import hashlib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sentences = []

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode(sentences)
# dim = embedding.shape[1]
# index = faiss.IndexFlatL2(dim)
# index.add(embedding)

# index_dir = f"indexes/local_test"
# os.makedirs(index_dir, exist_ok=True)

# index_path = f"{index_dir}/faiss.index"
# faiss.write_index(index, index_path)

print(type(embedding))


"""<faiss.swigfaiss.IndexFlatL2; proxy of <Swig Object of type 'faiss::IndexFlatL2 *' at 0x14305ff70> >
"""
