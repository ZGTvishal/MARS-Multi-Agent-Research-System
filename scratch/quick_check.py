import faiss
from bert_score import BERTScorer

print("loading scorer...")
scorer = BERTScorer(lang="en", device="cpu")
print(scorer.device)
print("scorer loaded")

print("running dummy faiss op...")
index = faiss.IndexFlatL2(384)
import numpy as np
index.add(np.zeros((3, 384), dtype=np.float32))
print("faiss op done")

print("scoring...")
P, R, F1 = scorer.score(["a test summary"], [["a test chunk", "another chunk"]])
print("scored:", F1.item())