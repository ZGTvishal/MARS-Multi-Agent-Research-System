from bert_score import BERTScorer
print("loading scorer...")
scorer = BERTScorer(lang="en", device="cpu")
print("scorer loaded, device:", scorer.device)
print("scoring...")
P, R, F1 = scorer.score(["a test summary"], [["a test chunk", "another chunk"]])
print("scored:", F1.item())