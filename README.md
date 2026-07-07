# MARS — Multi-Agent Research Summarisation System

MARS is a multi-agent research summarisation system built on LangGraph's `StateGraph`, using a custom control-flow loop to coordinate retrieval, indexing, and summarisation agents with a confidence-based validation step to catch inter-agent hallucination.

> **Status:** Early implementation. Crawler and indexing agents are implemented and under active bugfixing/testing. Summarisation Agent, validation loop, and evaluation harness are not yet built.

## Architecture

Agents communicate through a shared `AgentState` and follow a strict node contract:

Every agent takes the full state and returns a partial state update — no side-channel mutation, no agent reaching into another agent's internals.

### Locked design decisions

These are fixed for this project and treated as constraints, not preferences:

- **Retrieval:** the `arxiv` Python package, used directly (no custom HTTP wrapper).
- **Indexing:** FAISS index, persisted to disk, keyed by the MD5 hash of the query string.
- **Validation loop:** BERTScore-F1 against the paper abstract, threshold **0.65**. On failure, re-route to the Summarisation Agent, capped at **2 re-routes** before falling through.
- **Node contract:** `(state: AgentState) -> dict` for every agent, no exceptions.

## Current implementation state

### Crawler Agent (`crawler.py`)
Wraps `arxiv.Client.results` to fetch candidate papers for a query.

### Indexing Agent (`indexing.py`)
Chunks paper metadata and builds/persists the FAISS index using `SentenceTransformer` embeddings.

**Known issues (identified in review, not all fixed yet):**

| Issue | Description |
|---|---|
| Chunk format mismatch | Spec requires `"Title: {title}\nAbstract: {abstract}"`; implementation was producing a comma-separated, unlabelled string. |
| Model re-instantiation | `SentenceTransformer` was being constructed inside the function on every call instead of once at module scope. |
| Broken cache check | `os.path.exists` check was testing a file the same function had just unconditionally written, so the "cache hit" path could never trigger. |
| Stray leading space | Leftover from an earlier fix to the chunk format string. |
| Return-path bug | Function returns `None` when the existence check fails, instead of the expected state dict — breaks the node contract. |
| Hardcoded `index_dir` | Set as a relative path inside `indexing_agent` with no injection point, which blocks tests from redirecting file output to a temp directory. **This is currently blocking `test_indexing.py`.** |

### Tests

- **`test_crawler.py`** — rewritten to mock `arxiv.Client.results` via a reusable `mock_arxiv` fixture and a `FakeResult` dataclass. No live network calls. Tests are scoped to transformation logic, not to `arxiv` library internals.
- **`test_indexing.py`** — **not yet written.** Blocked on the `index_dir` path-injection issue above. Once resolved, still need to work through:
  1. Which state keys the fixture actually needs.
  2. How to handle `_model` being instantiated at import time (affects the monkeypatching strategy).
  3. Which of the originally planned eight test behaviours actually hold once checked against the real implementation — some may be circular or test the wrong thing.

## Not yet implemented

- Summarisation Agent
- Validation/re-routing loop (BERTScore-F1 gate)
- LangGraph `StateGraph` wiring across all agents
- Evaluation harness (ROUGE, BERTScore, RAGAS faithfulness) and the single-agent / no-RAG / RAG comparison described in the project proposal
- UI (Streamlit dashboard mockup exists in the proposal only)

## Requirements

- Python 3.x
- `arxiv`
- `sentence-transformers`
- `faiss`
- `langgraph`
- `pytest` (monkeypatching + fixtures for test mocking)

## Development notes

- Implementation is evaluated against the locked spec above — deviations are bugs, not acceptable variation.
- Structural/blocking issues (e.g. the `index_dir` injection problem) are resolved before writing dependent tests, rather than working around them.