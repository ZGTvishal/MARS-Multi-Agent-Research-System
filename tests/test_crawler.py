import pytest
import arxiv
from core.state import AgentState
from agents.crawler import crawler_agent
import datetime

@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "transformer architecture attention mechanism",
        "papers": [],
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

def test_crawler_returns_minimum_papers(base_state):
    result = crawler_agent(base_state)
    assert len(result["papers"]) >= 10

def test_crawler_paper_schema(base_state):
    result = crawler_agent(base_state)
    required_keys = {"title", "abstract", "published", "authors", "url", "source"}
    for paper in result["papers"]:
        assert required_keys == set(paper.keys())

def test_crawler_no_empty_fields(base_state):
    result = crawler_agent(base_state)
    for paper in result["papers"]:
        assert paper["title"], f"Empty title in paper: {paper}"
        assert paper["abstract"], f"Empty abstract in paper: {paper}"
        assert paper["url"], f"Empty URL in paper: {paper}"

def test_crawler_published_is_string(base_state):
    result = crawler_agent(base_state)
    for paper in result["papers"]:
        assert isinstance(paper["published"], str)

def test_crawler_source_is_arxiv(base_state):
    result = crawler_agent(base_state)
    for paper in result["papers"]:
        assert paper["source"] == "arxiv"

def test_crawler_raises_on_insufficient_papers(monkeypatch, base_state):
    def mock_results(self, search):
        return iter([])  
    monkeypatch.setattr(arxiv.Client, "results", mock_results)
    with pytest.raises(ValueError, match="Insufficient papers retrieved"):
        crawler_agent(base_state)