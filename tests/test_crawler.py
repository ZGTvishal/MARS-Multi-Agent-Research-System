import pytest
import arxiv
import datetime
from dataclasses import dataclass, field
from core.state import AgentState
from agents.crawler import crawler_agent


@dataclass
class FakeAuthor:
    name: str

@dataclass
class FakeResult:
    title: str
    summary: str
    published: datetime.datetime
    authors: list
    entry_id: str


def make_fake_papers(n=12, title="Test Paper", empty_fields=False):
    return [
        FakeResult(
            title="" if empty_fields else f"{title} {i}",
            summary="" if empty_fields else f"Abstract text {i}",
            published=datetime.datetime(2024, 1, i % 28 + 1),
            authors=[FakeAuthor(name="A. Author"), FakeAuthor(name="B. Author")],
            entry_id=f"http://arxiv.org/abs/{i:04d}.00000",
        )
        for i in range(n)
    ]


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
        "messages": [],
    }


@pytest.fixture
def mock_arxiv(monkeypatch):
    def _apply(fake_results):
        def mock_results(self, search):
            return iter(fake_results)
        monkeypatch.setattr(arxiv.Client, "results", mock_results)
    return _apply


def test_crawler_returns_minimum_papers(base_state, mock_arxiv):
    mock_arxiv(make_fake_papers(12))
    result = crawler_agent(base_state)
    assert len(result["papers"]) >= 10

def test_crawler_paper_schema(base_state, mock_arxiv):
    mock_arxiv(make_fake_papers(12))
    result = crawler_agent(base_state)
    required_keys = {"title", "abstract", "published", "authors", "url", "source"}
    for paper in result["papers"]:
        assert required_keys == set(paper.keys())

def test_crawler_field_transformation(base_state, mock_arxiv):
    mock_arxiv(make_fake_papers(10, title="Attention Is All You Need"))
    result = crawler_agent(base_state)
    paper = result["papers"][0]
    assert paper["title"] == "Attention Is All You Need 0"
    assert paper["published"] == "2024-01-01"
    assert paper["authors"] == "A. Author, B. Author"
    assert paper["source"] == "arxiv"

def test_crawler_raises_on_insufficient_papers(base_state, mock_arxiv):
    mock_arxiv(make_fake_papers(5))
    with pytest.raises(ValueError, match="Insufficient papers retrieved"):
        crawler_agent(base_state)

def test_crawler_raises_on_empty_results(base_state, mock_arxiv):
    mock_arxiv([])
    with pytest.raises(ValueError, match="Insufficient papers retrieved"):
        crawler_agent(base_state)