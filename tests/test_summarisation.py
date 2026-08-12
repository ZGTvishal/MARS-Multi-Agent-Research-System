import pytest
from core.state import AgentState
import agents.summarisation as summary_module
import torch


class FakeFaissIndex:
    def search(self, embedding, k):
        distances = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        indices = [[0, 1, 2, 3, 4]]

        return distances, indices

class FakeModel:
    def encode(self, texts, convert_to_numpy=True):
        return [[0.1, 0.2, 0.3, 0.4]]

class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invoked_messages = None

    def invoke(self, messages):
        self.invoked_messages = messages
        return FakeLLMResponse()

class FakeLLMResponse:
    content = [
        "Fake reasoning output",
        "This is the final generated research summary."
    ]

class FakeScorer:
    def __init__(self, score_value):
        self.score_value = score_value

    def score(self, candidates, references):
        return (
            torch.tensor([0.8]),
            torch.tensor([0.8]),
            torch.tensor([self.score_value]),
        )

class FakeNaNScorer:
    def score(self, candidates, references):
        nan = torch.tensor([float("nan")])

        return (
            torch.tensor([0.8]),
            torch.tensor([0.8]),
            nan,
        )

def test_dispatch_empty_papers(base_state):
    state = {**base_state, "papers": []}
    with pytest.raises(ValueError, match="No papers found"):
            summary_module.dispatch(state)


def test_dispatch_creates_summarise_sends(base_state):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    result = summary_module.dispatch(state)

    assert len(result) == len(state["papers"])

    for send, paper in zip(result, state["papers"]):
        assert send.node == "summarise"
        assert send.arg == {
            "paper": paper,
            "k": 5,
            "reroute_count": 0,
        }

def test_summarise_generates_summary_and_routes_to_validate(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state = {
        "paper": paper,
        "k": 5,
        "reroute_count": 0,
    }

    fake_index = FakeFaissIndex()
    fake_model = FakeModel()
    fake_llm = FakeLLM()

    # Mock index path
    monkeypatch.setattr(
        "agents.summarisation.get_index_path",
        lambda url: "/fake/index.faiss",
    )

    # Mock FAISS
    monkeypatch.setattr(
        "agents.summarisation.faiss.read_index",
        lambda path: fake_index,
    )

    # Mock embedding model
    monkeypatch.setattr(
        "agents.summarisation._model",
        fake_model,
    )

    # Mock text splitting
    fake_chunks = [
        "Chunk 0",
        "Chunk 1",
        "Chunk 2",
        "Chunk 3",
        "Chunk 4",
    ]

    monkeypatch.setattr(
        "agents.summarisation.split_text",
        lambda abstract: fake_chunks,
    )

    # Mock LLM
    monkeypatch.setattr(
        "agents.summarisation.ChatGoogleGenerativeAI",
        lambda **kwargs: fake_llm,
    )

    result = summary_module.summarise(state)

    assert result.update["summary"][paper["url"]] == (
    "This is the final generated research summary."
)

    assert result.goto.node == "validate"

    assert result.goto.arg == {
        "paper": paper,
        "summary": "This is the final generated research summary.",
        "retrieved_chunks": fake_chunks,
        "entry_id": paper["url"],
        "reroute_count": 0,
    }

def test_validate_passes_on_first_attempt(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Generated summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 0,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeScorer(0.80),
    )

    result = summary_module.validate(state_n)

    assert result == {
        "final_summary": {
            paper["url"]: "Generated summary"
        },
        "reroute_count": {
            paper["url"]: 0
        },
    }

def test_validate_reroutes_from_zero_to_one(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Poor summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 0,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeScorer(0.40),
    )

    result = summary_module.validate(state_n)

    assert result.update == {
        "reroute_count": {
            paper["url"]: 1
        }
    }

    assert result.goto.node == "summarise"

    assert result.goto.arg == {
        "paper": paper,
        "k": 3,
        "reroute_count": 1,
    }

def test_validate_reroutes_from_one_to_two(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Still poor summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 1,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeScorer(0.40),
    )

    result = summary_module.validate(state_n)

    assert result.update == {
        "reroute_count": {
            paper["url"]: 2
        }
    }

    assert result.goto.node == "summarise"

    assert result.goto.arg == {
        "paper": paper,
        "k": 3,
        "reroute_count": 2,
    }


def test_validate_stops_after_third_failed_attempt(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Final poor summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 2,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeScorer(0.40),
    )

    result = summary_module.validate(state_n)

    assert result["final_summary"] == {
        paper["url"]: "Final poor summary"
    }

    error = result["errors"][paper["url"]]

    assert error["reason"].startswith(
        "Low BERTScore.. The summary score is "
    )

    assert error["bertscore_f1"] == pytest.approx(0.40)

def test_validate_passes_on_last_attempt(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Generated summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 2,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeScorer(0.80),
    )

    result = summary_module.validate(state_n)

    assert result == {
        "final_summary": {
            paper["url"]: "Generated summary"
        },
        "reroute_count": {
            paper["url"]: 2
        },
    }



def test_validate_handles_nan_score(
    base_state,
    monkeypatch,
):
    state = {**base_state, "query": "transformer architecture attention mechanism",
        "papers": [
            {
                "title": "Example Paper 1",
                "abstract": (
                    "This paper presents a detailed study of transformer architectures "
                    "and their use of attention mechanisms in natural language processing. "
                    "The authors investigate how self-attention allows models to capture "
                    "relationships between tokens without relying on recurrent computation. "
                    "The paper evaluates the proposed approach on several benchmark datasets "
                    "and reports improvements over traditional sequence modelling methods. "
                    "The results demonstrate that attention-based architectures can efficiently "
                    "model long-range dependencies while enabling parallel computation during "
                    "training. The authors also discuss limitations and possible future work."
                ),
                "url": "https://arxiv.org/abs/1234.5678",
                "authors": ["Author One", "Author Two"],
                "year": 2024,
                "source": "arxiv",
            },
            {
                "title": "Example Paper 2",
                "abstract": (
                    "This paper examines modern neural network architectures for language "
                    "understanding and focuses specifically on attention-based representation "
                    "learning. The authors compare several transformer configurations and "
                    "analyse their performance across multiple natural language processing "
                    "tasks. Their experiments show that contextual representations produced "
                    "by self-attention can improve both accuracy and generalisation. The study "
                    "also investigates computational costs and identifies situations where "
                    "larger models provide diminishing returns. The findings provide useful "
                    "guidance for designing efficient transformer systems and suggest several "
                    "directions for future research in scalable language modelling."
                ),
                "url": "https://arxiv.org/abs/5678.1234",
                "authors": ["Author Three"],
                "year": 2023,
                "source": "arxiv",
            },
        ]}
    paper = state["papers"][0]

    state_n = {
        "paper": paper,
        "entry_id": paper["url"],
        "summary": "Generated summary",
        "retrieved_chunks": [
            "Chunk 1",
            "Chunk 2",
        ],
        "reroute_count": 0,
    }

    monkeypatch.setattr(
        "agents.summarisation.scorer",
        FakeNaNScorer(),
    )

    result = summary_module.validate(state_n)

    assert result["final_summary"] == {
        paper["url"]: "Generated summary"
    }

    assert result["errors"][paper["url"]]["reason"] == (
        "BERTScore F1 returned NaN"
    )

    assert torch.isnan(
        torch.tensor(result["errors"][paper["url"]]["bertscore_f1"])
    )