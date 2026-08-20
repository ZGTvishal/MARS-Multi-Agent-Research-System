import numpy as np
import pytest

from agents.knowledge_graph import knowledge_graph_agent


def test_knowledge_graph_no_validated_papers_raises(base_state):
    base_state["papers"] = []
    base_state["final_summary"] = {}
    base_state["errors"] = {}

    with pytest.raises(ValueError, match="No summary found"):
        knowledge_graph_agent(base_state)


def test_knowledge_graph_single_paper(base_state):
    base_state["papers"] = [
        {
            "url": "url1",
            "title": "Paper One",
        }
    ]

    base_state["final_summary"] = {
        "url1": "Summary one",
    }

    result = knowledge_graph_agent(base_state)

    graph = result["knowledge_graph"]

    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 0

    assert graph["nodes"][0]["id"] == "url1"
    assert graph["nodes"][0]["title"] == "Paper One"


def test_knowledge_graph_edges_above_floor(
    base_state,
    monkeypatch,
):
    base_state["papers"] = [
        {"url": "url1", "title": "Paper One"},
        {"url": "url2", "title": "Paper Two"},
        {"url": "url3", "title": "Paper Three"},
        {"url": "url4", "title": "Paper Four"},
        {"url": "url5", "title": "Paper Five"},
    ]

    base_state["final_summary"] = {
        "url1": "Summary one",
        "url2": "Summary two",
        "url3": "Summary three",
        "url4": "Summary four",
        "url5": "Summary five",
    }

    similarity_matrix = np.array([
        [1.00, 0.70, 0.30, 0.50, 0.20],
        [0.70, 1.00, 0.60, 0.35, 0.10],
        [0.30, 0.60, 1.00, 0.80, 0.25],
        [0.50, 0.35, 0.80, 1.00, 0.45],
        [0.20, 0.10, 0.25, 0.45, 1.00],
    ])

    monkeypatch.setattr(
        "agents.knowledge_graph._model.encode",
        lambda summaries, convert_to_numpy=True: np.array(
            [[1, 0] for _ in summaries]
        ),
    )

    monkeypatch.setattr(
        "agents.knowledge_graph.cosine_similarity",
        lambda embeddings: similarity_matrix,
    )

    result = knowledge_graph_agent(base_state)

    graph = result["knowledge_graph"]

    assert len(graph["nodes"]) == 5

    expected_edges = {
        frozenset(("url1", "url2")),
        frozenset(("url1", "url4")),
        frozenset(("url2", "url3")),
        frozenset(("url3", "url4")),
        frozenset(("url4", "url5")),
    }

    actual_edges = {
        frozenset((edge["source"], edge["target"]))
        for edge in graph["edges"]
    }

    assert actual_edges == expected_edges

    for edge in graph["edges"]:
        assert edge["source"] != edge["target"]

    expected_weights = {
        frozenset(("url1", "url2")): 0.70,
        frozenset(("url1", "url4")): 0.50,
        frozenset(("url2", "url3")): 0.60,
        frozenset(("url3", "url4")): 0.80,
        frozenset(("url4", "url5")): 0.45,
    }

    actual_weights = {
        frozenset((edge["source"], edge["target"])): edge["weight"]
        for edge in graph["edges"]
    }

    assert actual_weights == expected_weights


def test_knowledge_graph_excludes_error_papers(
    base_state,
    monkeypatch,
):
    base_state["papers"] = [
        {"url": "url1", "title": "Paper One"},
        {"url": "url2", "title": "Paper Two"},
        {"url": "url3", "title": "Paper Three"},
    ]

    base_state["final_summary"] = {
        "url1": "Summary one",
        "url2": "Summary two",
        "url3": "Summary three",
    }

    base_state["errors"] = {
        "url2": {
            "error": "Failed to summarise paper"
        }
    }

    # Only url1 and url3 survive filtering.
    similarity_matrix = np.array([
        [1.00, 0.80],
        [0.80, 1.00],
    ])

    monkeypatch.setattr(
        "agents.knowledge_graph._model.encode",
        lambda summaries, convert_to_numpy=True: np.array(
            [[1, 0] for _ in summaries]
        ),
    )

    monkeypatch.setattr(
        "agents.knowledge_graph.cosine_similarity",
        lambda embeddings: similarity_matrix,
    )

    result = knowledge_graph_agent(base_state)

    graph = result["knowledge_graph"]

    node_ids = {
        node["id"]
        for node in graph["nodes"]
    }

    assert node_ids == {"url1", "url3"}
    assert "url2" not in node_ids


def test_knowledge_graph_keeps_isolated_paper(
    base_state,
    monkeypatch,
):
    base_state["papers"] = [
        {"url": "url1", "title": "Paper One"},
        {"url": "url2", "title": "Paper Two"},
        {"url": "url3", "title": "Paper Three"},
    ]

    base_state["final_summary"] = {
        "url1": "Summary one",
        "url2": "Summary two",
        "url3": "Summary three",
    }

    similarity_matrix = np.array([
        [1.00, 0.30, 0.20],
        [0.30, 1.00, 0.35],
        [0.20, 0.35, 1.00],
    ])

    monkeypatch.setattr(
        "agents.knowledge_graph._model.encode",
        lambda summaries, convert_to_numpy=True: np.array(
            [[1, 0] for _ in summaries]
        ),
    )

    monkeypatch.setattr(
        "agents.knowledge_graph.cosine_similarity",
        lambda embeddings: similarity_matrix,
    )

    result = knowledge_graph_agent(base_state)

    graph = result["knowledge_graph"]

    node_ids = {
        node["id"]
        for node in graph["nodes"]
    }

    assert node_ids == {"url1", "url2", "url3"}

    assert graph["edges"] == []


def test_knowledge_graph_only_considers_top_three_neighbours(
    base_state,
    monkeypatch,
):
    base_state["papers"] = [
        {"url": "url1", "title": "Paper One"},
        {"url": "url2", "title": "Paper Two"},
        {"url": "url3", "title": "Paper Three"},
        {"url": "url4", "title": "Paper Four"},
        {"url": "url5", "title": "Paper Five"},
    ]

    base_state["final_summary"] = {
        "url1": "Summary one",
        "url2": "Summary two",
        "url3": "Summary three",
        "url4": "Summary four",
        "url5": "Summary five",
    }

    similarity_matrix = np.array([
        [1.00, 0.90, 0.80, 0.70, 0.20],
        [0.90, 1.00, 0.10, 0.10, 0.10],
        [0.80, 0.10, 1.00, 0.10, 0.10],
        [0.70, 0.10, 0.10, 1.00, 0.10],
        [0.20, 0.10, 0.10, 0.10, 1.00],
    ])

    monkeypatch.setattr(
        "agents.knowledge_graph._model.encode",
        lambda summaries, convert_to_numpy=True: np.array(
            [[1, 0] for _ in summaries]
        ),
    )

    monkeypatch.setattr(
        "agents.knowledge_graph.cosine_similarity",
        lambda embeddings: similarity_matrix,
    )

    result = knowledge_graph_agent(base_state)

    graph = result["knowledge_graph"]

    actual_edges = {
        frozenset((edge["source"], edge["target"]))
        for edge in graph["edges"]
    }

    # url1's top 3 neighbours are url2, url3, url4.
    # url5 has similarity 0.20 and is not selected.
    assert frozenset(("url1", "url2")) in actual_edges
    assert frozenset(("url1", "url3")) in actual_edges
    assert frozenset(("url1", "url4")) in actual_edges
    assert frozenset(("url1", "url5")) not in actual_edges