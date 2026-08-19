import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.state import AgentState
import networkx as nx
from core.utils import get_paper_by_id


_model = SentenceTransformer("all-MiniLM-L6-v2")


def knowledge_graph_agent(state: AgentState) -> dict:
    """
    Generates a dict of Knowledge graph and updates the Agent State.

        Args: 
            Agent state

        Returns: 
            Knowledge graph dict.
        
        Raises:
            Value error when no paper summaries are found.
    
    """

    final_summary = state["final_summary"]
    final_summary_without_error_entries = []
    papers = state["papers"]

    for url, summary in final_summary.items():
        if url not in state.get('errors', {}):
            paper = get_paper_by_id(papers, url)
            title = paper['title']
            final_summary_without_error_entries.append((url, title, summary))

    if len(final_summary_without_error_entries) == 0:
        raise ValueError("No summary found...")

    urls, titles, summaries = map(list, zip(*final_summary_without_error_entries))

    if len(final_summary_without_error_entries) == 1:
        graph = nx.Graph()
        graph.add_node(
                        urls[0],
                        title=titles[0]
        )
        knowledge_graph = nx.node_link_data(graph)
        return {
            "knowledge_graph": knowledge_graph
        }

    embeddings = _model.encode(summaries, convert_to_numpy=True)

    similarity_matrix = cosine_similarity(embeddings)
    # floor=0.40, k=3: calibrated against a real 15-paper similarity matrix this session; see design-decisions notes
    graph = nx.Graph()
    for idx, url in enumerate(urls):
                graph.add_node(url,
                               title = titles[idx]
                               )
    for i in range(len(urls)):
        sorted_similarity_matrix = np.argsort(similarity_matrix[i])
        sorted_similarity_matrix = sorted_similarity_matrix[:-1]
        top_three_neighbours = sorted_similarity_matrix[-3:]
        for j in top_three_neighbours:
            similarity = similarity_matrix[i][j]

            if similarity >= 0.40:
                graph.add_edge(
                    urls[i],
                    urls[j],
                    weight=similarity
                )

    knowledge_graph = nx.node_link_data(graph)

    return {
        "knowledge_graph": knowledge_graph
    }

