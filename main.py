from langgraph.graph import StateGraph, START
from core.state import AgentState
from agents.crawler import crawler_agent
from agents.indexing import indexing_agent, build_retrieval_indices
from agents.summarisation import dispatch, summarise, validate
from agents.knowledge_graph import knowledge_graph_agent


def build_graph():
    """
    Builds and compiles the MARS pipeline graph, covering Crawler -> Indexing
    (R5 + R7 index building) -> dispatch/Summarise/Validate fan-out loop.

    Knowledge Graph Agent is deliberately NOT wired as a graph edge. Verified
    empirically (toy graph, this session) that a plain add_edge off a
    Send-dispatched node with a variable-depth reroute loop fires once PER
    terminating branch, not once after all branches converge - LangGraph has
    no built-in fan-in for this dynamic case. Knowledge Graph Agent is run as
    an explicit second step in run_pipeline() against the fully-resolved
    final state instead.

    Similarly, no standalone Planner Agent node exists: R7's confidence-based
    coordination is implemented inside validate, and task ordering is
    implemented via the graph's own conditional/Send-based routing rather
    than a separate module.
    """
    g = StateGraph(AgentState)

    g.add_node("crawler", crawler_agent)
    g.add_node("indexing", indexing_agent)
    g.add_node("build_retrieval_indices", build_retrieval_indices)
    g.add_node("summarise", summarise)
    g.add_node("validate", validate)

    g.add_edge(START, "crawler")
    g.add_edge("crawler", "indexing")
    g.add_edge("indexing", "build_retrieval_indices")
    g.add_conditional_edges("build_retrieval_indices", dispatch)
    # validate's terminal branches return a plain dict with no outgoing edge
    # registered -> that branch of execution ends there (verified in toy graph).
    # validate's reroute branch uses Command(goto=Send("summarise", ...)) internally,
    # already handled without needing an add_edge here.

    return g.compile()


def initial_state(query: str) -> AgentState:
    return {
        "query": query,
        "papers": [],
        "chunks": [],
        "retrieved_chunks": {},
        "summary": {},
        "bertscore_f1": {},
        "reroute_count": {},
        "errors": {},
        "final_summary": {},
        "knowledge_graph": {},
        "messages": [],
    }


def run_pipeline(query: str) -> AgentState:
    graph = build_graph()
    result = graph.invoke(initial_state(query))

    # Knowledge Graph Agent runs as an explicit second step, not a graph edge -
    # see build_graph()'s docstring for why.
    kg_result = knowledge_graph_agent(result)
    result.update(kg_result)

    return result


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "transformer architecture attention mechanism"
    print(f"Running MARS pipeline for query: {query!r}\n")

    final_state = run_pipeline(query)

    print(f"\n{'='*60}")
    print(f"Papers found: {len(final_state['papers'])}")
    print(f"Summaries validated: {len(final_state['final_summary'])}")
    print(f"Errors (cap-exhausted): {len(final_state['errors'])}")
    print(f"{'='*60}\n")

    print("SAMPLE SUMMARIES\n")
    for i, (url, summary) in enumerate(final_state["final_summary"].items()):
        if i >= 3:
            break
        title = next((p["title"] for p in final_state["papers"] if p["url"] == url), url)
        score = final_state["bertscore_f1"].get(url, "N/A")
        print(f"[{i+1}] {title}")
        print(f"    BERTScore-F1: {score}")
        print(f"    {summary[:300]}...\n")

    kg = final_state["knowledge_graph"]
    print(f"KNOWLEDGE GRAPH — {len(kg.get('nodes', []))} nodes, {len(kg.get('edges', []))} edges\n")
    for edge in kg.get("edges", [])[:5]:
        src_title = next((n["title"] for n in kg["nodes"] if n["id"] == edge["source"]), edge["source"])
        tgt_title = next((n["title"] for n in kg["nodes"] if n["id"] == edge["target"]), edge["target"])
        print(f"  {src_title[:40]} <-> {tgt_title[:40]}  (similarity: {edge['weight']:.3f})")