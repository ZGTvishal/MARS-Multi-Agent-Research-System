import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

from main import run_pipeline

st.set_page_config(page_title="MARS — Multi-Agent Research System", layout="wide")

st.title("MARS: Multi-Agent Academic Research System")
st.caption("Automated literature review via a multi-agent pipeline (Crawler → Indexing → Summarisation → Validation → Knowledge Graph)")

query = st.text_input("Research topic", value="transformer architecture attention mechanism")
run = st.button("Run pipeline", type="primary")

if run and query.strip():
    with st.spinner("Running MARS pipeline — crawling, indexing, summarising, validating..."):
        result = run_pipeline(query)

    st.success(
        f"Done. {len(result['papers'])} papers crawled, "
        f"{len(result['final_summary'])} summaries validated, "
        f"{len(result['errors'])} flagged low-confidence."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Ranked papers & summaries")
        papers_by_url = {p["url"]: p for p in result["papers"]}

        for url, summary in result["final_summary"].items():
            paper = papers_by_url.get(url, {})
            score = result["bertscore_f1"].get(url)
            flagged = url in result["errors"]

            with st.expander(f"{'⚠️ ' if flagged else ''}{paper.get('title', url)}"):
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
                st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                if score is not None:
                    st.markdown(f"**BERTScore-F1:** {score:.3f}")
                if flagged:
                    st.warning(f"Low confidence: {result['errors'][url]['reason']}")
                st.write(summary)
                st.markdown(f"[View on arXiv]({url})")

    with col2:
        st.subheader("Knowledge graph")
        st.caption("Drag nodes, scroll to zoom, hover for full title")
        kg = result["knowledge_graph"]

        if kg.get("nodes"):
            G = nx.node_link_graph(kg)

            net = Network(
                height="600px",
                width="100%",
                bgcolor="#0e1117",
                font_color="white",
                notebook=False,
            )
            net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150, spring_strength=0.02)

            for node_id, attrs in G.nodes(data=True):
                title = attrs.get("title", node_id)
                short_label = title if len(title) <= 30 else title[:27] + "..."
                net.add_node(
                    node_id,
                    label=short_label,
                    title=title,
                    color="#4C72B0",
                    size=18,
                )

            max_weight = max((d.get("weight", 0.4) for _, _, d in G.edges(data=True)), default=1)
            for u, v, attrs in G.edges(data=True):
                w = attrs.get("weight", 0.4)
                net.add_edge(
                    u, v,
                    value=w,
                    width=1 + (w / max_weight) * 4,
                    title=f"similarity: {w:.3f}",
                    color="#888888",
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
                net.write_html(f.name, notebook=False)
                html_path = f.name

            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            components.html(html_content, height=620, scrolling=False)
        else:
            st.info("No graph edges — not enough validated papers or similarity too low.")
else:
    st.info("Enter a research topic and click Run pipeline.")