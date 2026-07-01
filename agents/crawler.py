from core.state import AgentState
from langchain_community.retrievers import ArxivRetriever

def crawler_agent(state: AgentState) -> AgentState:
    retriever = ArxivRetriever(load_max_docs = 10, get_full_documents = False)
    docs = retriever.invoke(state["query"])

    if len(docs) < 10:
        raise ValueError(f"Insufficient papers retrieved: {len(docs)}/10 minimum")


    papers = []
    for doc in docs:
        papers.append({
            'title':doc.metadata.get("Title",""),
            "abstract": doc.metadata.get("Summary", ""),
            "published":doc.metadata.get("Published", ""),
            "authors":doc.metadata.get("Authors", ""),
            "url": doc.metadata.get("Entry ID", ""),
            "source": "arxiv"


        })
    
    return {"paper": papers}




    
    