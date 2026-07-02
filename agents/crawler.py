import arxiv
from core.state import AgentState

def crawler_agent(state: AgentState) -> dict:
    """
    Crawls into the arxiv database and fetchs up max 15 relevent papers for a given query. 
    
    Args:
        AgentState: A typeDict defining the input order for the function.
    
    Returns: 
        List of max 15 papers based on relevence of the user query
    
    Raises: 
        ValueError of the number od papers are less than 10
    
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=state["query"],
        max_results=15,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "published": str(result.published.date()),
            "authors": ", ".join(a.name for a in result.authors),
            "url": result.entry_id,
            "source": "arxiv"
        })

    if len(papers) < 10:
        raise ValueError(f"Insufficient papers retrieved: {len(papers)}/10 minimum")

    return {"papers": papers}