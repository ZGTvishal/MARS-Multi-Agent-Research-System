from core.state import AgentState
from langgraph.types import Send, Command
from typing_extensions import TypedDict
from core.utils import get_index_path, split_text
import faiss
import os
from agents.indexing import _model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()



def dispatch(state: AgentState) -> list[Send]:
    k = 5
    dispatched_output = [Send("summarise", {"paper": p,
                      "k":k,
                      }) for p in state["papers"]]

    return dispatched_output

class PaperDict(TypedDict):
    title: str
    abstract: str
    url: str
    authors: list[str]
    year: int
    source: str


class SummariseInput(TypedDict):
    paper : PaperDict
    k : int



def summarise(state: SummariseInput) -> dict:
    # This function's node is an exception to the locked priniciple, as it is only invoked via Send with a custom payload and never via normal graph edge carrying the full AgentState.
    """
    Generates the summary of the papers by fetching paper path, individual title and abstract as query. Builds mapped chunks with top k chunks.
    Generates the summary with Gemma 4 31B model with standard prompt and facts based on the mapped chunks. 

    Args: 
        invoked exclusively via Send, hence typed against SummariseInput rather than AgentState
    
    Updates:
        AgentState with Summary and Retrived chunks of a particular entry ID 
    
    """
    index_path = get_index_path(state["paper"]["url"])
    index = faiss.read_index(index_path)
    query = f"{state["paper"]["title"]} {state['paper']['abstract']}"
    encode_query = _model.encode([query], convert_to_numpy=True)
    #top k chunks indices
    Dist, Idx = index.search(encode_query, state["k"])
    actual_chunks = split_text(state["paper"]["abstract"])

    # top k chunks
    mapped_chunk = [actual_chunks[i] for i in Idx[0][:state["k"]]]

    #llm call
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=os.getenv("GEMMA_API_KEY"),
        temperature=1.0
    )
    system_prompt = f"""
    You are a research assistant that will help generate detailed summary of a research paper.
    Your grounding context lies in this list of top {state['k']} chunks -> {mapped_chunk}
    """
    # message body
    messages = [
    (
        "system",system_prompt,
    ),
    ("human", query),
    ]

    #llm invokation
    raw_response = llm.invoke(messages)
    # content contains two part, 0th index with the reasoning(not usefull for our usecase), 1st index with the actual summary which is being retrived by content[-1].
    output_text = raw_response.content[-1] 

    return Command(update= {'summary': {state['paper']['url']:output_text}, 'retrieved_chunks': {state["paper"]['url']:mapped_chunk} }, goto= Send("validate", {'paper':state["paper"],'entry_id': state["paper"]["url"]}))


class ValidateInput(TypedDict):
    paper: PaperDict
    entry_id: str







