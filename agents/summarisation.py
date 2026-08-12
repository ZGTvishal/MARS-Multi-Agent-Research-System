from core.state import AgentState
from langgraph.types import Send, Command
from typing_extensions import TypedDict
from core.utils import get_index_path, split_text
import faiss
import os
from agents.indexing import _model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from bert_score import BERTScorer

load_dotenv()



def dispatch(state: AgentState) -> list[Send]:
    k = 5
    if len(state["papers"]) == 0:
        raise ValueError("No papers found")
    dispatched_output = [Send("summarise", {"paper": p,
                      "k":k,
                      "reroute_count": 0
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
    reroute_count: int



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

    #LLM invokation
    raw_response = llm.invoke(messages)
    # Gemma 4 (31B) returns AIMessage.content as a list with two blocks, an undocumented shape distinct from both cases in the langchain-google-genai docs (Gemini 2.5-and-earlier: plain string; Gemini 3.x: single dict with a 'type' key). Verified across multiple real calls this session: block [0] is a reasoning/scratchpad (bullet fragments, no prose), block [-1] is the polished final answer (markdown headers, full paragraphs).usage_metadata confirms reasoning tokens are counted separately from output tokens (e.g. 565 reasoning vs 581 output vs 1388 total in one sample), supporting that block [0] is a distinct reasoning pass rather than a formatting quirk. `.text` (BaseMessage) does NOT handle this shape correctly — it concatenates both blocks, same as a naive join. Extraction here is positional (content[-1]), verified consistent across 3 real samples with no exceptions; if a future SDK/model update changes block order or count, this will silently break — no runtime guard exists.
    output_text = raw_response.content[-1]

    return Command(update= {'summary': {state['paper']['url']:output_text}, 'retrieved_chunks': {state["paper"]['url']:mapped_chunk} }, goto= Send("validate", {'paper':state["paper"],
    'summary': output_text,
    'retrieved_chunks': mapped_chunk,
    'entry_id': state["paper"]["url"],
    'reroute_count': state["reroute_count"]
    }))


class ValidateInput(TypedDict):
    paper: PaperDict
    entry_id: str
    summary: str
    retrieved_chunks: list[str]
    reroute_count: int

scorer = BERTScorer(lang="en")
def validate(state: ValidateInput) -> dict:
    """
        Produces BERTScore_F1 for every paper summary. Updates the AgentState on terminal cases (BERTScore = 0.65, Reroute_count >= 2).
        Reroutes to summarise function if terminal has not been achieved. 
    
        Args: 
            invoked exclusively via Send, hence typed against ValidateInput rather than AgentState
        
        Updates:
            AgentState with final_summary, reroute_count of, errors for every paper.  
        
    """
    P, R, F1 = scorer.score(
    [state["summary"]],
    [state["retrieved_chunks"]],
)
    
    F1_float = F1.item()
    if F1_float >= 0.65:
        return {'final_summary':{state['paper']['url']: state['summary']}, 'reroute_count': {state['entry_id']:state["reroute_count"]}}
    elif F1_float < 0.65 and state['reroute_count'] < 2:
        new_count = state['reroute_count'] + 1
        return Command(update ={'reroute_count':{state["entry_id"]: new_count}}, goto=Send("summarise", {"paper":state["paper"], "k":3, 'reroute_count':new_count }))
    elif state['reroute_count'] >= 2:
        return {'final_summary':{state['paper']['url']: state['summary']}, 'errors':{state["entry_id"]:{'reason':f"Low BERTScore.. The summary score is {F1_float}", "bertscore_f1":F1_float}}}
    else:
        return {
            'final_summary':{state['paper']['url']: state['summary']},
            'errors':{state["entry_id"]:{'reason': "BERTScore F1 returned NaN", "bertscore_f1":F1_float}}
        }