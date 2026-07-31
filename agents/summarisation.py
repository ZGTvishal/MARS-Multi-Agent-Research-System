from core.state import AgentState
from langgraph.types import Send
from typing_extensions import TypedDict

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
    pass
