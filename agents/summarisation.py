from core.state import AgentState
from langgraph.types import Send

def dispatch(state: AgentState) -> list[Send]:
    k = 5
    index_path = state["index_path"]
    dispatched_output = [Send("summarise", {"paper": p,
                      "k":k,
                      "index_path": index_path}) for p in state["papers"]]

    return dispatched_output