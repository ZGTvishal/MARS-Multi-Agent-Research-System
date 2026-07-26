from core.state import AgentState
from langgraph.types import Send

def dispatch(state: AgentState) -> list[Send]:
    k = 5
    dispatched_output = [Send("summarise", {"paper": p,
                      "k":k,
                      }) for p in state["papers"]]

    return dispatched_output