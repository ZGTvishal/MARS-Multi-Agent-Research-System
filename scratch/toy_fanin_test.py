from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command

def merge_counts(l, r):
    return {**l, **r}

class State(TypedDict):
    papers: list[str]
    rounds_done: Annotated[dict[str, int], merge_counts]
    validated: Annotated[dict[str, bool], merge_counts]
    downstream_ran: Annotated[list[str], lambda l, r: l + r]

def dispatch(state: State):
    return [Send("summarise", {"paper": p, "round": 0}) for p in state["papers"]]

def summarise(state: dict) -> dict:
    paper = state["paper"]
    rnd = state["round"]
    print(f"[summarise] {paper} round {rnd}")
    return Command(
        update={},
        goto=Send("validate", {"paper": paper, "round": rnd})
    )

# paper B needs 2 rounds to "pass", A and C pass immediately
NEEDED_ROUNDS = {"A": 0, "B": 1, "C": 0}

def validate(state: dict) -> dict:
    paper = state["paper"]
    rnd = state["round"]
    print(f"[validate] {paper} round {rnd}")
    if rnd < NEEDED_ROUNDS[paper]:
        return Command(
            update={"rounds_done": {paper: rnd + 1}},
            goto=Send("summarise", {"paper": paper, "round": rnd + 1})
        )
    # terminal - plain dict return, falls through to normal edges
    return {"validated": {paper: True}, "rounds_done": {paper: rnd}}

def downstream(state: State) -> dict:
    print(f"[downstream] FIRED. validated so far = {state.get('validated', {})}")
    return {"downstream_ran": ["yes"]}

g = StateGraph(State)
g.add_node("summarise", summarise)
g.add_node("validate", validate)
g.add_node("downstream", downstream)
g.add_conditional_edges(START, dispatch)
# g.add_edge("validate", "downstream")
# g.add_edge("downstream", END)

graph = g.compile()

print("=== INVOKING ===")
result = graph.invoke({"papers": ["A", "B", "C"], "rounds_done": {}, "validated": {}, "downstream_ran": []})
print("=== FINAL RESULT ===")
print(result)
print(f"\ndownstream fired {len(result['downstream_ran'])} times")