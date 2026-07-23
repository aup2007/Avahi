import sqlite3
from typing import Iterator, Optional

from langgraph.graph import END, START, StateGraph

from arch3_agent import pev_nodes
from arch3_agent.pev_state import Arch3Result, PEVState, traj
from arch3_agent.schemas import EvidencePackage

# Loop is planner -> executor -> verifier -> (continue) planner. Each iteration
# is 3 node visits; MAX_TOOL_CALLS=8 bounds iterations, so this is headroom.
RECURSION_LIMIT = 60


def _build_graph():
    b = StateGraph(PEVState)
    b.add_node("planner", pev_nodes.planner_node)
    b.add_node("executor", pev_nodes.executor_node)
    b.add_node("verifier", pev_nodes.verifier_node)
    b.add_node("adjudicate", pev_nodes.adjudicate_node)
    b.add_node("escalate", pev_nodes.escalate_terminal_node)

    b.add_edge(START, "planner")
    b.add_conditional_edges("planner", pev_nodes.route_after_plan,
                            {"executor": "executor", "adjudicate": "adjudicate"})
    b.add_edge("executor", "verifier")
    b.add_conditional_edges("verifier", pev_nodes.route_after_verify,
                            {"planner": "planner", "adjudicate": "adjudicate", "escalate": "escalate"})
    b.add_edge("adjudicate", END)
    b.add_edge("escalate", END)
    return b.compile()


# Compiled once; the graph is stateless (conn is passed per-run in the state).
GRAPH = _build_graph()


def _result_from_state(claim_id: str, image_path: str, final: dict) -> Arch3Result:
    evidence = final.get("evidence") or EvidencePackage()
    return Arch3Result(
        claim_id=claim_id, image_path=image_path,
        route=final.get("route", "escalate"),
        payout=final.get("payout"),
        deductible_applied=final.get("deductible_applied"),
        confidence=final.get("confidence", 0.0),
        reasons=final.get("reasons", []),
        evidence=evidence.model_dump(mode="json"),
        tool_calls=final.get("tool_calls", 0),
        replans=final.get("replans", 0),
        trajectory=final.get("trajectory", []),
    )


def _invalid_input_result(claim_id: str, image_path: str) -> Arch3Result:
    return Arch3Result(
        claim_id=claim_id, image_path=image_path, route="escalate",
        payout=None, deductible_applied=None, confidence=0.0,
        reasons=["invalid_claim_input"], evidence={}, tool_calls=0, replans=0,
        trajectory=[traj("input", "invalid_claim_input")],
    )


def _initial_state(claim_id, image_path, claim_story, conn, live_customer_id) -> PEVState:
    return {
        "conn": conn, "claim_id": claim_id, "image_path": image_path,
        "claim_story": claim_story, "live_customer_id": live_customer_id,
        "evidence": EvidencePackage(), "workspace": {},
        "tool_calls": 0, "replans": 0, "reasons": [], "trajectory": [],
    }


def _config(claim_id: str) -> dict:
    # run_name + metadata label each claim's trace in LangSmith (auto-traced when
    # LANGSMITH_TRACING=true; a harmless no-op otherwise).
    return {
        "recursion_limit": RECURSION_LIMIT,
        "run_name": f"arch3-claim-{claim_id}",
        "metadata": {"claim_id": claim_id},
        "tags": ["arch3", "pev"],
    }


def stream_claim(
    conn: sqlite3.Connection,
    claim_id: str,
    image_path: str,
    claim_story: Optional[str] = None,
    live_customer_id: Optional[str] = None,
) -> Iterator[tuple[str, object]]:
    """Same run as run_claim, but yields each trajectory step the moment its node
    finishes, then ("result", Arch3Result). Lets the UI show the agent thinking
    instead of staring at a spinner for ~10 sequential LLM calls."""
    if not claim_id or not image_path:
        yield ("result", _invalid_input_result(claim_id, image_path))
        return

    final: dict = {}
    emitted = 0
    initial = _initial_state(claim_id, image_path, claim_story, conn, live_customer_id)
    # stream_mode="values" hands back the whole state after every node; the
    # trajectory only ever grows, so anything past `emitted` is newly done.
    for state in GRAPH.stream(initial, config=_config(claim_id), stream_mode="values"):
        final = state
        steps = state.get("trajectory", [])
        while emitted < len(steps):
            yield ("step", steps[emitted])
            emitted += 1
    yield ("result", _result_from_state(claim_id, image_path, final))


def run_claim(
    conn: sqlite3.Connection,
    claim_id: str,
    image_path: str,
    claim_story: Optional[str] = None,
    live_customer_id: Optional[str] = None,
) -> Arch3Result:
    if not claim_id or not image_path:
        return _invalid_input_result(claim_id, image_path)

    initial = _initial_state(claim_id, image_path, claim_story, conn, live_customer_id)
    final = GRAPH.invoke(initial, config=_config(claim_id))
    return _result_from_state(claim_id, image_path, final)
