"""Tests for the Planner-Executor-Verifier agent (arch3_agent).

Everything is mocked -- no Groq, no DB, no images. We monkeypatch the check
registry (fake tools), the planner's LLM pick (deterministic), and the rules
engine (money), so the tests exercise the *graph and its bounds*, not the models.
"""
import pytest

from arch3_agent import agent, pev_nodes
from arch3_agent.checks import CHECK_REGISTRY as REAL_REGISTRY
from arch3_agent.pev_state import MAX_REPLANS, MAX_TOOL_CALLS
from arch3_agent.schemas import CheckName, EvidenceItem
from common.rules_engine import PayoutResult

C = CheckName


def _item(check, status="ok", **data):
    return EvidenceItem(check=check, status=status, data=data)


def _clean_registry():
    """All six checks pass cleanly."""
    return {
        C.IMAGE_QUALITY: lambda ctx: _item(C.IMAGE_QUALITY, valid=True, confidence=0.9),
        C.DAMAGE_TYPE: lambda ctx: _item(
            C.DAMAGE_TYPE, instances=[{"damage_category": "dent", "severity": "minor", "confidence": 0.9}], confidence=0.9
        ),
        C.STORY_CONSISTENCY: lambda ctx: _item(C.STORY_CONSISTENCY, consistent=True, reason_code="consistent", confidence=0.9),
        C.CLAIMED_PERIL: lambda ctx: _item(C.CLAIMED_PERIL, peril="collision", reason_code="collision", confidence=0.9),
        C.POLICY_COVERAGE: lambda ctx: _item(C.POLICY_COVERAGE, peril="collision", covered_for_peril=True),
        C.POLICY_DATES: lambda ctx: _item(C.POLICY_DATES, policy_status="active"),
    }


@pytest.fixture
def wired(monkeypatch):
    """Deterministic planner + a call-counting fake rules engine. Returns a
    helper that installs a given registry and runs a claim."""
    monkeypatch.setattr(pev_nodes, "_llm_pick_next", lambda state, candidates: candidates[0])

    calls = {"compute_payout": 0}

    def fake_payout(conn, claim_id, confidence):
        calls["compute_payout"] += 1
        return PayoutResult(route="auto_approve", payout=100.0, deductible_applied=50.0,
                            total_cost=150.0, covered_cost=150.0, reasons=[])

    monkeypatch.setattr(pev_nodes.rules_engine, "compute_payout", fake_payout)

    def run(registry, story="a normal claim story"):
        monkeypatch.setattr(pev_nodes, "CHECK_REGISTRY", registry)
        return agent.run_claim(conn=None, claim_id="CLM-TEST", image_path="img.jpg", claim_story=story)

    run.payout_calls = calls  # type: ignore[attr-defined]
    return run


def _executed(result):
    return [t["summary"].split(" -> ")[0] for t in result.trajectory if t["node"] == "executor"]


# 1. Different claims follow different tool paths -----------------------------

def test_different_claims_take_different_paths(wired):
    clean = wired(_clean_registry())

    bad_image = _clean_registry()
    bad_image[C.IMAGE_QUALITY] = lambda ctx: _item(C.IMAGE_QUALITY, "flagged", valid=False, confidence=0.3)
    blurry = wired(bad_image)

    assert _executed(clean) == ["image_quality", "damage_type", "story_consistency",
                                "claimed_peril", "policy_coverage", "policy_dates"]
    assert _executed(blurry) == ["image_quality"]      # escalates immediately
    assert _executed(clean) != _executed(blurry)


# 2. The agent can replan after new evidence ----------------------------------

def test_agent_replans_after_new_evidence(wired):
    seen = {"peril": 0}

    def flaky_peril(ctx):
        seen["peril"] += 1
        if seen["peril"] == 1:                          # first look: ambiguous
            return _item(C.CLAIMED_PERIL, "flagged", peril="unknown", reason_code="vague", confidence=0.4)
        return _item(C.CLAIMED_PERIL, "ok", peril="collision", reason_code="collision", confidence=0.9)

    reg = _clean_registry()
    reg[C.CLAIMED_PERIL] = flaky_peril
    result = wired(reg)

    assert seen["peril"] == 2                            # it re-investigated
    assert result.replans == 1
    assert result.route == "auto_approve"               # recovered, then completed


# 3. The agent stops within its limits ----------------------------------------

def test_agent_stops_within_limits(wired):
    def always_unknown(ctx):
        return _item(C.CLAIMED_PERIL, "flagged", peril="unknown", reason_code="vague", confidence=0.4)

    reg = _clean_registry()
    reg[C.CLAIMED_PERIL] = always_unknown
    result = wired(reg)

    assert result.route == "escalate"
    assert "unproductive_repeat" in result.reasons
    assert result.tool_calls <= MAX_TOOL_CALLS
    assert result.replans <= MAX_REPLANS


# 4. Unsafe or uncertain cases escalate ---------------------------------------

def test_prompt_injection_escalates(wired):
    reg = _clean_registry()
    reg[C.STORY_CONSISTENCY] = lambda ctx: _item(
        C.STORY_CONSISTENCY, "flagged", consistent=False, reason_code="possible_prompt_injection", confidence=1.0
    )
    result = wired(reg)
    assert result.route == "escalate"
    assert "possible_prompt_injection" in result.reasons


def test_tool_failure_escalates(wired):
    def boom(ctx):
        raise RuntimeError("vision service down")

    reg = _clean_registry()
    reg[C.DAMAGE_TYPE] = boom
    result = wired(reg)
    assert result.route == "escalate"
    assert "tool_failure" in result.reasons


def test_inconsistent_story_escalates(wired):
    reg = _clean_registry()
    reg[C.STORY_CONSISTENCY] = lambda ctx: _item(
        C.STORY_CONSISTENCY, "flagged", consistent=False, reason_code="story_damage_inconsistent", confidence=0.9
    )
    result = wired(reg)
    assert result.route == "escalate"
    assert "story_damage_inconsistent" in result.reasons


# 5. The agent cannot invoke payout or policy-modification actions -------------

def test_allowlist_has_no_payout_or_policy_write_tool():
    # The registry IS the allowlist. It contains exactly the 6 investigation
    # checks -- no pay/approve/deny/modify action is present.
    assert set(REAL_REGISTRY.keys()) == set(CheckName)
    forbidden = ("payout", "approve", "deny", "pay", "refund", "modify", "write", "update")
    for check in REAL_REGISTRY:
        assert not any(word in check.value for word in forbidden)


def test_escalation_never_touches_the_money(wired):
    # Money lives only in the adjudicate node, reached only on a clean complete.
    reg = _clean_registry()
    reg[C.IMAGE_QUALITY] = lambda ctx: _item(C.IMAGE_QUALITY, "flagged", valid=False, confidence=0.3)
    result = wired(reg)
    assert result.route == "escalate"
    assert wired.payout_calls["compute_payout"] == 0     # rules engine never ran


def test_clean_claim_reaches_money_exactly_once(wired):
    result = wired(_clean_registry())
    assert result.route == "auto_approve"
    assert wired.payout_calls["compute_payout"] == 1     # only via adjudicate
