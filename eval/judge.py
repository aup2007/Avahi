import json
import os
import time

from groq import BadRequestError, Groq, RateLimitError

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b"))
MAX_ATTEMPTS = 3

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT = (
    "You audit an automated insurance decision for internal consistency. You are given a "
    "policy, the decision a model produced, and the free-text reasoning it wrote to justify "
    "that decision.\n"
    "You are NOT judging whether the decision is correct -- you cannot know the true repair "
    "cost. Judge only whether the model's own reasoning supports its own output:\n"
    "  - Does the stated dollar amount match any arithmetic the reasoning describes?\n"
    "  - Does the reasoning cite the policy's real limit, deductible and status?\n"
    "  - Does the reasoning contradict the covered/not-covered flag?\n"
    "  - Does the reasoning show a derivation at all, or just assert a number?\n"
    "Respond with strict JSON and no other text:\n"
    '{"coherent": bool, "derivation_shown": bool, "contradictions": [string], '
    '"note": string}\n'
    'Set "coherent" false if the reasoning contradicts the output or cites policy figures '
    'that differ from the ones given. Keep "note" to one sentence.'
)


def _call(user_prompt: str, model: str) -> dict:
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=3000,
            )
            return json.loads(response.choices[0].message.content)
        except RateLimitError as e:
            last_err = e
            time.sleep(20 * (attempt + 1))
        except BadRequestError as e:
            if getattr(e, "code", None) != "json_validate_failed" and "json_validate_failed" not in str(e):
                raise
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"judge call failed after {MAX_ATTEMPTS} attempts: {last_err}")


def judge_reasoning(policy: dict, result, *, model: str = JUDGE_MODEL) -> dict:
    user_prompt = (
        f"POLICY\n"
        f"  status: {policy['policy_status']}\n"
        f"  collision: {'active' if policy['collision_active'] else 'inactive'}, "
        f"limit ${policy['collision_limit']:,.2f}\n"
        f"  comprehensive: {'active' if policy['comprehensive_active'] else 'inactive'}, "
        f"limit ${policy['comprehensive_limit']:,.2f}\n"
        f"  deductible: ${policy['deductible']:,.2f}\n\n"
        f"MODEL DECISION\n"
        f"  covered: {result.covered}\n"
        f"  coverage_type: {result.coverage_type}\n"
        f"  payout: {result.payout}\n\n"
        f"MODEL REASONING\n{result.reasoning}\n\n"
        "Audit the reasoning against the decision."
    )
    raw = _call(user_prompt, model)
    return {
        "coherent": bool(raw.get("coherent")),
        "derivation_shown": bool(raw.get("derivation_shown")),
        "contradictions": [str(c) for c in (raw.get("contradictions") or [])],
        "note": str(raw.get("note") or ""),
    }
