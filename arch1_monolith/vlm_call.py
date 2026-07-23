import base64
import io
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from groq import BadRequestError, Groq, RateLimitError
from PIL import Image

# Standalone by design (SPEC.md §3, Plan.md step 20): no imports from common/ or
# arch2_split/. The image encoder, Groq client and policy read below are
# re-implemented rather than shared -- if this architecture borrowed Arch 2's
# rules engine it would stop being the thing the comparison argues against.

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")
MAX_EDGE = 512
MAX_ATTEMPTS = 4

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


SYSTEM_PROMPT = (
    "You are an automated car-insurance claims adjuster. You are given one photo of a "
    "damaged vehicle, the customer's policy, and the story submitted with the claim. "
    "Decide the claim end to end: read the damage off the photo, estimate what the repair "
    "costs, judge whether the policy covers it, and state the exact dollar amount owed.\n"
    "Contract rules: the payout is capped at the limit of the applicable coverage, the "
    "deductible is subtracted once, and a lapsed policy or an inactive coverage pays nothing.\n"
    "Respond with strict JSON and no other text:\n"
    '{"covered": bool, "coverage_type": "collision"|"comprehensive"|"none", '
    '"payout": number or null, "reasoning": string}\n'
    'Set "payout" to the exact dollar amount owed (a single number, not a range) when '
    'covered, and null when not covered. Keep "reasoning" to two sentences.'
)


def _policy_text(policy: dict) -> str:
    data = policy.get("policy_data") or {}
    vehicle = data.get("vehicle") or {}
    return (
        f"Policyholder: {data.get('name', 'unknown')}\n"
        f"Policy number: {data.get('policy_number', 'unknown')}\n"
        f"Vehicle: {vehicle.get('year', '')} {vehicle.get('make', '')} {vehicle.get('model', '')} "
        f"(class: {policy['car_class']})\n"
        f"Policy status: {policy['policy_status']}\n"
        f"Collision coverage: {'active' if policy['collision_active'] else 'inactive'}, "
        f"limit ${policy['collision_limit']:,.2f}\n"
        f"Comprehensive coverage: {'active' if policy['comprehensive_active'] else 'inactive'}, "
        f"limit ${policy['comprehensive_limit']:,.2f}\n"
        f"Deductible: ${policy['deductible']:,.2f}"
    )


def _fetch_policy(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    row = conn.execute(
        "SELECT car_class, policy_status, collision_active, comprehensive_active, "
        "collision_limit, comprehensive_limit, deductible, policy_data "
        "FROM policies WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "customer_id": customer_id,
        "car_class": row[0],
        "policy_status": row[1],
        "collision_active": bool(row[2]),
        "comprehensive_active": bool(row[3]),
        "collision_limit": row[4],
        "comprehensive_limit": row[5],
        "deductible": row[6],
        "policy_data": json.loads(row[7]) if row[7] else {},
    }


def _fetch_claim(conn: sqlite3.Connection, claim_id: str) -> dict | None:
    row = conn.execute(
        "SELECT customer_id, photo_file, claim_story FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return None
    return {"customer_id": row[0], "photo_file": row[1], "claim_story": row[2]}


@dataclass
class MonolithResult:
    claim_id: str
    image_path: str
    covered: bool
    coverage_type: str
    payout: float | None
    reasoning: str
    route: str
    latency_s: float
    raw: dict = field(default_factory=dict)


def _to_route(covered: bool) -> str:
    # There is no escalate branch: the schema (SPEC.md:76) has no field for it, so
    # the monolith decides every claim including the ones whose golden route is
    # escalate. That over-deciding is the behaviour the eval is meant to expose.
    return "auto_approve" if covered else "auto_deny"


def _coerce_payout(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Despite the forced schema the model sometimes returns "$1,250.00" or
    # "1250 USD". Salvage the number instead of raising -- a malformed payout is
    # itself a data point for the reproducibility metric.
    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _call(image_path: str, user_prompt: str, model: str) -> dict:
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
            ],
        },
    ]
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=5000,
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
    raise RuntimeError(f"monolith call failed after {MAX_ATTEMPTS} attempts: {last_err}")


def _build_prompt(policy: dict, story: str) -> str:
    return (
        f"POLICY\n{_policy_text(policy)}\n\n"
        f"CLAIM STORY\n{story}\n\n"
        "The photo of the damage is attached. Decide the claim."
    )


def _result(claim_id: str, image_path: str, raw: dict, latency: float) -> MonolithResult:
    covered = bool(raw.get("covered"))
    return MonolithResult(
        claim_id=claim_id,
        image_path=image_path,
        covered=covered,
        coverage_type=str(raw.get("coverage_type") or "none"),
        payout=_coerce_payout(raw.get("payout")),
        reasoning=str(raw.get("reasoning") or ""),
        route=_to_route(covered),
        latency_s=latency,
        raw=raw,
    )


def decide(
    conn: sqlite3.Connection,
    claim_id: str,
    image_path: str,
    claim_story: str | None = None,
    *,
    model: str = VISION_MODEL,
) -> MonolithResult:
    claim = _fetch_claim(conn, claim_id)
    if claim is None:
        raise ValueError(f"unknown claim_id: {claim_id}")
    policy = _fetch_policy(conn, claim["customer_id"])
    if policy is None:
        raise ValueError(f"no policy for customer: {claim['customer_id']}")

    story = claim_story if claim_story is not None else (claim["claim_story"] or "(no story provided)")
    started = time.monotonic()
    raw = _call(image_path, _build_prompt(policy, story), model)
    return _result(claim_id, image_path, raw, time.monotonic() - started)


def decide_upload(
    conn: sqlite3.Connection,
    customer_id: str,
    image_path: str,
    claim_story: str | None = None,
    *,
    model: str = VISION_MODEL,
) -> MonolithResult:
    # Live-upload entry point mirroring arch2_split.pipeline.run_upload, so the app
    # can drive all three architectures. No claim or damage rows are written: the
    # monolith holds no intermediate state, which is the observability point.
    policy = _fetch_policy(conn, customer_id)
    if policy is None:
        raise ValueError(f"no policy for customer: {customer_id}")

    started = time.monotonic()
    raw = _call(image_path, _build_prompt(policy, claim_story or "(no story provided)"), model)
    return _result("live-" + uuid.uuid4().hex[:12], image_path, raw, time.monotonic() - started)
