import json
import os
import time

from groq import BadRequestError, Groq, RateLimitError

from arch3_agent.tracing import add_trace_metadata, traceable

# Same placeholder model Arch 2's vision stages use. The story-vs-damage judge
# is a text-only call (no image), so it lives here rather than reaching into
# arch2_split/vlm_client -- Arch 2 stays frozen; Arch 3 keeps its own entry point.
TEXT_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")

MAX_ATTEMPTS = 4

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


@traceable(run_type="llm", name="text_llm")
def call_text_json(system_prompt: str, user_prompt: str, *, model: str = TEXT_MODEL) -> dict:
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Mirrors the retry policy of arch2_split/vlm_client: the reasoning model can
    # burn its whole budget in <think> and return no JSON (json_validate_failed);
    # a plain retry clears it. RateLimitError (the 8000 TPM cap) backs off longer.
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
            usage = getattr(response, "usage", None)
            if usage is not None:
                add_trace_metadata(
                    {"model": model, "usage": usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)}
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
    raise RuntimeError(f"text call failed after {MAX_ATTEMPTS} attempts: {last_err}")
