import base64
import io
import json
import os
import time

from groq import BadRequestError, Groq, RateLimitError
from PIL import Image

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")

# Groq free tier caps at ~8000 tokens/min (prompt image tokens + output). A
# full-res photo can blow the prompt budget on its own, so downscale the long
# edge to this before encoding. Images already at/under it are left untouched.
MAX_EDGE = 512

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path)
    # Normalize any format (AVIF/PNG-with-alpha/etc) to plain RGB JPEG so the
    # payload is small and the mime type is always correct.
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


MAX_ATTEMPTS = 4


def call_vision_json(image_path: str, system_prompt: str, user_prompt: str, *, model: str = VISION_MODEL) -> dict:
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
            ],
        },
    ]
    # The reasoning model is non-deterministic even at temperature 0: its <think>
    # block occasionally eats the whole output budget and returns no JSON
    # (json_validate_failed). A plain retry clears it. RateLimitError (the 8000
    # TPM cap) backs off longer since the window is per-minute.
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
    raise RuntimeError(f"vision call failed after {MAX_ATTEMPTS} attempts: {last_err}")
