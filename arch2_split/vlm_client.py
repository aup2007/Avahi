import base64
import json
import os
from pathlib import Path

from groq import Groq

VISION_MODEL = "qwen/qwen3.6-27b"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _encode_image(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"


def call_vision_json(image_path: str, system_prompt: str, user_prompt: str, *, model: str = VISION_MODEL) -> dict:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)
