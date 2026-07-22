from dataclasses import dataclass

from arch2_split.vlm_client import call_vision_json

SYSTEM_PROMPT = (
    "You inspect a single photo submitted with a car insurance claim. "
    "Decide only whether the photo is usable for damage assessment: is it "
    "actually a car, in focus, and well-lit enough to see its condition? "
    "Do not describe or assess any damage. "
    'Respond with strict JSON: {"valid": bool, "reason": string, "confidence": float 0-1}. '
    '"reason" should be a short phrase, e.g. "clear photo of a car" or "blurry, retake".'
)

USER_PROMPT = "Is this photo valid for a damage claim?"


@dataclass
class GatekeeperResult:
    valid: bool
    reason: str
    confidence: float


def check_photo(image_path: str) -> GatekeeperResult:
    result = call_vision_json(image_path, SYSTEM_PROMPT, USER_PROMPT)
    return GatekeeperResult(
        valid=bool(result["valid"]),
        reason=str(result["reason"]),
        confidence=float(result["confidence"]),
    )
