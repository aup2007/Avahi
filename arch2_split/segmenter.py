from dataclasses import dataclass

from arch2_split.vlm_client import call_vision_json

SYSTEM_PROMPT = (
    "You inspect a single photo of a car submitted with an insurance claim. "
    "List which panels/parts of the car are visible in frame (e.g. "
    "\"front bumper\", \"driver door\", \"left headlight\", \"rear windshield\"). "
    "This is for a human-readable audit log only -- it is not used to compute "
    "cost or coverage, so do not assess damage or severity here. "
    'Respond with strict JSON: {"panels": [{"panel": string, "confidence": float 0-1}]}.'
)

USER_PROMPT = "Which car panels are visible in this photo?"


@dataclass
class PanelObservation:
    panel: str
    confidence: float


def segment_panels(image_path: str) -> list[PanelObservation]:
    result = call_vision_json(image_path, SYSTEM_PROMPT, USER_PROMPT)
    return [
        PanelObservation(panel=str(p["panel"]), confidence=float(p["confidence"]))
        for p in result["panels"]
    ]
