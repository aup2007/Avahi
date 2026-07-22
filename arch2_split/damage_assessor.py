from dataclasses import dataclass

from arch2_split.vlm_client import call_vision_json
from common.severity_map import DAMAGE_CATEGORIES, severity_for

SYSTEM_PROMPT = (
    "You inspect a single photo of a car submitted with an insurance claim. "
    "Identify every distinct damage instance visible. Each instance must be "
    f"categorized as exactly one of: {', '.join(DAMAGE_CATEGORIES)}. "
    "If there is no damage visible, return an empty list. "
    'Respond with strict JSON: {"instances": [{"damage_category": string, "confidence": float 0-1}]}.'
)

USER_PROMPT = "What damage instances are visible in this photo?"


@dataclass
class DamageInstance:
    damage_category: str
    severity: str
    operation: str
    confidence: float


def assess_damage(image_path: str) -> list[DamageInstance]:
    result = call_vision_json(image_path, SYSTEM_PROMPT, USER_PROMPT)
    instances = []
    for item in result["instances"]:
        damage_category = str(item["damage_category"])
        severity, operation = severity_for(damage_category)
        instances.append(DamageInstance(
            damage_category=damage_category,
            severity=severity,
            operation=operation,
            confidence=float(item["confidence"]),
        ))
    return instances
