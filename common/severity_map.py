SEVERITY_MAP = {
    "dent": ("minor", "repair"),
    "scratch": ("minor", "repair"),
    "crack": ("moderate", "repair"),
    "tire flat": ("moderate", "repair"),
    "lamp broken": ("severe", "replace"),
    "glass shatter": ("severe", "replace"),
}

DAMAGE_CATEGORIES = tuple(SEVERITY_MAP.keys())


def severity_for(damage_category: str) -> tuple[str, str]:
    try:
        return SEVERITY_MAP[damage_category]
    except KeyError:
        raise ValueError(f"unknown damage_category={damage_category!r}, expected one of {DAMAGE_CATEGORIES}")
