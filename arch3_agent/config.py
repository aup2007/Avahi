import os

# Role-based model routing: match model size to task difficulty x call frequency.
#   - Perception (VLM) is the vision call in arch2_split/vlm_client (VISION_MODEL).
#   - Orchestration (Planner + Verifier) is the loop's hot path -- called up to
#     MAX_TOOL_CALLS times per claim over a tiny allowlist -> a SMALL/fast model.
#   - Semantic judgment (story consistency, peril) is rare (1-2x/claim) but
#     nuanced and fraud-adjacent (adversarial) -> a LARGER model.
#
# All three default to the same placeholder so a first run works with one model;
# the split is a config change (env var), not a code change.
_DEFAULT_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")

# Small/fast: adaptive tool selection + continue/complete/escalate.
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", _DEFAULT_MODEL)

# Larger: natural-language judgment where reliability matters most.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", _DEFAULT_MODEL)
