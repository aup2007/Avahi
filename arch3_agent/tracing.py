"""LangSmith tracing shim.

If langsmith is installed and LANGSMITH_TRACING=true, `traceable` is the real
decorator and calls are reported to LangSmith (raw prompt/response, latency,
tokens, trace id). If not, `traceable` is a no-op passthrough so the codebase
runs unchanged. LangGraph auto-traces the graph itself once the env vars are set;
this only enriches it with the Groq call details.

NOTE: LangSmith is an external SaaS -- fine here because the golden set is
synthetic (no real PII). Do not enable against real claimant data (see
arch3_realworld_review.md / SPEC privacy note).
"""

try:
    from langsmith import traceable  # type: ignore
    from langsmith.run_helpers import get_current_run_tree  # type: ignore

    _ENABLED = True
except Exception:  # langsmith not installed -> tracing becomes a no-op
    _ENABLED = False

    def traceable(*d_args, **d_kwargs):
        # Support both bare @traceable and @traceable(run_type=..., name=...).
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def _wrap(fn):
            return fn

        return _wrap

    def get_current_run_tree():  # type: ignore
        return None


def add_trace_metadata(meta: dict) -> None:
    """Best-effort attach (e.g. token usage) to the current run. Never raises --
    tracing must not be able to break a claim."""
    if not _ENABLED:
        return
    try:
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata(meta)
    except Exception:
        pass
