import sqlite3
from typing import Iterator, Optional

from arch3_agent import agent
from arch3_agent.live_claim import create_live_claim
from arch3_agent.pev_state import Arch3Result


def run_upload(
    conn: sqlite3.Connection,
    customer_id: str,
    image_path: str,
    claim_story: Optional[str] = None,
) -> Arch3Result:
    # Live-upload entry point, mirroring arch2_split.pipeline.run_upload: the
    # uploaded photo *is* a new claim. Open the claim, let the PEV loop
    # investigate it, and let adjudicate persist what it perceived before pricing.
    # run_claim (the eval path) is untouched.
    claim_id = create_live_claim(conn, customer_id, image_path, claim_story)
    return agent.run_claim(
        conn, claim_id, image_path, claim_story, live_customer_id=customer_id
    )


def stream_upload(
    conn: sqlite3.Connection,
    customer_id: str,
    image_path: str,
    claim_story: Optional[str] = None,
) -> Iterator[tuple[str, object]]:
    # Streaming twin of run_upload: same work, but each node's step is yielded as
    # it lands so the UI can render the investigation in progress.
    claim_id = create_live_claim(conn, customer_id, image_path, claim_story)
    yield ("claim", {"claim_id": claim_id})
    yield from agent.stream_claim(
        conn, claim_id, image_path, claim_story, live_customer_id=customer_id
    )
