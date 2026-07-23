import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arch2_split import pipeline
from common import cost_lookup, policy_lookup, rules_engine

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "avahi.db"
if not DB_PATH.exists():
    DB_PATH = ROOT / "db" / "avahi.db"
IMAGES_DIR = Path(__file__).resolve().parent / "web_images"
if not IMAGES_DIR.exists():
    IMAGES_DIR = ROOT / "web_images"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Avahi — Arch 2 Live Demo")


def _conn() -> sqlite3.Connection:
    # Read-only Option A path; a fresh connection per request keeps it thread-safe
    # under uvicorn without sharing a cursor across the pool.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _policy_view(policy: dict) -> dict:
    data = policy.get("policy_data") or {}
    return {
        "customer_id": policy["customer_id"],
        "car_class": policy["car_class"],
        "policy_status": policy["policy_status"],
        "collision_active": policy["collision_active"],
        "comprehensive_active": policy["comprehensive_active"],
        "collision_limit": policy["collision_limit"],
        "comprehensive_limit": policy["comprehensive_limit"],
        "deductible": policy["deductible"],
        "name": data.get("name"),
        "policy_number": data.get("policy_number"),
        "vehicle": data.get("vehicle"),
    }


def _truth_damage(conn: sqlite3.Connection, claim_id: str, car_class: str) -> list[dict]:
    costs = cost_lookup.claim_costs_by_coverage(conn, claim_id, car_class)
    return costs["instances"]


def _deterministic_preview(conn: sqlite3.Connection, claim_id: str) -> dict:
    # Offline route/payout at confidence 1.0 — no VLM, no token spend. Lets the
    # catalog show route variety before anyone runs the live pipeline.
    try:
        r = rules_engine.compute_payout(conn, claim_id, 1.0)
        return {"route": r.route, "payout": r.payout}
    except Exception:
        return {"route": None, "payout": None}


def _serialize(obj):
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@app.get("/api/claims")
def list_claims():
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT claim_id, customer_id, photo_file, claim_story, claim_date FROM claims ORDER BY claim_id"
        ).fetchall()
        out = []
        for row in rows:
            policy = policy_lookup.get_policy(conn, row["customer_id"])
            if policy is None:
                continue
            preview = _deterministic_preview(conn, row["claim_id"])
            out.append({
                "claim_id": row["claim_id"],
                "photo_file": row["photo_file"],
                "claim_story": row["claim_story"],
                "claim_date": row["claim_date"],
                "policy": _policy_view(policy),
                "preview_route": preview["route"],
                "preview_payout": preview["payout"],
            })
        return out
    finally:
        conn.close()


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: str):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT claim_id, customer_id, photo_file, claim_story, claim_date FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"claim {claim_id} not found")
        policy = policy_lookup.get_policy(conn, row["customer_id"])
        if policy is None:
            raise HTTPException(status_code=404, detail=f"no policy for claim {claim_id}")
        preview = _deterministic_preview(conn, row["claim_id"])
        return {
            "claim_id": row["claim_id"],
            "photo_file": row["photo_file"],
            "claim_story": row["claim_story"],
            "claim_date": row["claim_date"],
            "policy": _policy_view(policy),
            "truth_damage": _truth_damage(conn, row["claim_id"], policy["car_class"]),
            "preview": preview,
        }
    finally:
        conn.close()


@app.get("/api/image/{photo_file}")
def get_image(photo_file: str):
    # Guard against path traversal; only serve a bare filename from IMAGES_DIR.
    name = Path(photo_file).name
    path = IMAGES_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"image {name} not found")
    return FileResponse(path)


@app.post("/api/claims/{claim_id}/run")
def run_claim(claim_id: str):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT photo_file FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"claim {claim_id} not found")
        image_path = IMAGES_DIR / Path(row["photo_file"]).name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"image {row['photo_file']} not bundled")
        try:
            result = pipeline.run_claim(conn, claim_id, str(image_path))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"pipeline error: {e}")
        return _serialize(result)
    finally:
        conn.close()


# Static frontend — mounted last so /api/* wins.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
