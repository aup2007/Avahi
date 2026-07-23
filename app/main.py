import json
import shutil
import sqlite3
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()  # read GROQ_API_KEY / VISION_MODEL from .env into the environment

from arch2_split import pipeline
from arch3_agent import intake as arch3_intake

from app import store

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "avahi.db"
if not DB_PATH.exists():
    DB_PATH = ROOT / "db" / "avahi.db"
UPLOADS_DIR = ROOT / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Avahi — Live Demo (Arch 2 · Arch 3)")


@app.on_event("startup")
def _init_schema() -> None:
    conn = _conn()
    try:
        store.ensure_schema(conn)
    finally:
        conn.close()


def _conn() -> sqlite3.Connection:
    # Read-write: the live-upload intake writes a new claim + damage rows.
    # check_same_thread=False: the SSE generator is iterated on a threadpool that
    # is not guaranteed to be the thread the connection was opened on.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _require_customer(conn: sqlite3.Connection, customer_id: str) -> None:
    if conn.execute("SELECT 1 FROM policies WHERE customer_id = ?", (customer_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"unknown customer_id {customer_id!r}")


def _save_photo(photo: UploadFile) -> Path:
    if not (photo.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file must be an image")
    UPLOADS_DIR.mkdir(exist_ok=True)
    suffix = Path(photo.filename or "").suffix or ".jpg"
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(photo.file, f)
    return dest


def _serialize(obj):
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@app.get("/api/customers")
def list_customers():
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT customer_id, car_class, policy_status, collision_active, comprehensive_active, "
            "collision_limit, comprehensive_limit, deductible, policy_data FROM policies ORDER BY customer_id"
        ).fetchall()
        out = []
        for r in rows:
            data = json.loads(r["policy_data"]) if r["policy_data"] else {}
            out.append({
                "customer_id": r["customer_id"],
                "name": data.get("name"),
                "car_class": r["car_class"],
                "policy_status": r["policy_status"],
                "collision_active": bool(r["collision_active"]),
                "comprehensive_active": bool(r["comprehensive_active"]),
                "collision_limit": r["collision_limit"],
                "comprehensive_limit": r["comprehensive_limit"],
                "deductible": r["deductible"],
            })
        return out
    finally:
        conn.close()


@app.post("/api/upload")
async def upload(
    customer_id: str = Form(...),
    photo: UploadFile = File(...),
    arch: str = Form("2"),
    claim_story: str = Form(""),
):
    if arch not in ("2", "3"):
        raise HTTPException(status_code=400, detail=f"unknown arch {arch!r}, expected '2' or '3'")

    conn = _conn()
    try:
        _require_customer(conn, customer_id)
        dest = _save_photo(photo)

        # Arch 2 is story-blind by design, so its story field is dropped here
        # rather than silently ignored downstream.
        story = claim_story.strip() or None
        try:
            if arch == "3":
                result = arch3_intake.run_upload(conn, customer_id, str(dest), story)
            else:
                result = pipeline.run_upload(conn, customer_id, str(dest))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"pipeline error: {e}")

        payload = _serialize(result)
        store.record(conn, arch, payload)
        return {"arch": arch, "result": payload}
    finally:
        conn.close()


@app.post("/api/upload/stream")
async def upload_stream(
    customer_id: str = Form(...),
    photo: UploadFile = File(...),
    claim_story: str = Form(""),
):
    # Arch 3 only. The agent makes ~10 sequential LLM calls, so the trace is
    # pushed step-by-step as SSE rather than making the user wait for all of it.
    conn = _conn()
    try:
        _require_customer(conn, customer_id)
        dest = _save_photo(photo)
    except Exception:
        conn.close()
        raise

    def events():
        try:
            stream = arch3_intake.stream_upload(
                conn, customer_id, str(dest), claim_story.strip() or None
            )
            for kind, payload in stream:
                data = _serialize(payload) if kind == "result" else payload
                yield f"data: {json.dumps({'type': kind, 'payload': data})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'payload': {'detail': f'agent error: {e}'}})}\n\n"
        finally:
            conn.close()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Without this, a proxy buffering the response defeats the whole point.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Static frontend — mounted last so /api/* wins.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
