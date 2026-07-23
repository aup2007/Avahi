# Avahi — Deployment & Frontend Plan (Arch 2 Live Demo)

Status: **plan only — no code written yet.** This documents what will be built and why, for review before implementation.

## Decisions locked
- **Demo shape:** live upload. A visitor **picks a customer from a dropdown** (which loads that customer's real policy) and **uploads a damage photo**. That submission **creates a new claim**, the Arch 2 pipeline runs live (real Groq VLM calls) on the uploaded photo, and the UI shows the decision + payout + full stage-by-stage audit trail.
- **Host:** single managed host (Render / Railway / Fly), one service, Dockerized.

## The live-upload flow (customer + photo → new claim → run)
Uploading a photo **is filing a new claim.** So rather than special-casing the money code, intake writes a real claim and the normal pipeline runs on it:

1. **Customer dropdown** → selects a `customer_id` → loads their policy (car_class, coverages active, limits, deductible). The photo cannot supply the policy, so the customer brings it.
2. **Photo upload** → saved to an ephemeral `uploads/` dir → this is the damage of the new claim.
3. **Intake:** run gatekeeper → segmenter → damage_assessor **once** on the uploaded photo. The vision-detected damage becomes the claim's `claim_damage_instances` (damage_category from vision; severity from the fixed `common/severity_map`). A new `claim_id` + damage rows are **written to the DB**, linked to the chosen customer.
4. **Adjudication:** the existing `common/rules_engine` runs on that claim exactly like any seeded one → route + payout.

The money is still never model-emitted: the model only names the **damage categories**; the frozen severity map + cost table price them and the rules engine does the arithmetic. This is the honest production model — a fresh claim's damage of record is *established* from the photo at intake, then treated as authoritative.

### Coverage handling — LOCKED: Option 1 (single pool)
Arch 2 is **story-blind by design** (reading the incident narrative is an Arch 3 feature). The eval-mode payout math splits damage into two coverage pools — **collision** (crash) vs **comprehensive** (non-crash) — each with its own active-flag and limit; that `coverage_type` was pre-stored on each seeded claim (derived from the story *at seed time*). A fresh uploaded photo has no story and no stored value, and Arch 2 has no way to infer the peril, so for the live path it **does not split the pools**:

- **Covered?** If the policy is active and has *any* coverage active → covered; if lapsed → `auto_deny`.
- **Cap:** `min(total_repair_cost, limit)` (seeded policies set collision_limit == comprehensive_limit, so "the limit" is a single unambiguous number).
- **Deductible + escalation gate + deductible-≥-limit guard:** identical to eval mode.

Known limitation (accepted): this assumes any damage falls under whatever coverage the customer holds, so it **cannot produce a `not_covered` deny for a peril the customer didn't buy** (e.g. collision-only policy + hail damage would be approved, not denied). That peril-mismatch judgement is precisely the gap Arch 3 (story-reading) is meant to fill. Implemented as a live-only branch in the rules engine; the eval path (`compute_payout` by `claim_id`) is untouched.

## What exists vs. what this deploy wraps
- Exists and is wrapped: `arch2_split/pipeline.py` (gatekeeper → segmenter → damage_assessor → `common/rules_engine`), the seeded `db/avahi.db` (cost_table 18, **policies 24** — these drive the customer dropdown), `common/severity_map`, `arch2_split/vlm_client`.
- The seeded **claims** (368) and their bundled images are **not needed** for the live-upload demo — customers upload their own photos. The DB is still required for **policies** and the cost table.
- NOT part of this deploy (not built yet): Arch 1, Arch 3, `eval/`, golden-set export. This deploys **Arch 2 only**.

## Known blockers / notes
1. **`VISION_MODEL`** is now env-overridable (`os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")`) — set a real Groq vision model at deploy time. *(done in `vlm_client.py`)*
2. **VLM robustness** is in place: `vlm_client._encode_image` downscales the long edge to 512px, and calls retry on `json_validate_failed` + rate limits. *(done)*
3. **Groq free tier ~8000 tokens/min**; each run = 3 VLM calls (gatekeeper, segmenter, damage_assessor). The 512px downscale keeps a single run under the cap.
4. **DB must be writable** — intake creates new claim rows. Uploaded claims are ephemeral (lost on host restart); they do **not** touch the frozen golden set (separate files).
5. **No gatekeeper "retake" demo by default** — the customer supplies the photo, so a genuinely bad photo *can* now trigger the retake path organically (unlike the seeded set, which had no junk images).

---

## Deployment architecture

| Piece | Choice | Notes |
|---|---|---|
| Backend | FastAPI + uvicorn | wraps the intake + `pipeline`; serves API + static frontend from one process |
| Frontend | single-page vanilla HTML/CSS/JS | no build step — served by FastAPI StaticFiles |
| DB | bundled `db/avahi.db`, **read-write** | needs writes for new uploaded claims; ephemeral host FS is fine |
| Uploads | ephemeral `uploads/` dir | user photos saved per-run; not committed, not persistent |
| Secret | `GROQ_API_KEY` env var on host | never committed |
| Model | `VISION_MODEL` env var | overrides the placeholder without a code change |
| Container | `Dockerfile` (python:3.12-slim) | binds `$PORT`; portable across Render/Railway/Fly |
| Host config | `render.yaml` (+ Fly notes in DEPLOY.md) | free tier |

### API surface (FastAPI, same-origin, no CORS)
- `GET /` → serve `app/static/index.html`
- `GET /api/customers` → list the 24 customers/policies for the dropdown: `customer_id`, name, car_class, status, coverages active, limits, deductible.
- `POST /api/upload` → multipart: **photo file + `customer_id`**. Saves the photo, runs intake (vision → new claim rows) + adjudication, returns the serialized pipeline result (route, payout, confidence, per-stage logs, gatekeeper, panels, predicted damage, reasons) **plus the new `claim_id`**. **This is the only endpoint that spends tokens.**

### Files to be created
```
app/
  __init__.py
  main.py                # FastAPI app: customers, upload+run, serialization
  static/
    index.html           # the whole frontend (inline CSS/JS)
requirements.txt         # fastapi, uvicorn[standard], python-multipart, groq, pillow
Dockerfile
.dockerignore            # exclude CarDD_release/, archive.zip, .git, __pycache__, .env, uploads/
render.yaml
DEPLOY.md                # step-by-step: set env, local test, deploy
```
### Files to be edited
- `arch2_split/pipeline.py`: add a live-intake path — run vision on the uploaded photo, write the new claim + damage rows (severity from `severity_map`), then adjudicate via the Option-1 single-pool branch. Eval path (`run_claim`) untouched.
- `common/rules_engine.py`: add an Option-1 single-pool coverage branch for live claims (covered if policy active + any coverage active; cap by the single limit; same deductible + gates). Eval path unchanged.
- `app/main.py`: replace the pick-a-claim endpoints with `GET /api/customers` + `POST /api/upload`; open the DB read-write.

### Deploy paths (documented in DEPLOY.md)
- **No image bundling needed** — the ~87MB `web_images/` and `scripts/bundle_images.py` are **dropped**; photos come from the uploader at runtime. This removes the biggest repo-size problem.
- **Render (from GitHub):** commit `db/avahi.db` (small); set `GROQ_API_KEY` + `VISION_MODEL` in the dashboard. One-click from `render.yaml`.
- **Fly.io (local build push):** same, no large assets to ship.

---

## Frontend plan (`app/static/index.html`)

**Goal:** make the SPEC's core thesis *visible* — perception is probabilistic (the model, with confidence), adjudication is deterministic (the money, from code).

### Layout
- **Top — claim intake.** A **customer dropdown** (shows name, car_class, status, coverage chips, limits/deductible once selected — so the visitor sees the policy the payout will run against), a **photo upload / drag-drop** with a client-side preview, and one primary button: **"▶ Run Arch 2 pipeline (live)"** with a note "makes 3 live VLM calls".
- **Below — the live run timeline** (fills in as results arrive):
  1. **Gatekeeper** — valid? reason, confidence bar. If invalid → route `retake`, stop.
  2. **Segmenter** — panel chips (labelled "observability-only, not used for money").
  3. **Damage assessor** — **predicted** damage chips with per-instance confidence. (No "truth damage" to compare against — this is a fresh claim; the photo *is* the source of truth.)
  4. **Rules engine** — the deterministic block: cost breakdown → limit cap → deductible → **payout**, then the escalation gate ($2000 + confidence), ending in the final **route** badge. Labelled "pure code — no model touched this number".

### Result header
Banner with the final route (colour-coded), the payout (or "—" for deny/escalate/retake), the weakest-link confidence that gated it, and a short "why" line from `reasons`.

### The teaching callout
Persistent near the payout: **"The model read the damage. The code computed the dollars. No model emitted this number."**

### Style
Neutral, clean, light/dark aware. No external CDNs (self-contained). Confidence as small bars. Route colours: approve=green, escalate=amber, deny=red, retake=grey.

### Explicitly out of scope (stated in the UI, not faked)
- No incident-story input — that's an Arch 3 feature; Arch 2 is story-blind.
- No Arch 1 / Arch 3 comparison (not built).
