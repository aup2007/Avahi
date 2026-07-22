import argparse
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH_DEFAULT = Path(__file__).parent / "avahi.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "curated_manifest.json"

# claim_date is drawn deterministically from a window ending at this fixed
# reference date (kept out of datetime.now() so re-runs are reproducible).
DATE_REF = date(2026, 7, 1)
DATE_WINDOW_DAYS = 365

# Story themes. Each theme fixes the coverage_type of the whole claim, because a
# claim is a single incident caused by a single peril (see coverage heuristic below).
# {road}/{place}/{side} are filled deterministically from the pools underneath.
COLLISION_STORIES = [
    "Rear-ended at a red light on {road}.",
    "Sideswiped by another car while merging onto {road}.",
    "Backed into a concrete pillar in the {place} parking garage.",
    "Clipped a guardrail swerving to avoid traffic on {road}.",
    "Collided with a car that ran a stop sign near {place}.",
    "Hit a curb hard and scraped the {side} side pulling into {place}.",
]
COMPREHENSIVE_STORIES = [
    "Caught in a hailstorm overnight while parked outside {place}.",
    "Vandalized in the {place} lot -- panels keyed and glass broken.",
    "A deer ran into the car at dusk on {road}.",
    "A tree branch fell on the car during a windstorm at {place}.",
    "Attempted break-in in the {place} parking lot damaged the car.",
    "Flying storm debris struck the car on {road}.",
]
ROADS = ["Route 9", "the I-40 on-ramp", "Elm Street", "the coastal highway", "Maple Avenue", "5th Street"]
PLACES = ["the mall", "the office", "downtown", "the grocery store", "the stadium", "the airport"]
SIDES = ["driver", "passenger"]

# coverage_type affinity per damage_category, used only to bias which story
# theme (collision vs comprehensive) is drawn for a photo -- impact damage
# (dent/scratch/crack) leans collision; glass/lamp are peril-ambiguous
# (hail and vandalism shatter them just as often as a crash); tire flat is neutral.
# Once a theme is chosen, EVERY instance in the claim takes that theme's
# coverage_type -- one incident, one peril, one coverage bucket (auditable).
CATEGORY_AFFINITY = {
    "dent":          {"collision": 1.0, "comprehensive": 0.2},
    "scratch":       {"collision": 1.0, "comprehensive": 0.3},
    "crack":         {"collision": 0.9, "comprehensive": 0.4},
    "glass shatter": {"collision": 0.6, "comprehensive": 0.8},
    "lamp broken":   {"collision": 0.6, "comprehensive": 0.7},
    "tire flat":     {"collision": 0.5, "comprehensive": 0.5},
}


def choose_coverage(instances: list[dict], rng: random.Random) -> str:
    coll = 1.0
    comp = 1.0
    for inst in instances:
        aff = CATEGORY_AFFINITY[inst["damage_category"]]
        coll += aff["collision"]
        comp += aff["comprehensive"]
    return "collision" if rng.random() < coll / (coll + comp) else "comprehensive"


def make_story(coverage_type: str, rng: random.Random) -> str:
    template = rng.choice(COLLISION_STORIES if coverage_type == "collision" else COMPREHENSIVE_STORIES)
    return template.format(road=rng.choice(ROADS), place=rng.choice(PLACES), side=rng.choice(SIDES))


def make_claim_date(rng: random.Random) -> str:
    return (DATE_REF - timedelta(days=rng.randint(0, DATE_WINDOW_DAYS))).isoformat()


def seed(db_path: Path, seed_value: int = 42) -> None:
    rng = random.Random(seed_value)
    manifest = json.loads(MANIFEST_PATH.read_text())
    images = manifest["images"]

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())  # DDL is all IF NOT EXISTS -- touches no data

    customer_ids = [r[0] for r in conn.execute("SELECT customer_id FROM policies ORDER BY customer_id")]
    if not customer_ids:
        raise SystemExit("policies table is empty -- run seed_policies.py first")

    # Idempotent replace: clear claims + child rows only, never policies/cost_table.
    conn.execute("DELETE FROM claim_damage_instances")
    conn.execute("DELETE FROM claims")

    claim_rows = []
    instance_rows = []
    n = 0
    for img in images:
        if img.get("is_junk") or not img.get("instances"):
            continue
        # 2-4 distinct policies per photo -> same damage, different policy math.
        k = rng.randint(2, 4)
        paired = rng.sample(customer_ids, k=min(k, len(customer_ids)))
        for customer_id in paired:
            n += 1
            claim_id = f"CLM-{n:05d}"
            coverage_type = choose_coverage(img["instances"], rng)
            story = make_story(coverage_type, rng)
            claim_rows.append((claim_id, customer_id, img["file_name"], story, make_claim_date(rng)))
            for inst in img["instances"]:
                instance_rows.append((claim_id, inst["damage_category"], inst["severity"], coverage_type))

    conn.executemany(
        "INSERT INTO claims (claim_id, customer_id, photo_file, claim_story, claim_date) "
        "VALUES (?, ?, ?, ?, ?)",
        claim_rows,
    )
    conn.executemany(
        "INSERT INTO claim_damage_instances (claim_id, damage_category, severity, coverage_type) "
        "VALUES (?, ?, ?, ?)",
        instance_rows,
    )
    conn.commit()

    claims_n = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    inst_n = conn.execute("SELECT COUNT(*) FROM claim_damage_instances").fetchone()[0]
    print(f"claims seeded: {claims_n} rows, claim_damage_instances: {inst_n} rows at {db_path}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default=str(DB_PATH_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed(Path(args.db_path), args.seed)
