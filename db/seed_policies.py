"""Populates policies: one synthetic customer/policy row per generated customer.

Per SPEC.md §2/§8, each curated photo gets 2-4 *deliberately varied* policies
(not random noise) so the same damage produces different routes depending on
policy math -- active/lapsed, over/under the escalation threshold,
covered/uncovered. `generate_varied_policies()` is the reusable generator
that db/seed_claims.py (Plan.md step 6, built once photos are curated) will
call per photo. Running this file directly seeds a standalone demo pool so
the table/generator can be inspected before curation is done.

Usage: python3 db/seed_policies.py [path/to/avahi.db] [--count N]
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path

DB_PATH_DEFAULT = Path(__file__).parent / "avahi.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

VEHICLES = {
    "economy": [("Toyota", "Corolla"), ("Honda", "Civic"), ("Hyundai", "Elantra")],
    "midsize": [("Toyota", "Camry"), ("Honda", "Accord"), ("Nissan", "Altima")],
    "luxury":  [("BMW", "3 Series"), ("Mercedes-Benz", "C-Class"), ("Audi", "A4")],
}
FIRST_NAMES = ["James", "Maria", "Wei", "Fatima", "Liam", "Sofia", "Noah", "Aisha",
               "Lucas", "Yuki", "Omar", "Elena", "David", "Priya", "Carlos", "Grace"]
LAST_NAMES = ["Smith", "Garcia", "Chen", "Khan", "Mueller", "Rossi", "Kim", "Silva",
              "Novak", "Diallo", "Nguyen", "Patel", "Brown", "Kowalski", "Reyes", "Andersson"]

# Deliberate archetypes: each hits a specific route per SPEC.md §10's payout logic.
# limit_tier/deductible_tier are dollar ranges (SPEC.md's $2,000 escalation threshold
# is the reference point -- "low" limits sit below/near it, "high" sit comfortably above).
ARCHETYPES = [
    {"name": "active_full_coverage_high_limit",
     "policy_status": "active", "collision_active": 1, "comprehensive_active": 1,
     "limit_range": (5000, 10000), "deductible_range": (100, 300)},
    {"name": "active_full_coverage_low_limit",
     "policy_status": "active", "collision_active": 1, "comprehensive_active": 1,
     "limit_range": (800, 1800), "deductible_range": (500, 1000)},
    {"name": "lapsed_full_coverage",
     "policy_status": "lapsed", "collision_active": 1, "comprehensive_active": 1,
     "limit_range": (5000, 10000), "deductible_range": (100, 300)},
    {"name": "active_collision_only",
     "policy_status": "active", "collision_active": 1, "comprehensive_active": 0,
     "limit_range": (3000, 8000), "deductible_range": (250, 750)},
    {"name": "active_comprehensive_only",
     "policy_status": "active", "collision_active": 0, "comprehensive_active": 1,
     "limit_range": (3000, 8000), "deductible_range": (250, 750)},
    {"name": "active_full_coverage_mid_limit",
     "policy_status": "active", "collision_active": 1, "comprehensive_active": 1,
     "limit_range": (2000, 4000), "deductible_range": (250, 500)},
]


def _make_vin(rng: random.Random) -> str:
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # VIN excludes I, O, Q
    return "".join(rng.choice(chars) for _ in range(17))


def _make_plate(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
    digits = "".join(rng.choice("0123456789") for _ in range(4))
    return f"{letters}-{digits}"


def make_policy(customer_id: str, archetype: dict, rng: random.Random) -> dict:
    car_class = rng.choice(list(VEHICLES.keys()))
    make, model = rng.choice(VEHICLES[car_class])
    limit = round(rng.uniform(*archetype["limit_range"]), -1)
    deductible = round(rng.uniform(*archetype["deductible_range"]), -1)
    # A deductible at/above the limit makes the policy structurally void -- no
    # loss of any size ever recovers a dollar. Keep it meaningfully below the
    # limit so every generated policy is a legitimate product.
    deductible = min(deductible, round(limit * 0.7, -1))

    policy_data = {
        "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "policy_number": f"AVH-{rng.randint(100000, 999999)}",
        "vehicle": {
            "make": make,
            "model": model,
            "year": rng.randint(2016, 2024),
            "vin": _make_vin(rng),
            "plate_number": _make_plate(rng),
        },
        "archetype": archetype["name"],  # traceability for debugging, not load-bearing
    }

    return {
        "customer_id": customer_id,
        "car_class": car_class,
        "policy_status": archetype["policy_status"],
        "collision_active": archetype["collision_active"],
        "comprehensive_active": archetype["comprehensive_active"],
        "collision_limit": limit,
        "comprehensive_limit": limit,
        "deductible": deductible,
        "policy_data": json.dumps(policy_data),
    }


def generate_varied_policies(photo_id: str, n: int, rng: random.Random) -> list[dict]:
    """2-4 deliberately varied policies for one curated photo (SPEC.md §2).

    Samples n distinct archetypes (without replacement, cycling if n > len(ARCHETYPES))
    so the same damage produces different routes depending on policy math.
    """
    archetypes = rng.sample(ARCHETYPES, k=min(n, len(ARCHETYPES)))
    while len(archetypes) < n:
        archetypes.append(rng.choice(ARCHETYPES))
    return [
        make_policy(f"{photo_id}-cust{i+1}", archetype, rng)
        for i, archetype in enumerate(archetypes)
    ]


def seed(db_path: Path, count: int, seed_value: int = 42) -> None:
    rng = random.Random(seed_value)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())

    policies = []
    for i in range(count):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        policies.append(make_policy(f"demo-cust{i+1}", archetype, rng))

    conn.executemany(
        """
        INSERT INTO policies
            (customer_id, car_class, policy_status, collision_active, comprehensive_active,
             collision_limit, comprehensive_limit, deductible, policy_data)
        VALUES (:customer_id, :car_class, :policy_status, :collision_active, :comprehensive_active,
                :collision_limit, :comprehensive_limit, :deductible, :policy_data)
        ON CONFLICT (customer_id) DO UPDATE SET
            car_class = excluded.car_class,
            policy_status = excluded.policy_status,
            collision_active = excluded.collision_active,
            comprehensive_active = excluded.comprehensive_active,
            collision_limit = excluded.collision_limit,
            comprehensive_limit = excluded.comprehensive_limit,
            deductible = excluded.deductible,
            policy_data = excluded.policy_data
        """,
        policies,
    )
    conn.commit()

    result_count = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
    print(f"policies seeded: {result_count} rows at {db_path}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default=str(DB_PATH_DEFAULT))
    parser.add_argument("--count", type=int, default=len(ARCHETYPES) * 4,
                         help="demo pool size when run standalone (real counts come from seed_claims.py per curated photo)")
    args = parser.parse_args()
    seed(Path(args.db_path), args.count)
