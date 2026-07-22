"""Populates cost_table: (damage_category, car_class) -> cost.

severity/operation are fixed by damage_category (SPEC.md §8a) rather than
varying independently -- CarDD has no native severity field, so this map is
an explicit, documented approximation, not a measured value.

Base costs (economy tier, labour_rate=95/hr) are ballpark U.S. body-shop
figures for each operation type -- named in SPEC.md §11 as the one openly
invented ground-truth piece. car_class multipliers approximate how much
more OEM parts + labour cost on a midsize/luxury vehicle for the same
physical damage (SPEC.md's "why car_class matters" callout).

Usage: python3 db/seed_cost_table.py [path/to/avahi.db]
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "avahi.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# damage_category -> (severity, operation, base_parts_cost, base_labour_hours)
# Fixed map from SPEC.md §8a.
DAMAGE_BASE = {
    "dent":          ("minor",    "repair",  0,   1.5),
    "scratch":       ("minor",    "repair",  50,  1.0),
    "crack":         ("moderate", "repair",  120, 2.5),
    "tire flat":     ("moderate", "repair",  150, 0.5),
    "lamp broken":   ("severe",   "replace", 250, 1.0),
    "glass shatter": ("severe",   "replace", 350, 1.5),
}

# car_class -> (parts_cost_multiplier, labour_rate)
CAR_CLASS = {
    "economy":  (1.0, 95),
    "midsize":  (1.6, 110),
    "luxury":   (4.0, 160),
}


def seed(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())

    rows = []
    for damage_category, (severity, operation, base_parts_cost, labour_hours) in DAMAGE_BASE.items():
        for car_class, (parts_multiplier, labour_rate) in CAR_CLASS.items():
            rows.append((
                damage_category,
                severity,
                car_class,
                operation,
                round(base_parts_cost * parts_multiplier, 2),
                labour_hours,
                labour_rate,
            ))

    conn.executemany(
        """
        INSERT INTO cost_table
            (damage_category, severity, car_class, operation, parts_cost, labour_hours, labour_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (damage_category, car_class) DO UPDATE SET
            severity = excluded.severity,
            operation = excluded.operation,
            parts_cost = excluded.parts_cost,
            labour_hours = excluded.labour_hours,
            labour_rate = excluded.labour_rate
        """,
        rows,
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM cost_table").fetchone()[0]
    print(f"cost_table seeded: {count} rows at {db_path}")
    conn.close()


if __name__ == "__main__":
    seed(DB_PATH)
