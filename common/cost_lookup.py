"""Table 1 (cost_table) lookup helper. Pure lookup + arithmetic, no policy logic.

Plan.md step 9. Per SPEC.md §8a, `part` was dropped from the schema -- there
is no same-part dedup step anymore (that only applied when two instances
could land on the same physical panel). Every row in claim_damage_instances
is treated as a distinct physical repair and summed.
"""
import sqlite3


def instance_cost(conn: sqlite3.Connection, damage_category: str, car_class: str) -> float:
    """total_repair_cost = parts_cost + labour_hours * labour_rate (SPEC.md §8, Table 1)."""
    row = conn.execute(
        "SELECT parts_cost, labour_hours, labour_rate FROM cost_table WHERE damage_category = ? AND car_class = ?",
        (damage_category, car_class),
    ).fetchone()
    if row is None:
        raise ValueError(f"no cost_table entry for damage_category={damage_category!r}, car_class={car_class!r}")
    parts_cost, labour_hours, labour_rate = row
    return parts_cost + labour_hours * labour_rate


def claim_costs_by_coverage(conn: sqlite3.Connection, claim_id: str, car_class: str) -> dict:
    """Per-instance costs for a claim, split by coverage_type (collision vs comprehensive).

    Split-by-coverage-type is what lets common/rules_engine.py check each
    instance's coverage_type against the policy's coverages_active and apply
    each coverage type's own limit (SPEC.md §4).
    """
    rows = conn.execute(
        "SELECT damage_category, severity, coverage_type FROM claim_damage_instances WHERE claim_id = ?",
        (claim_id,),
    ).fetchall()

    instances = []
    totals = {"collision": 0.0, "comprehensive": 0.0}
    for damage_category, severity, coverage_type in rows:
        cost = instance_cost(conn, damage_category, car_class)
        instances.append({
            "damage_category": damage_category,
            "severity": severity,
            "coverage_type": coverage_type,
            "cost": cost,
        })
        totals[coverage_type] += cost

    return {
        "instances": instances,
        "collision_cost": totals["collision"],
        "comprehensive_cost": totals["comprehensive"],
        "total_cost": totals["collision"] + totals["comprehensive"],
    }
