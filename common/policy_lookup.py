"""Table 2 (policies) lookup helper. Pure DB reads, no business logic.

Used by common/rules_engine.py and, indirectly, by both arch2_split/ and
arch3_agent/ (via arch2_split's pipeline / arch3_agent's tools.py wrapper) --
this is the one place a policy row is read from the DB.
"""
import json
import sqlite3


def get_policy(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT customer_id, car_class, policy_status, collision_active, comprehensive_active,
               collision_limit, comprehensive_limit, deductible, policy_data
        FROM policies WHERE customer_id = ?
        """,
        (customer_id,),
    ).fetchone()
    if row is None:
        return None

    (customer_id, car_class, policy_status, collision_active, comprehensive_active,
     collision_limit, comprehensive_limit, deductible, policy_data) = row

    policy = {
        "customer_id": customer_id,
        "car_class": car_class,
        "policy_status": policy_status,
        "collision_active": bool(collision_active),
        "comprehensive_active": bool(comprehensive_active),
        "collision_limit": collision_limit,
        "comprehensive_limit": comprehensive_limit,
        "deductible": deductible,
    }
    if policy_data:
        policy["policy_data"] = json.loads(policy_data)
    return policy


def get_policy_for_claim(conn: sqlite3.Connection, claim_id: str) -> dict | None:
    row = conn.execute("SELECT customer_id FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
    if row is None:
        return None
    return get_policy(conn, row[0])
