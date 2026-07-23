"""Unit tests for common/rules_engine.py against hand-computed cases.

Plan.md step 10: "Unit-test this against hand-computed cases -- this is the
piece that must be ~100% correct, since any miss here is a code bug, not a
model error." Uses stdlib unittest (no pytest dependency installed).

Run: python3 -m unittest common.test_rules_engine -v
"""
import sqlite3
import unittest
from pathlib import Path

from common import rules_engine

SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"


class RulesEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(SCHEMA_PATH.read_text())
        # Fixed cost_table fixture, independent of the real seed data, so
        # expected payouts below are hand-computable from these numbers alone.
        self.conn.executemany(
            """INSERT INTO cost_table
               (damage_category, severity, car_class, operation, parts_cost, labour_hours, labour_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                ("dent", "minor", "economy", "repair", 0, 1.5, 100),      # $150
                ("scratch", "minor", "economy", "repair", 50, 1.0, 100),  # $150
                ("scratch", "minor", "midsize", "repair", 50, 1.0, 100),  # $150
                ("glass shatter", "severe", "economy", "replace", 400, 1.0, 100),  # $500
                ("lamp broken", "severe", "midsize", "replace", 400, 1.0, 100),    # $500
            ],
        )

    def tearDown(self):
        self.conn.close()

    def _add_policy(self, **overrides):
        policy = {
            "customer_id": "cust1", "car_class": "economy", "policy_status": "active",
            "collision_active": 1, "comprehensive_active": 1,
            "collision_limit": 5000, "comprehensive_limit": 5000, "deductible": 200,
            "policy_data": None,
        }
        policy.update(overrides)
        self.conn.execute(
            """INSERT INTO policies
               (customer_id, car_class, policy_status, collision_active, comprehensive_active,
                collision_limit, comprehensive_limit, deductible, policy_data)
               VALUES (:customer_id, :car_class, :policy_status, :collision_active, :comprehensive_active,
                       :collision_limit, :comprehensive_limit, :deductible, :policy_data)""",
            policy,
        )
        return policy["customer_id"]

    def _add_claim(self, claim_id, customer_id, instances):
        self.conn.execute(
            "INSERT INTO claims (claim_id, customer_id, photo_file, claim_story, claim_date) VALUES (?, ?, 'x.jpg', '', '2026-01-01')",
            (claim_id, customer_id),
        )
        self.conn.executemany(
            "INSERT INTO claim_damage_instances (claim_id, damage_category, severity, coverage_type) VALUES (?, ?, ?, ?)",
            [(claim_id, dc, sev, cov) for dc, sev, cov in instances],
        )
        self.conn.commit()

    def test_simple_covered_auto_approve(self):
        # single $150 dent, full coverage, $200 deductible -> payout floors at 0, still auto_approve
        customer_id = self._add_policy(deductible=100)
        self._add_claim("c1", customer_id, [("dent", "minor", "collision")])
        result = rules_engine.compute_payout(self.conn, "c1", confidence=0.9)
        self.assertEqual(result.route, "auto_approve")
        self.assertAlmostEqual(result.payout, 150 - 100)  # $50
        self.assertEqual(result.deductible_applied, 100)

    def test_lapsed_policy_auto_deny(self):
        customer_id = self._add_policy(policy_status="lapsed")
        self._add_claim("c2", customer_id, [("dent", "minor", "collision")])
        result = rules_engine.compute_payout(self.conn, "c2", confidence=0.9)
        self.assertEqual(result.route, "auto_deny")
        self.assertIsNone(result.payout)
        self.assertIn("policy_lapsed", result.reasons)

    def test_uncovered_coverage_type_auto_deny(self):
        customer_id = self._add_policy(comprehensive_active=0)
        self._add_claim("c3", customer_id, [("glass shatter", "severe", "comprehensive")])
        result = rules_engine.compute_payout(self.conn, "c3", confidence=0.9)
        self.assertEqual(result.route, "auto_deny")
        self.assertIsNone(result.payout)
        self.assertIn("not_covered", result.reasons)

    def test_low_confidence_escalates_despite_cheap_clean_payout(self):
        customer_id = self._add_policy(deductible=100)
        self._add_claim("c4", customer_id, [("dent", "minor", "collision")])
        result = rules_engine.compute_payout(self.conn, "c4", confidence=0.5)
        self.assertEqual(result.route, "escalate")
        self.assertIn("confidence_below_threshold", result.reasons)
        self.assertAlmostEqual(result.payout, 50)  # payout is still computed, just not auto-approved

    def test_payout_above_threshold_escalates(self):
        customer_id = self._add_policy(deductible=0)
        self._add_claim("c5", customer_id, [
            ("glass shatter", "severe", "comprehensive"),
            ("glass shatter", "severe", "comprehensive"),
            ("glass shatter", "severe", "comprehensive"),
            ("glass shatter", "severe", "comprehensive"),
            ("glass shatter", "severe", "comprehensive"),  # 5 x $500 = $2500 > $2000
        ])
        result = rules_engine.compute_payout(self.conn, "c5", confidence=0.99)
        self.assertEqual(result.route, "escalate")
        self.assertIn("payout_above_auto_approve_threshold", result.reasons)
        self.assertAlmostEqual(result.payout, 2500)

    def test_limit_caps_payout_per_coverage_pool(self):
        customer_id = self._add_policy(collision_limit=100, deductible=0)
        self._add_claim("c6", customer_id, [("scratch", "minor", "collision")])  # $150 cost, capped at $100 limit
        result = rules_engine.compute_payout(self.conn, "c6", confidence=0.9)
        self.assertAlmostEqual(result.payout, 100)  # capped by collision_limit, not raw $150 cost

    def test_mixed_coverage_types_summed_after_separate_caps(self):
        customer_id = self._add_policy(car_class="midsize", collision_limit=100, comprehensive_limit=5000, deductible=0)
        self._add_claim("c7", customer_id, [
            ("scratch", "minor", "collision"),           # $150 cost, capped at $100
            ("lamp broken", "severe", "comprehensive"),  # $500 cost, well under $5000 limit
        ])
        result = rules_engine.compute_payout(self.conn, "c7", confidence=0.9)
        self.assertAlmostEqual(result.payout, 100 + 500)  # collision capped $100 + comprehensive uncapped $500

    def test_deductible_exceeds_limit_escalates_as_data_integrity(self):
        # limit $100, deductible $1000 -> no loss ever recovers a dollar; this
        # is a malformed policy, not a $0 auto-approve.
        customer_id = self._add_policy(collision_limit=100, deductible=1000)
        self._add_claim("c9", customer_id, [("scratch", "minor", "collision")])  # $150 cost
        result = rules_engine.compute_payout(self.conn, "c9", confidence=0.99)
        self.assertEqual(result.route, "escalate")
        self.assertIsNone(result.payout)
        self.assertIn("policy_deductible_exceeds_limit", result.reasons)

    def test_partial_coverage_excludes_uncovered_instance(self):
        customer_id = self._add_policy(collision_active=0, deductible=0)
        self._add_claim("c8", customer_id, [
            ("dent", "minor", "collision"),          # not covered, excluded
            ("scratch", "minor", "comprehensive"),   # wrong coverage_type for the damage in practice,
        ])                                            # but tests that only comprehensive counts here.
        result = rules_engine.compute_payout(self.conn, "c8", confidence=0.9)
        self.assertEqual(result.route, "auto_approve")
        self.assertAlmostEqual(result.payout, 150)  # only the comprehensive $150 scratch counts
        self.assertIn("partial_coverage_some_instances_excluded", result.reasons)


if __name__ == "__main__":
    unittest.main()
