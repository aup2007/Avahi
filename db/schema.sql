-- Avahi Car Insurance — SQLite schema
-- See SPEC.md §8 (data model) and §8a (why `part` was dropped in favor of `damage_category`).

PRAGMA foreign_keys = ON;

-- Table 1 — cost_table: pricing reference, not per-customer.
-- severity/operation are a fixed function of damage_category (SPEC.md §8a);
-- car_class is the only independent pricing axis besides damage_category.
CREATE TABLE IF NOT EXISTS cost_table (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    damage_category TEXT    NOT NULL CHECK (damage_category IN
                        ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')),
    severity        TEXT    NOT NULL CHECK (severity IN ('minor', 'moderate', 'severe')),
    car_class       TEXT    NOT NULL CHECK (car_class IN ('economy', 'midsize', 'luxury')),
    operation       TEXT    NOT NULL CHECK (operation IN ('repair', 'replace')),
    parts_cost      REAL    NOT NULL,
    labour_hours    REAL    NOT NULL,
    labour_rate     REAL    NOT NULL,
    UNIQUE (damage_category, car_class)
);

-- Table 2 — policies: one row per synthetic customer.
CREATE TABLE IF NOT EXISTS policies (
    customer_id           TEXT    PRIMARY KEY,
    car_class              TEXT    NOT NULL CHECK (car_class IN ('economy', 'midsize', 'luxury')),
    policy_status           TEXT    NOT NULL CHECK (policy_status IN ('active', 'lapsed')),
    collision_active         INTEGER NOT NULL CHECK (collision_active IN (0, 1)),
    comprehensive_active      INTEGER NOT NULL CHECK (comprehensive_active IN (0, 1)),
    collision_limit            REAL    NOT NULL,
    comprehensive_limit         REAL    NOT NULL,
    deductible                   REAL    NOT NULL,
    policy_data                   TEXT    -- JSON: name, policy_number, vehicle{make,model,year,vin,plate_number}
);

-- Table 3 — claims: ties a photo + story to a customer.
CREATE TABLE IF NOT EXISTS claims (
    claim_id    TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES policies (customer_id),
    photo_file  TEXT NOT NULL,  -- file_name within CarDD_release/CarDD_COCO/test2017
    claim_story TEXT,
    claim_date  TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_customer_id ON claims (customer_id);

-- claim_damage_instances: normalized child table (not a JSON blob) since it
-- must join against cost_table and group for the payout sum (SPEC.md §8).
CREATE TABLE IF NOT EXISTS claim_damage_instances (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        TEXT    NOT NULL REFERENCES claims (claim_id),
    damage_category TEXT    NOT NULL CHECK (damage_category IN
                        ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')),
    severity        TEXT    NOT NULL CHECK (severity IN ('minor', 'moderate', 'severe')),
    coverage_type   TEXT    NOT NULL CHECK (coverage_type IN ('collision', 'comprehensive'))
);
CREATE INDEX IF NOT EXISTS idx_claim_damage_instances_claim_id ON claim_damage_instances (claim_id);
