import argparse
import json
from collections import defaultdict
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DEFAULT = REPO_ROOT / "CarDD_release" / "CarDD_COCO" / "annotations" / "instances_test2017.json"
IMAGES_DIR = REPO_ROOT / "CarDD_release" / "CarDD_COCO" / "test2017"
MANIFEST_DEFAULT = REPO_ROOT / "data" / "curated_manifest.json"

# Fixed deterministic map off damage_category (SPEC.md §8a). CarDD has no native
# severity/operation field -- these are derived, not read from the annotations.
SEVERITY_MAP = {
    "dent":          ("minor",    "repair"),
    "scratch":       ("minor",    "repair"),
    "crack":         ("moderate", "repair"),
    "tire flat":     ("moderate", "repair"),
    "lamp broken":   ("severe",   "replace"),
    "glass shatter": ("severe",   "replace"),
}

TARGET_TOTAL = 120  # lands inside the 80-150 window from Plan.md step 3
JUNK_CAP = 8


def load_annotations(path: Path) -> dict:
    return json.loads(path.read_text())


def build_image_instances(coco: dict) -> tuple[dict, dict]:
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    file_name = {im["id"]: im["file_name"] for im in coco["images"]}
    # one entry per annotation -> preserves multiple instances (even same category) per image
    instances = defaultdict(list)
    for ann in coco["annotations"]:
        instances[ann["image_id"]].append(cat_name[ann["category_id"]])
    return file_name, instances


def curate(coco: dict, target_total: int, rng: random.Random) -> dict:
    file_name, instances = build_image_instances(coco)
    all_ids = [im["id"] for im in coco["images"]]

    # images with zero expert annotations are the only genuine "junk" candidates
    junk_ids = sorted(i for i in all_ids if i not in instances)

    # per-category queues, shuffled deterministically, for round-robin balanced fill
    by_category = defaultdict(list)
    for img_id, cats in instances.items():
        for cat in set(cats):
            by_category[cat].append(img_id)
    queues = {}
    for cat in sorted(by_category):
        ids = sorted(by_category[cat])
        rng.shuffle(ids)
        queues[cat] = ids

    selected: list[int] = []
    selected_set: set[int] = set()
    categories = sorted(queues)
    exhausted = set()
    while len(selected) < target_total and len(exhausted) < len(categories):
        for cat in categories:
            if len(selected) >= target_total:
                break
            if cat in exhausted:
                continue
            queue = queues[cat]
            picked = None
            while queue:
                candidate = queue.pop()
                if candidate not in selected_set:
                    picked = candidate
                    break
            if picked is None:
                exhausted.add(cat)
                continue
            selected.append(picked)
            selected_set.add(picked)

    junk_selected = junk_ids[:JUNK_CAP]

    def record(img_id: int, is_junk: bool) -> dict:
        insts = []
        for cat in instances.get(img_id, []):
            severity, operation = SEVERITY_MAP[cat]
            insts.append({"damage_category": cat, "severity": severity, "operation": operation})
        return {
            "image_id": img_id,
            "file_name": file_name[img_id],
            "image_path": str((IMAGES_DIR / file_name[img_id]).relative_to(REPO_ROOT)),
            "is_junk": is_junk,
            "instances": insts,
        }

    images = [record(i, False) for i in sorted(selected)]
    images += [record(i, True) for i in junk_selected]
    return images, junk_ids


def summarize(images: list[dict]) -> dict:
    per_category = defaultdict(int)
    per_severity = defaultdict(int)
    damage_images = 0
    junk_images = 0
    for img in images:
        if img["is_junk"]:
            junk_images += 1
        if img["instances"]:
            damage_images += 1
        for inst in img["instances"]:
            per_category[inst["damage_category"]] += 1
            per_severity[inst["severity"]] += 1
    return {
        "total_images": len(images),
        "damage_images": damage_images,
        "junk_images": junk_images,
        "instances_by_category": dict(sorted(per_category.items())),
        "instances_by_severity": dict(sorted(per_severity.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(ANNOTATIONS_DEFAULT))
    parser.add_argument("--out", default=str(MANIFEST_DEFAULT))
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco = load_annotations(Path(args.annotations))
    rng = random.Random(args.seed)
    images, junk_ids = curate(coco, args.target, rng)
    counts = summarize(images)

    junk_available = len(junk_ids) > 0
    junk_note = (
        f"{len(junk_ids)} images in test2017 carry no expert annotation; using up to "
        f"{JUNK_CAP} as gatekeeper junk candidates."
        if junk_available else
        "CarDD test2017 contains no unannotated/junk images -- every image is a real "
        "car-damage photo. Genuine junk/blurry/non-car images for the gatekeeper stage "
        "are NOT available here and must be sourced separately or synthesized in a later step."
    )

    manifest = {
        "source": "CarDD test2017 (expert COCO annotations)",
        "annotations_file": str(Path(args.annotations).relative_to(REPO_ROOT))
        if Path(args.annotations).is_relative_to(REPO_ROOT) else args.annotations,
        "seed": args.seed,
        "severity_map": {k: {"severity": v[0], "operation": v[1]} for k, v in SEVERITY_MAP.items()},
        "junk_available": junk_available,
        "junk_note": junk_note,
        "counts": counts,
        "images": images,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"manifest written: {out_path}")
    print(f"total images: {counts['total_images']} "
          f"(damage: {counts['damage_images']}, junk: {counts['junk_images']})")
    print(f"instances by category: {counts['instances_by_category']}")
    print(f"instances by severity: {counts['instances_by_severity']}")
    print(f"junk available: {junk_available} -- {junk_note}")


if __name__ == "__main__":
    main()
