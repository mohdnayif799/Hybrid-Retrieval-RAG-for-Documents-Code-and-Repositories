"""
Judge vs human agreement on the 20-question judge check.

    python -m benchmark.v3.generation.agreement

Reads the hand labels in judge_check_dev.yaml (written next to the reading
sheet judge_check_dev.md; both local) and the judge's verdicts in
results/dev.json. Prints, per field:
* raw agreement and Cohen's kappa;
* the direction of every disagreement: whether the judge was more lenient or
  stricter than the human.
Also writes results/judge_check_labels_dev.json: the labels and the agreement,
no text, safe to commit.

The bar (docs/ROADMAP_v4.md, decisions taken): at least 80% raw agreement on
`correct` and on `grounded`.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(GENERATION_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.v3.generation import judge as J

SHEET_PATH = os.path.join(GENERATION_DIR, "judge_check_dev.yaml")
RESULTS_PATH = os.path.join(GENERATION_DIR, "results", "dev.json")
LABELS_OUT = os.path.join(GENERATION_DIR, "results", "judge_check_labels_dev.json")
BAR = 0.80
BAR_FIELDS = ("correct", "grounded")

# Higher = more favourable to the answer. A judge grade above the human's is
# "more lenient". `refused` is not ordered (neither value is better in
# general), so only its confusion pairs are reported.
ORDER = {
    "correct": {"no": 0, "partial": 1, "yes": 2},
    "grounded": {"no": 0, "yes": 1},
    "length_fit": {"too_short": 0, "right": 1, "too_long": 0},
}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[c] * right[c] for c in set(left) | set(right)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def compare(human: dict, judged: dict) -> dict:
    """Per-field agreement between {id: labels} and {id: verdict}."""
    out = {}
    for field in J.FIELDS:
        pairs, lenient, stricter, other = [], [], [], []
        for qid, labels in human.items():
            verdict = judged.get(qid)
            if not verdict or "error" in verdict:
                continue
            h, j = labels[field], verdict[field]
            pairs.append((h, j))
            if h == j:
                continue
            order = ORDER.get(field)
            if order and h in order and j in order and order[h] != order[j]:
                (lenient if order[j] > order[h] else stricter).append(qid)
            else:
                other.append(qid)
        agree = sum(a == b for a, b in pairs)
        kappa = cohen_kappa(pairs)
        out[field] = {
            "n": len(pairs), "agree": agree,
            "raw": round(agree / len(pairs), 3) if pairs else None,
            "kappa": None if kappa is None else round(kappa, 3),
            "judge_more_lenient": lenient, "judge_stricter": stricter,
            "other_disagreements": other,
            "confusion_human_to_judge": dict(Counter(f"{h}->{j}" for h, j in pairs if h != j)),
        }
    out["passes_bar"] = all((out[f]["raw"] or 0) >= BAR for f in BAR_FIELDS)
    return out


def load_labels(path: str = SHEET_PATH) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        sheet = yaml.safe_load(handle)
    labels, problems = {}, []
    for item in sheet["items"]:
        lab = {k: str(v or "").strip().lower() for k, v in (item.get("label") or {}).items()}
        for field, allowed in J.FIELDS.items():
            if lab.get(field) not in allowed:
                problems.append(f"{item['id']}: {field}={lab.get(field)!r} (allowed: {', '.join(allowed)})")
        labels[item["id"]] = {f: lab.get(f) for f in J.FIELDS}
    if problems:
        raise SystemExit("unfinished or invalid labels:\n  " + "\n  ".join(problems))
    return labels


def main() -> int:
    if not os.path.exists(SHEET_PATH):
        raise SystemExit(f"no judge-check sheet at {SHEET_PATH}: run the full dev eval first")
    human = load_labels()
    with open(RESULTS_PATH, encoding="utf-8") as handle:
        results = json.load(handle)
    judged = {qid: row["verdict"] for qid, row in results["per_question"].items()}
    missing = sorted(set(human) - set(judged))
    if missing:
        raise SystemExit(f"labelled ids missing from results/dev.json: {', '.join(missing)}")
    agreement = compare(human, judged)

    print(f"JUDGE vs HUMAN on {len(human)} dev questions (judge: {results['judge_model']}, "
          f"rubric {results['judge_version']})")
    for field in J.FIELDS:
        a = agreement[field]
        kappa = "-" if a["kappa"] is None else f"{a['kappa']:.2f}"
        print(f"  {field:11s} agree {a['agree']}/{a['n']} ({100 * (a['raw'] or 0):.0f}%), "
              f"kappa {kappa}; judge more lenient: {len(a['judge_more_lenient'])}, "
              f"stricter: {len(a['judge_stricter'])}, other: {len(a['other_disagreements'])}")
        if a["confusion_human_to_judge"]:
            print("              human->judge: " + ", ".join(
                f"{k} x{v}" for k, v in sorted(a["confusion_human_to_judge"].items())))
        for kind in ("judge_more_lenient", "judge_stricter", "other_disagreements"):
            if a[kind]:
                print(f"              {kind}: {', '.join(a[kind])}")
    verdict = "PASSES" if agreement["passes_bar"] else "DOES NOT PASS"
    print(f"\nBar (>= {int(BAR * 100)}% on {' and '.join(BAR_FIELDS)}): {verdict}")

    os.makedirs(os.path.dirname(LABELS_OUT), exist_ok=True)
    with open(LABELS_OUT, "w", encoding="utf-8") as handle:
        json.dump({"judge_model": results["judge_model"], "judge_version": results["judge_version"],
                   "judge_prompt_sha256": results["judge_prompt_sha256"],
                   "labels": human, "agreement": agreement}, handle, indent=1)
        handle.write("\n")
    print(f"written: {os.path.relpath(LABELS_OUT, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
