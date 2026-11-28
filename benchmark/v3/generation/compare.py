"""
Compare a tagged generation run against the dev baseline, question by question.

    python -m benchmark.v3.generation.compare --tag refusal

Reads results/dev.json (the baseline) and results/dev_<tag>.json. Both must
use the same frozen questions, the same judge and the same judge prompt; only
the answers may differ. Prints, with no text, and writes
results/compare_dev_<tag>.json:

* Phase 2b's success criteria, each PASS or FAIL:
  1. Of the baseline's answer-step refusals (refused with every part, or some
     parts, of the gold in context), at most 3 remain.
  2. `correct` goes up, question by question: the mean paired difference of
     the correct score (yes 1, partial 0.5, no 0) is above 0, and more
     questions improved than got worse.
  3. Every unanswerable question is still refused.
  4. Grounded stays at or above 97%.
  5. (added for the refusal2 revision) multi-hop correct_yes is at least the
     baseline's.
* Paired differences with 95% bootstrap intervals, overall, by query shape
  (bare vs sentence) and by question type.
* The questions that flipped each way.
* Every answerable question with no gold in context, by outcome, in both
  runs: refused, correct, partial, grounded but wrong (answering a nearby
  question), or ungrounded (guessing).
* What happened to the baseline's refusals with no gold in context. Those
  refusals were right for the answer step. If one is now answered without
  grounding, the model is guessing: a grounding failure, not a win.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(GENERATION_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(GENERATION_DIR, "results")
MAX_REMAINING = 3
MIN_GROUNDED = 0.97
SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def _ok(v: dict) -> bool:
    return bool(v) and "error" not in v


def check_comparable(base: dict, new: dict) -> list[str]:
    """Hard mismatches (refuse) and soft ones (warn) between the two runs."""
    for key in ("split", "manifest_sha256", "judge_model", "judge_version", "judge_prompt_sha256"):
        if base.get(key) != new.get(key):
            raise SystemExit(f"not comparable: {key} differs ({base.get(key)} vs {new.get(key)})")
    if set(base["per_question"]) != set(new["per_question"]):
        raise SystemExit("not comparable: the two runs scored different questions")
    warnings = []
    if base.get("answer_prompt_sha256") == new.get("answer_prompt_sha256"):
        warnings.append("the answer prompt is the same in both runs")
    return warnings


def paired(ids: list, a: dict, b: dict, fn) -> dict | None:
    """Paired bootstrap of fn(verdict) between base (a) and new (b) over ids."""
    from benchmark.v3 import stats as S

    ids = [i for i in sorted(ids) if _ok(a[i]["verdict"]) and _ok(b[i]["verdict"])]
    if not ids:
        return None
    x = [fn(a[i]["verdict"]) for i in ids]
    y = [fn(b[i]["verdict"]) for i in ids]
    idx = S.resample_matrix(len(ids), 10_000, 20261123)
    d = S.paired_diff_ci(y, x, idx)
    return {"n": len(ids), "base": statistics.mean(x), "new": statistics.mean(y), **d}


def compare(base: dict, new: dict) -> dict:
    a, b = base["per_question"], new["per_question"]
    answerable = [i for i, r in a.items() if r["qtype"] != "unanswerable"]
    unanswerable = [i for i, r in a.items() if r["qtype"] == "unanswerable"]
    wr = base["wrong_refusals"]
    target = sorted(wr["all_parts_in_context"] + wr["some_parts_in_context"])
    no_gold = sorted(wr["gold_not_in_context"])

    def refused(rows, i):
        return bool(rows[i]["verdict"].get("wrongly_refused"))

    remaining = [i for i in target if refused(b, i)]

    def correct_score(v):
        return SCORE[v["correct"]]

    up = sorted(i for i in answerable if _ok(a[i]["verdict"]) and _ok(b[i]["verdict"])
                and correct_score(b[i]["verdict"]) > correct_score(a[i]["verdict"]))
    down = sorted(i for i in answerable if _ok(a[i]["verdict"]) and _ok(b[i]["verdict"])
                  and correct_score(b[i]["verdict"]) < correct_score(a[i]["verdict"]))
    score_diff = paired(answerable, a, b, correct_score)

    unans_refused = sum(1 for i in unanswerable if b[i]["verdict"].get("refusal_correct") == "yes")
    graded = [b[i]["verdict"] for i in b if _ok(b[i]["verdict"]) and b[i]["verdict"]["grounded"] != "na"]
    grounded_rate = (sum(v["grounded"] == "yes" for v in graded) / len(graded)) if graded else None

    criteria = {
        "answer_step_refusals_remaining": {
            "value": f"{len(remaining)} of {len(target)}", "remaining": remaining,
            "pass": len(remaining) <= MAX_REMAINING},
        "correct_up": {
            "value": (f"score {100 * score_diff['diff']:+.1f} "
                      f"[{100 * score_diff['ci_low']:+.1f}, {100 * score_diff['ci_high']:+.1f}]; "
                      f"{len(up)} up, {len(down)} down") if score_diff else "n/a",
            "pass": bool(score_diff) and score_diff["diff"] > 0 and len(up) > len(down)},
        "unanswerable_refused": {
            "value": f"{unans_refused}/{len(unanswerable)}",
            "pass": unans_refused == len(unanswerable)},
        "grounded": {
            "value": "n/a" if grounded_rate is None else f"{100 * grounded_rate:.1f}%",
            "pass": grounded_rate is not None and grounded_rate >= MIN_GROUNDED},
    }
    # Added for the refusal2 revision: the first 2b wording cost multi-hop
    # answers their "yes" (wrong "not covered" caveats), so multi-hop
    # correct_yes must be back to at least the baseline's.
    mh_base, mh_new = _yes_rate(a, "multi_hop"), _yes_rate(b, "multi_hop")
    if mh_base is not None and mh_new is not None:
        criteria["multi_hop_correct_yes"] = {
            "value": (f"{mh_new[0]}/{mh_new[1]} ({100 * mh_new[0] / mh_new[1]:.0f}%), "
                      f"baseline {mh_base[0]}/{mh_base[1]} ({100 * mh_base[0] / mh_base[1]:.0f}%)"),
            "pass": mh_new[0] / mh_new[1] >= mh_base[0] / mh_base[1]}

    # The answers that were right to refuse: what did the new prompt do with them?
    no_gold_after = {}
    for i in no_gold:
        v = b[i]["verdict"]
        if not _ok(v):
            status = "judge error"
        elif v.get("wrongly_refused"):
            status = "still refused"
        elif v["grounded"] == "no":
            status = "answered, NOT grounded (guessing)"
        else:
            status = f"answered, grounded, correct={v['correct']}"
        no_gold_after[i] = status

    groups = {"all answerable": answerable}
    for i in answerable:
        groups.setdefault(f"shape:{a[i].get('query_shape')}", []).append(i)
        groups.setdefault(a[i]["qtype"], []).append(i)
    by_group = {}
    for name, ids in sorted(groups.items(), key=lambda kv: (kv[0] != "all answerable", kv[0])):
        by_group[name] = {
            "n": len(ids),
            "correct_score": paired(ids, a, b, correct_score),
            "correct_yes": paired(ids, a, b, lambda v: float(v["correct"] == "yes")),
            "wrongly_refused": paired(ids, a, b, lambda v: float(bool(v.get("wrongly_refused")))),
            "length_right": paired(ids, a, b, lambda v: float(v["length_fit"] == "right")),
            "median_words": [statistics.median([r[i]["verdict"]["word_count"] for i in ids
                                                if _ok(r[i]["verdict"])] or [0]) for r in (a, b)],
        }

    flips = {
        "correct_up": up, "correct_down": down,
        "refused_to_answered": sorted(i for i in answerable if refused(a, i) and not refused(b, i)),
        "answered_to_refused": sorted(i for i in answerable if not refused(a, i) and refused(b, i)),
        "grounded_yes_to_no": sorted(i for i in b if _ok(a[i]["verdict"]) and _ok(b[i]["verdict"])
                                     and a[i]["verdict"]["grounded"] == "yes"
                                     and b[i]["verdict"]["grounded"] == "no"),
        "unanswerable_now_answered": sorted(i for i in unanswerable
                                            if b[i]["verdict"].get("refusal_correct") != "yes"),
    }
    return {"criteria": criteria, "passes": all(c["pass"] for c in criteria.values()),
            "target_set": target, "gold_not_in_context_after": no_gold_after,
            "no_gold_outcomes": {"base": no_gold_outcomes(a), "new": no_gold_outcomes(b)},
            "by_group": by_group, "flips": flips,
            "noise_floor": base.get("noise"), "shape": {i: a[i].get("query_shape") for i in a}}


def _yes_rate(rows: dict, qtype: str) -> tuple[int, int] | None:
    graded = [r["verdict"] for r in rows.values() if r["qtype"] == qtype and _ok(r["verdict"])]
    return (sum(v["correct"] == "yes" for v in graded), len(graded)) if graded else None


def no_gold_outcomes(rows: dict) -> dict:
    """
    Answerable questions whose gold was not in the context, by what the answer
    did. Refusing is right for the answer step. "grounded_wrong" answers from
    the context but not the asked fact (a nearby question); "ungrounded" is
    guessing.
    """
    out = {k: [] for k in ("refused", "correct", "partial", "grounded_wrong", "ungrounded",
                           "judge_error")}
    for qid, r in sorted(rows.items()):
        if r["qtype"] == "unanswerable" or (r.get("gold_context") or {}).get("gold_in_context"):
            continue
        v = r["verdict"]
        if not _ok(v):
            out["judge_error"].append(qid)
        elif v.get("wrongly_refused"):
            out["refused"].append(qid)
        elif v["grounded"] == "no":
            out["ungrounded"].append(qid)
        elif v["correct"] == "yes":
            out["correct"].append(qid)
        elif v["correct"] == "partial":
            out["partial"].append(qid)
        else:
            out["grounded_wrong"].append(qid)
    return out


def _fmt(d: dict | None) -> str:
    if not d:
        return "-"
    star = " *" if d["excludes_zero"] else ""
    return (f"{100 * d['base']:.0f} -> {100 * d['new']:.0f} "
            f"({100 * d['diff']:+.1f} [{100 * d['ci_low']:+.1f},{100 * d['ci_high']:+.1f}]{star})")


def print_report(out: dict, base: dict, new: dict, warnings: list) -> None:
    shape = out["shape"]

    def tag(ids) -> str:
        return ", ".join(f"{i}({shape.get(i) or '?'})" for i in ids) or "none"

    print(f"COMPARE {out.get('base_name', 'dev')} vs dev_{new.get('tag')} (judge {new['judge_model']}, "
          f"rubric {new['judge_version']}); answer prompt "
          f"{base['answer_prompt_sha256'][:12]} -> {new['answer_prompt_sha256'][:12]}")
    for w in warnings:
        print(f"WARNING: {w}")
    print("\nSUCCESS CRITERIA")
    for name, c in out["criteria"].items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name.replace('_', ' ')}: {c['value']}")
    left = out["criteria"]["answer_step_refusals_remaining"]["remaining"]
    print(f"  target set ({len(out['target_set'])}): {tag(out['target_set'])}")
    print(f"  still refused: {tag(left)}")
    print(f"\nOVERALL: {'PASS' if out['passes'] else 'FAIL'}")

    print("\nBY GROUP (base -> new, paired difference [95% CI]; * excludes 0)")
    for name, g in out["by_group"].items():
        print(f"  {name:18s} n={g['n']:3d}  correct score {_fmt(g['correct_score'])}  "
              f"wrongly refused {_fmt(g['wrongly_refused'])}  length right "
              f"{_fmt(g['length_right'])}  median words {g['median_words'][0]} -> "
              f"{g['median_words'][1]}")

    print("\nFLIPS")
    for name, ids in out["flips"].items():
        print(f"  {name.replace('_', ' ')} ({len(ids)}): {tag(ids)}")

    print("\nBASELINE REFUSALS WITH NO GOLD IN CONTEXT (right for the answer step), now:")
    for i, status in out["gold_not_in_context_after"].items():
        print(f"  {i}({shape.get(i) or '?'}): {status}")
    guessing = [i for i, s in out["gold_not_in_context_after"].items() if "guessing" in s]
    print(f"  guessing (answered without grounding): {len(guessing)}")

    print("\nALL ANSWERABLE QUESTIONS WITH NO GOLD IN CONTEXT, by outcome (base -> new):")
    ng = out["no_gold_outcomes"]
    for kind in ng["new"]:
        b_ids, n_ids = ng["base"][kind], ng["new"][kind]
        if b_ids or n_ids:
            print(f"  {kind.replace('_', ' '):15s} {len(b_ids):2d} -> {len(n_ids):2d}: {tag(n_ids)}")

    noise = out.get("noise_floor") or {}
    if noise:
        print("\nNOISE FLOOR (baseline, 20 answers): regenerated-answer agreement "
              + ", ".join(f"{f} {v['agree']}/{v['n']}" for f, v in noise["regenerate"].items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base", default="dev",
                        help="results file to compare against (default: dev, the baseline; "
                             "the success criteria are defined against the baseline)")
    parser.add_argument("--split", default="dev", choices=("dev",))
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    with open(os.path.join(RESULTS_DIR, f"{args.base}.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    new_path = os.path.join(RESULTS_DIR, f"{args.split}_{args.tag}.json")
    with open(new_path, encoding="utf-8") as handle:
        new = json.load(handle)
    warnings = check_comparable(base, new)
    out = compare(base, new)
    out["base_name"] = args.base
    print_report(out, base, new, warnings)

    name = (f"compare_{args.split}_{args.tag}" if args.base == args.split
            else f"compare_{args.base}_vs_{args.tag}")
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"base_answer_prompt_sha256": base["answer_prompt_sha256"],
                   "new_answer_prompt_sha256": new["answer_prompt_sha256"],
                   "warnings": warnings, **{k: v for k, v in out.items() if k != "shape"}},
                  handle, indent=1)
        handle.write("\n")
    print(f"\nwritten: {os.path.relpath(path, PROJECT_ROOT)}")
    return 0 if out["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
