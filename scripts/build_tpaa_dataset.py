"""Build, validate and audit the TPAA MRBench-format dataset (v1).

Outputs (default: data/tpaa_dataset/):
  tpaa_mrbench_v1.json        MRBench V1 schema; Expert (+Novice) responses with
                              empty 8-dimension annotations, ready for multi-LLM
                              response generation and human annotation.
  tpaa_metadata_v1.csv        Sidecar provenance per conversation (not in the
                              MRBench schema, so the main file stays identical in shape).
  annotation_sheet_v1.csv     One row per (conversation, tutor) response to label.
  overlap_audit_v1.csv        Nearest MRBench neighbour + within-set neighbour.
  quality_summary_v1.json     All checks and statistics.
  draft_disposition_v0.3.csv  What happened to each of the 184 earlier drafts.

Usage:
  python scripts/build_tpaa_dataset.py --mrbench data/raw/MRBench_V1.json \
      --drafts data/tpaa_dataset/drafts_v0.3/dataset_draft.jsonl --out data/tpaa_dataset
  Optional: --gsm8k path/to/gsm8k_train.jsonl [--gsm8k path/to/gsm8k_test.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tpaa_items_mathdial_a import ITEMS_A  # noqa: E402
from tpaa_items_mathdial_b import ITEMS_B  # noqa: E402
from tpaa_items_bridge import ITEMS_BRIDGE  # noqa: E402
from tpaa_verbose_solutions import VERBOSE  # noqa: E402
from tpaa_explanations import EXPLAIN, PROBES  # noqa: E402
from tpaa_expert_revisions import EXPERT_OVERRIDE  # noqa: E402

VERSION = "tpaa-mrbench-v1.0"
SEP = " \n "  # turn separator used in MRBench V1
OPENER = ("Hi, could you please provide a step-by-step solution for the question below? "
          "The question is: ")
DIMS = ["Mistake_Identification", "Mistake_Location", "Revealing_of_the_Answer",
        "Providing_Guidance", "Actionability", "humanlikeness", "Coherence", "Tutor_Tone"]
SEED = 42

# ---------------------------------------------------------------- arithmetic
NUM = r"-?\$?-?\d[\d,]*(?:\.\d+)?%?"
EQ = re.compile(r"([0-9$.,%()\s+\-*/x×÷]+?)=\s*(" + NUM + r")(?:/(\d+))?(\s*remainder)?")


def _num(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").replace("%", ""))


def check_equations(text: str) -> list[dict]:
    """Evaluate every 'expr = value' in text; return list of results."""
    out = []
    for m in EQ.finditer(text):
        left, right, den, rem = m.group(1), m.group(2), m.group(3), m.group(4)
        if rem:
            continue
        if not re.search(r"\d", left) or not re.search(r"\d\s*[+\-*/x×÷]\s*\(?\s*-?\$?\d", left):
            continue
        expr = left.replace("$", "").replace(",", "").replace("%", "")
        expr = expr.replace("×", "*").replace("÷", "/").replace("x", "*")
        expr = expr.strip()
        # drop a leading unmatched ')' or operator fragment
        expr = re.sub(r"^[^0-9(\-]+", "", expr)
        try:
            lhs = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - digits/operators only
        except Exception:
            continue
        rhs = _num(right) / (float(den) if den else 1.0)
        tol = max(0.011, abs(rhs) * 0.005)
        out.append(dict(expr=expr, lhs=round(lhs, 4), rhs=rhs, ok=abs(lhs - rhs) <= tol))
    return out


def norm_ans(a: str) -> str:
    return re.sub(r"[\s$,%]", "", a.lower()).rstrip(".").removesuffix(".00").removesuffix("0") \
        if re.fullmatch(r"-?[\d.,$%\s]+", a.strip()) else a.strip().lower()


# ---------------------------------------------------------------- builders
def mathdial_history(it: dict) -> str:
    sol = "\n".join(it["stu"])
    # Answer line is joined with "\n\u00a0" as in 119/139 MRBench MathDial histories
    # (the remaining ones put it in a separate "\u00a0\n\u00a0" segment).
    parts = [f"Tutor: {OPENER}{it['problem']}", f"Student: {sol}\n\u00a0{it['sans']}"]
    parts += [f"{r}: {t}" for r, t in it["turns"]]
    return SEP.join(parts)


def bridge_history(it: dict) -> str:
    return SEP.join(f"{r}: {t}" for r, t in it["turns"])


def empty_annotation() -> dict:
    return {d: "" for d in DIMS}


def assign_splits(ids: list[str], seed: int, frac=(0.7, 0.1, 0.2)) -> dict:
    rng = random.Random(seed)
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = round(n * frac[0]), round(n * frac[1])
    return {cid: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
            for i, cid in enumerate(ids)}


def words(s: str) -> int:
    return len(s.split())


def n_turns(h: str) -> int:
    return len(re.findall(r"(?:^|\n |\n)(Tutor|Student):", h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrbench", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm8k", action="append", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md_items = [dict(it) for it in ITEMS_A + ITEMS_B]
    for i, it in enumerate(md_items, start=1):
        if i in VERBOSE:
            it["stu"] = VERBOSE[i]
        turns = list(it["turns"])
        if i in EXPLAIN:
            assert turns[1][0] == "Student"
            turns[1] = ("Student", EXPLAIN[i])
        it["turns"] = turns
        it["expert"] = EXPERT_OVERRIDE.get(("MD", i), it["expert"])
    br_items = [dict(it) for it in ITEMS_BRIDGE]
    for i, it in enumerate(br_items, start=1):
        it["expert"] = EXPERT_OVERRIDE.get(("BR", i), it["expert"])
    k = 0
    for it in md_items:
        if it["turns"][0][1] == "Can you walk me through your solution?":
            it["turns"][0] = ("Tutor", PROBES[k % len(PROBES)])
            k += 1
    if len(VERBOSE) != len(md_items):
        raise SystemExit("verbose solution missing for some items")
    issues: list[str] = []
    dataset, meta = [], []

    md_ids = [f"TPAA-MD-{i:03d}" for i in range(1, len(md_items) + 1)]
    br_ids = [f"TPAA-BR-{i:03d}" for i in range(1, len(br_items) + 1)]
    split = {**assign_splits(md_ids, SEED), **assign_splits(br_ids, SEED + 1)}

    # ---- MathDial-style
    eq_total = eq_bad_gt = eq_bad_stu = 0
    for cid, it in zip(md_ids, md_items):
        gt_checks = [c for s in it["gt"] for c in check_equations(s)]
        st_checks = [c for s in it["stu"] for c in check_equations(s)]
        turn_checks = [c for r, t in it["turns"] if r == "Student" for c in check_equations(t)]
        eq_total += len(gt_checks) + len(st_checks) + len(turn_checks)
        for c in gt_checks:
            if not c["ok"]:
                eq_bad_gt += 1
                issues.append(f"{cid}: ground-truth arithmetic {c}")
        bad_st = [c for c in st_checks + turn_checks if not c["ok"]]
        eq_bad_stu += len(bad_st)
        if bad_st and not it["slip"]:
            issues.append(f"{cid}: unintended student arithmetic error {bad_st}")
        if it["slip"] and not bad_st:
            issues.append(f"{cid}: marked slip but no arithmetic error found")
        if norm_ans(it["ans"]) == norm_ans(it["sans"]):
            issues.append(f"{cid}: student answer equals ground truth")
        if it["turns"] and it["turns"][-1][0] != "Student":
            issues.append(f"{cid}: last turn is not Student")
        hist = mathdial_history(it)
        gts = "\n".join(it["gt"]) + "\n " + it["ans"]
        dataset.append({
            "conversation_id": cid,
            "conversation_history": hist,
            "Data": "MathDial",
            "Split": split[cid],
            "Topic": "Not Available",
            "Ground_Truth_Solution": gts,
            "anno_llm_responses": {
                "Expert": {"response": it["expert"], "annotation": empty_annotation()},
            },
        })
        meta.append(dict(conversation_id=cid, style="MathDial", split=split[cid], topic="",
                         ground_truth_answer=it["ans"], student_answer=it["sans"],
                         misconception=it["mis"], intended_arithmetic_slip=it["slip"],
                         derived_from_draft=it["derived"] or "",
                         n_turns=n_turns(hist), history_words=words(hist),
                         expert_words=words(it["expert"]), has_novice=False,
                         equations_checked=len(gt_checks) + len(st_checks) + len(turn_checks)))

    # ---- Bridge-style
    for cid, it in zip(br_ids, br_items):
        if it["turns"][-1][0] != "Student":
            issues.append(f"{cid}: last turn is not Student")
        hist = bridge_history(it)
        dataset.append({
            "conversation_id": cid,
            "conversation_history": hist,
            "Data": "Bridge",
            "Split": split[cid],
            "Topic": it["topic"],
            "Ground_Truth_Solution": "Not Available",
            "anno_llm_responses": {
                "Expert": {"response": it["expert"], "annotation": empty_annotation()},
                "Novice": {"response": it["novice"], "annotation": empty_annotation()},
            },
        })
        meta.append(dict(conversation_id=cid, style="Bridge", split=split[cid], topic=it["topic"],
                         ground_truth_answer=it["correct"], student_answer=it["turns"][-1][1],
                         misconception=it["mis"], intended_arithmetic_slip=False,
                         derived_from_draft=it["derived"] or "",
                         n_turns=n_turns(hist), history_words=words(hist),
                         expert_words=words(it["expert"]), has_novice=True,
                         equations_checked=0))

    # ---- schema check against MRBench
    mr = json.loads(Path(args.mrbench).read_text(encoding="utf-8"))
    mr_keys = list(mr[0].keys())
    for d in dataset:
        if list(d.keys()) != mr_keys:
            issues.append(f"{d['conversation_id']}: key order differs from MRBench")
        for tutor, r in d["anno_llm_responses"].items():
            if list(r.keys()) != ["response", "annotation"] or list(r["annotation"].keys()) != DIMS:
                issues.append(f"{d['conversation_id']}/{tutor}: response schema differs")
    ids = [d["conversation_id"] for d in dataset]
    if len(set(ids)) != len(ids):
        issues.append("duplicate conversation_id")

    # ---- overlap audit (char n-gram TF-IDF, same method as the earlier draft audit)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def clean(s):
        return re.sub(r"\s+", " ", s.replace(" ", " ")).strip().lower()

    def problem_text(d):
        h = d["conversation_history"]
        if d["Data"] == "MathDial":
            return clean(h.split(OPENER, 1)[1].split(" \n Student:", 1)[0])
        return clean(h)

    def mr_problem(e):
        h = e["conversation_history"]
        if "The question is:" in h:
            return clean(h.split("The question is:", 1)[1].split("Student:", 1)[0])
        return clean(h)

    new_txt = [problem_text(d) for d in dataset]
    mr_txt = [mr_problem(e) for e in mr]
    ext_txt, ext_src = [], []
    for p in args.gsm8k:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                ext_txt.append(clean(json.loads(line)["question"]))
                ext_src.append(Path(p).name)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vec.fit(new_txt + mr_txt + ext_txt)
    A, Bm = vec.transform(new_txt), vec.transform(mr_txt)
    sim_mr = cosine_similarity(A, Bm)
    sim_in = cosine_similarity(A)
    for i in range(len(new_txt)):
        sim_in[i, i] = 0
    sim_ext = cosine_similarity(A, vec.transform(ext_txt)) if ext_txt else None

    mr_resp = [clean(r["response"]) for e in mr for r in e["anno_llm_responses"].values()]
    new_resp = [(d["conversation_id"], t, clean(r["response"]))
                for d in dataset for t, r in d["anno_llm_responses"].items()]
    vec_r = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(mr_resp + [x[2] for x in new_resp])
    sim_resp = cosine_similarity(vec_r.transform([x[2] for x in new_resp]), vec_r.transform(mr_resp))
    exact_resp = sum(x[2] in set(mr_resp) for x in new_resp)
    exact_ctx = sum(t in set(mr_txt) for t in new_txt)

    TH_MR, TH_IN, TH_RESP = 0.60, 0.70, 0.80
    audit_rows = []
    for i, d in enumerate(dataset):
        j = int(sim_mr[i].argmax())
        k = int(sim_in[i].argmax())
        row = dict(conversation_id=d["conversation_id"], style=d["Data"],
                   nearest_mrbench_id=mr[j]["conversation_id"], sim_mrbench=round(float(sim_mr[i, j]), 3),
                   nearest_mrbench_text=mr_txt[j][:160],
                   nearest_internal_id=dataset[k]["conversation_id"], sim_internal=round(float(sim_in[i, k]), 3))
        if sim_ext is not None:
            e = int(sim_ext[i].argmax())
            row.update(nearest_gsm8k_text=ext_txt[e][:160], sim_gsm8k=round(float(sim_ext[i, e]), 3))
        flag = []
        if row["sim_mrbench"] >= TH_MR:
            flag.append("mrbench")
        if row["sim_internal"] >= TH_IN:
            flag.append("internal")
        if sim_ext is not None and row["sim_gsm8k"] >= TH_MR:
            flag.append("gsm8k")
        row["flag"] = ";".join(flag)
        audit_rows.append(row)
    resp_flags = [(x[0], x[1], round(float(sim_resp[i].max()), 3))
                  for i, x in enumerate(new_resp) if sim_resp[i].max() >= TH_RESP]

    # ---- surface-form checks
    md = [d for d in dataset if d["Data"] == "MathDial"]
    br = [d for d in dataset if d["Data"] == "Bridge"]
    first_tutor_after_solution = Counter(it["turns"][0][1] for it in md_items)
    bridge_openers = Counter(it["turns"][0][1] for it in br_items)
    expert_dupes = [k for k, v in Counter(clean(d["anno_llm_responses"]["Expert"]["response"]) for d in dataset).items() if v > 1]
    if expert_dupes:
        issues.append(f"duplicate Expert responses: {expert_dupes}")

    # ---- statistics vs MRBench
    def stats(items, src=None):
        it = [e for e in items if src is None or e["Data"] == src]
        h = [e["conversation_history"] for e in it]
        exp = [e["anno_llm_responses"]["Expert"]["response"] for e in it]
        nov = [e["anno_llm_responses"]["Novice"]["response"] for e in it if "Novice" in e["anno_llm_responses"]]
        last = Counter(re.findall(r"(Tutor|Student):", x)[-1] for x in h)
        return {
            "conversations": len(it),
            "mean_turns": round(st.mean(n_turns(x) for x in h), 2),
            "mean_history_words": round(st.mean(words(x) for x in h), 1),
            "median_history_words": st.median(words(x) for x in h),
            "last_turn_student_pct": round(100 * last["Student"] / len(it), 1),
            "expert_mean_words": round(st.mean(words(x) for x in exp), 1),
            "expert_question_pct": round(100 * sum(x.strip().endswith("?") for x in exp) / len(exp), 1),
            "novice_mean_words": round(st.mean(words(x) for x in nov), 1) if nov else None,
            "distinct_topics": len({e["Topic"] for e in it}) if src == "Bridge" else None,
            "split": dict(Counter(e["Split"] for e in it)),
        }

    def resp_count(items):
        return sum(len(e["anno_llm_responses"]) for e in items)

    summary = {
        "version": VERSION,
        "conversations": len(dataset),
        "by_style": dict(Counter(d["Data"] for d in dataset)),
        "human_style_responses_included": resp_count(dataset),
        "expected_responses_after_7_llm_tutors": resp_count(dataset) + 7 * len(dataset),
        "stats_new": {"MathDial": stats(dataset, "MathDial"), "Bridge": stats(dataset, "Bridge")},
        "stats_mrbench_v1": {"MathDial": stats(mr, "MathDial"), "Bridge": stats(mr, "Bridge")},
        "checks": {
            "schema_matches_mrbench_v1": not any("schema" in s or "key order" in s for s in issues),
            "equations_machine_checked": eq_total,
            "ground_truth_equation_errors": eq_bad_gt,
            "student_equation_errors_total": eq_bad_stu,
            "student_equation_errors_unintended": sum("unintended" in s for s in issues),
            "student_answer_equals_truth": sum("equals ground truth" in s for s in issues),
            "last_turn_not_student": sum("last turn" in s for s in issues),
            "exact_context_matches_mrbench": exact_ctx,
            "exact_response_matches_mrbench": exact_resp,
            "context_flags_vs_mrbench_ge_%.2f" % TH_MR: sum("mrbench" in r["flag"] for r in audit_rows),
            "context_flags_internal_ge_%.2f" % TH_IN: sum("internal" in r["flag"] for r in audit_rows),
            "context_flags_vs_gsm8k": (sum("gsm8k" in r["flag"] for r in audit_rows) if ext_txt else "not run (pass --gsm8k)"),
            "response_flags_vs_mrbench_ge_%.2f" % TH_RESP: len(resp_flags),
            "max_context_similarity_mrbench": round(float(sim_mr.max()), 3),
            "max_context_similarity_internal": round(float(sim_in.max()), 3),
            "most_repeated_mathdial_probe": first_tutor_after_solution.most_common(3),
            "most_repeated_bridge_opener": bridge_openers.most_common(2),
            "duplicate_expert_responses": len(expert_dupes),
        },
        "response_flags": resp_flags,
        "issues": issues,
        "annotation_status": "all 8 dimensions empty; to be labelled by the team",
    }

    # ---- draft disposition
    used = Counter()
    for m in meta:
        if m["derived_from_draft"]:
            used[m["derived_from_draft"]] = m["conversation_id"]
    disp = []
    for line in Path(args.drafts).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cid = r["context_id"]
        if cid in used:
            action, reason = "rewritten", f"Misconception reused and rewritten into {used[cid]} in MRBench format"
        elif r["grade_band"] in ("pre_k", "tertiary"):
            action, reason = "removed", "Outside MRBench scope (paper covers primary/middle-school math)"
        elif r["grade_band"] == "upper_secondary":
            action, reason = "removed", "Upper-secondary topic outside MRBench scope (MathDial middle school, Bridge elementary)"
        else:
            action, reason = "removed", "2-turn templated context; not converted (MathDial-style needs multi-step word problems, Bridge-style quota filled)"
        disp.append(dict(draft_context_id=cid, grade_band=r["grade_band"], action=action, reason=reason))
    summary["draft_disposition"] = dict(Counter(x["action"] for x in disp))

    # ---- write
    (out / "tpaa_mrbench_v1.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "tpaa_metadata_v1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    with open(out / "annotation_sheet_v1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["conversation_id", "Data", "Split", "tutor", "response", *DIMS,
                    "annotator_1", "annotator_2", "adjudicator", "notes"])
        for d in dataset:
            for t, r in d["anno_llm_responses"].items():
                w.writerow([d["conversation_id"], d["Data"], d["Split"], t, r["response"], *[""] * len(DIMS), "", "", "", ""])
    with open(out / "overlap_audit_v1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    with open(out / "draft_disposition_v0.3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(disp[0].keys()))
        w.writeheader()
        w.writerows(disp)
    (out / "quality_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("conversations", "by_style", "checks", "issues", "draft_disposition")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
