"""Merge generated LLM tutor responses into the TPAA MRBench-format dataset.

Input CSV(s) need the columns: conversation_id, response, and either
`tutor` or `model_alias` (+ optional `method`). This matches the output of
notebooks/molab_mrbench_multimodel_benchmark.py (multimodel_fig2_responses.csv);
API runs can use the same columns.

Each response is added under anno_llm_responses[<tutor key>] with an empty
8-dimension annotation, and a fresh annotation sheet (one row per response) is written.

Usage:
  python scripts/merge_llm_responses.py \
      --dataset data/tpaa_dataset/tpaa_mrbench_v1.json \
      --generations path/to/multimodel_fig2_responses.csv [--generations api_runs.csv] \
      --out data/tpaa_dataset/tpaa_mrbench_v1_with_llm.json \
      --sheet data/tpaa_dataset/annotation_sheet_with_llm.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DIMS = ["Mistake_Identification", "Mistake_Location", "Revealing_of_the_Answer",
        "Providing_Guidance", "Actionability", "humanlikeness", "Coherence", "Tutor_Tone"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--generations", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--allow-qa-errors", action="store_true",
                    help="also merge rows whose qa_status is 'error' (default: skip them)")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    by_id = {d["conversation_id"]: d for d in data}

    rows = []
    for p in args.generations:
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    methods = defaultdict(set)
    for r in rows:
        methods[r.get("tutor") or r["model_alias"]].add(r.get("method", ""))

    added, skipped, empty = Counter(), Counter(), 0
    for r in rows:
        if r.get("qa_status") == "error" and not args.allow_qa_errors:
            skipped[f"qa_error {r.get('qa_errors', '')}"] += 1
            print(f"  skip QA error: {r['conversation_id']} / {r.get('tutor') or r['model_alias']} "
                  f"{r.get('method', '')}: {r.get('qa_errors', '')}")
            continue
        cid = r["conversation_id"]
        if cid not in by_id:
            skipped["unknown conversation_id"] += 1
            continue
        base = r.get("tutor") or r["model_alias"]
        key = base if len(methods[base]) <= 1 else f"{base}_{r.get('method', '')}"
        text = (r.get("response") or "").strip()
        if not text:
            empty += 1
        resp = by_id[cid]["anno_llm_responses"]
        if key in resp:
            skipped[f"duplicate {key}"] += 1
            continue
        resp[key] = {"response": text, "annotation": {d: "" for d in DIMS}}
        added[key] += 1

    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(args.sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["conversation_id", "Data", "Split", "tutor", "response", *DIMS,
                    "annotator_1", "annotator_2", "adjudicator", "notes"])
        for d in data:
            for t, r in d["anno_llm_responses"].items():
                w.writerow([d["conversation_id"], d["Data"], d["Split"], t, r["response"],
                            *[r["annotation"].get(x, "") for x in DIMS], "", "", "", ""])
    total = sum(len(d["anno_llm_responses"]) for d in data)
    missing = {k: len(data) - v for k, v in added.items() if v < len(data)}
    print(json.dumps({"added_per_tutor": added, "skipped": skipped, "empty_responses": empty,
                      "tutors_missing_some_conversations": missing,
                      "total_responses_now": total}, indent=1, default=dict))


if __name__ == "__main__":
    main()
