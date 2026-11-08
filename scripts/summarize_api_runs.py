"""Summarize API tutor runs: coverage, QA status and billed cost per tutor.

Usage:
  python scripts/summarize_api_runs.py data/tpaa_dataset/generations/api_responses.csv
"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "data/tpaa_dataset/generations/api_responses.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    groups[(r["model_alias"], r["model_id"])].append(r)

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

print(f"{'tutor':14s} {'model':40s} {'rows':>5s} {'convs':>5s} {'qa_err':>6s} {'words':>6s} {'cost_usd':>9s}")
total = 0.0
for (alias, model), g in groups.items():
    cost = sum(num(r.get("cost_usd")) for r in g)
    total += cost
    words = statistics.mean(len(r["response"].split()) for r in g)
    errs = sum(r.get("qa_status") == "error" for r in g)
    print(f"{alias:14s} {model:40s} {len(g):5d} {len({r['conversation_id'] for r in g}):5d} {errs:6d} {words:6.1f} {cost:9.4f}")
print(f"{'TOTAL':14s} {'':40s} {len(rows):5d} {'':5s} {'':6s} {'':6s} {total:9.4f}")
errs = [r for r in rows if r.get("qa_status") == "error"]
for r in errs[:20]:
    print(f"  QA error {r['model_alias']} {r['conversation_id']}: {r['qa_errors']} | {r['response_raw'][:100]!r}")
if len(errs) > 20:
    print(f"  ... {len(errs) - 20} more")
