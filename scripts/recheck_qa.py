"""Re-run the current QA rules on an existing generations CSV (no API calls, no cost).

Only qa_errors / qa_info / qa_status are recomputed; responses are not changed.
A line is printed for every row whose status changes.

Usage:
  python scripts/recheck_qa.py data/tpaa_dataset/generations/api_responses.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tutor_cleaning import qa_check  # noqa: E402

path = Path(sys.argv[1])
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header, rows = reader.fieldnames, list(reader)

def as_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0

changed = 0
for r in rows:
    method = r.get("method", "")
    if "max_output_tokens" in r:          # API runs
        hit = as_int(r.get("output_tokens")) >= as_int(r.get("max_output_tokens")) > 0
        errors, info = qa_check(r["response"], method="" if method == "paper_prompt" else method, hit_max_tokens=hit)
    else:                                 # moLab open-model runs
        errors, info = qa_check(
            r["response"], method=method, thinking_tokens=as_int(r.get("thinking_tokens")),
            reasoning_closed=r.get("reasoning_closed") == "True", hit_max_tokens=r.get("hit_max_tokens") == "True",
        )
        if r.get("thinking_budget_forced") == "True":
            info.append("thinking_budget_forced")
    status = "error" if errors else "ok"
    if status != r.get("qa_status"):
        changed += 1
        print(f"{r.get('model_alias')} {r['conversation_id']}: {r.get('qa_status')} -> {status} {';'.join(errors)}")
    r["qa_errors"], r["qa_info"], r["qa_status"] = ";".join(errors), ";".join(info), status

with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows rechecked, {changed} changed status, {sum(r['qa_status'] == 'error' for r in rows)} errors remain.")
