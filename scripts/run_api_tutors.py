"""Generate MRBench-style tutor responses with API models (Gemini / Claude).

Uses the same Figure-2 prompt as notebooks/molab_mrbench_multimodel_benchmark.py
(system prompt by Data == "Bridge" vs MathDial, one-sentence tutor response).
Output CSV columns match scripts/merge_llm_responses.py:
conversation_id, model_alias, method, model_id, response, ...

Resumable: rows already in --out are skipped, results are appended after every call.

Examples
  # Gemini API free tier (key from https://aistudio.google.com/apikey)
  set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...")
  python scripts/run_api_tutors.py --provider gemini --model gemini-2.5-flash \
      --alias Gemini --dataset data/tpaa_dataset/tpaa_mrbench_v1.json \
      --out data/tpaa_dataset/generations/api_responses.csv --sleep 6

  # Claude API (needs prepaid credits on platform.claude.com, separate from Claude Pro)
  set ANTHROPIC_API_KEY=...
  python scripts/run_api_tutors.py --provider anthropic --model claude-sonnet-5-5 \
      --alias Sonnet --dataset data/tpaa_dataset/tpaa_mrbench_v1.json \
      --out data/tpaa_dataset/generations/api_responses.csv

  # OpenRouter: one prepaid balance for Gemini, Claude, Llama-3.1-405B, GPT ...
  # (model ids exactly as listed on https://openrouter.ai/models)
  set OPENROUTER_API_KEY=...
  python scripts/run_api_tutors.py --provider openrouter --model meta-llama/llama-3.1-405b-instruct \
      --alias Llama31405B --dataset data/tpaa_dataset/tpaa_mrbench_v1.json \
      --out data/tpaa_dataset/generations/api_responses.csv

  Use --limit 3 first to check the output.
Requires: pip install google-genai (Gemini) / anthropic (Claude) / openai (OpenRouter)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPT_VERSION = "mrbench_fig2_paper_prompt_api_v1"


def build_prompt(d: dict) -> tuple[str, str]:
    """Return (system, user) exactly as the notebook builds them, with the
    assistant prefix appended as the last line of the user turn."""
    history = str(d.get("conversation_history", "")).strip()
    if d.get("Data") == "Bridge":
        system = ("You are an experienced elementary school math teacher, and you are going "
                  "to respond to a student's mistake in a useful and caring way")
        topic = str(d.get("Topic", "")).strip()
        user = (f"The problem your student is solving is on the topic: {topic}\n"
                f" Conversation History: {history}")
        prefix = ("Tutor response (maximum one sentence that is most appropriate given "
                  "topic and conversation history):")
    else:
        system = ("You are an experienced middle school math teacher, and you are going "
                  "to respond to a student's mistake in a useful and caring way")
        user = f"The conversation history of the problem your student is solving is: {history}"
        prefix = ("Tutor response (maximum one sentence that aligns most appropriately "
                  "with conversation history):")
    return system, f"{user}\n\n{prefix}"


from tutor_cleaning import CLEANING_VERSION, clean_response, qa_check  # noqa: E402


def make_caller(provider: str, model: str, max_tokens: int, thinking_budget: int | None,
                reasoning: dict | None = None):
    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        def call(system: str, user: str) -> tuple[str, dict]:
            cfg = dict(system_instruction=system, temperature=0.0, max_output_tokens=max_tokens)
            if thinking_budget is not None:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            r = client.models.generate_content(model=model, contents=user,
                                               config=types.GenerateContentConfig(**cfg))
            u = getattr(r, "usage_metadata", None)
            usage = {"input_tokens": getattr(u, "prompt_token_count", None),
                     "output_tokens": getattr(u, "candidates_token_count", None),
                     "thinking_tokens": getattr(u, "thoughts_token_count", None)}
            return r.text or "", usage
        return call

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def call(system: str, user: str) -> tuple[str, dict]:
            r = client.messages.create(model=model, system=system, max_tokens=max_tokens,
                                       temperature=0.0,
                                       messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, {"input_tokens": r.usage.input_tokens,
                          "output_tokens": r.usage.output_tokens, "thinking_tokens": None}
        return call

    if provider == "openrouter":
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        extra = {"usage": {"include": True}}  # OpenRouter returns the real billed cost
        if reasoning:
            extra["reasoning"] = reasoning

        def call(system: str, user: str) -> tuple[str, dict]:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=extra)
            u = r.usage
            details = getattr(u, "completion_tokens_details", None)
            return (r.choices[0].message.content or ""), {
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
                "thinking_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                "cost_usd": getattr(u, "cost", None),
                "served_model": getattr(r, "model", None)}
        return call

    raise SystemExit(f"unknown provider {provider}")


FIELDS = ["conversation_id", "Data", "model_alias", "method", "provider", "model_id",
          "prompt_version", "temperature", "max_output_tokens", "input_tokens",
          "output_tokens", "thinking_tokens", "cost_usd", "served_model", "reasoning",
          "seconds", "called_at_utc", "cleaning_version", "cleaning_actions", "qa_errors",
          "qa_info", "qa_status", "response_raw", "response"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gemini", "anthropic", "openrouter"], required=True)
    ap.add_argument("--model", required=True, help="exact API model id, e.g. gemini-2.5-flash")
    ap.add_argument("--alias", required=True, help="tutor key in the dataset, e.g. Gemini, Sonnet")
    ap.add_argument("--method", default="paper_prompt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Gemini only: 0 turns thinking off on models that allow it")
    ap.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"],
                    default=None, help="OpenRouter only: reasoning effort; 'none' disables it "
                                       "where the model allows (e.g. Gemini 2.5 Flash)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (free-tier RPM)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {(r["conversation_id"], r["model_alias"], r["method"]) for r in csv.DictReader(f)}
    reasoning = None
    if args.reasoning_effort == "none":
        reasoning = {"enabled": False}
    elif args.reasoning_effort:
        reasoning = {"effort": args.reasoning_effort, "exclude": True}
    call = make_caller(args.provider, args.model, args.max_tokens, args.thinking_budget, reasoning)

    todo = [d for d in data if (d["conversation_id"], args.alias, args.method) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} rows already in {out.name}; {len(todo)} to call.")
    for n, d in enumerate(todo, 1):
        system, user = build_prompt(d)
        for attempt in range(5):
            try:
                t0 = time.time()
                text, usage = call(system, user)
                break
            except Exception as e:  # rate limit / transient error -> back off
                wait = min(60, 5 * 2 ** attempt)
                print(f"  {d['conversation_id']}: {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on {d['conversation_id']} for now; rerun later to resume")
            continue
        row = dict(conversation_id=d["conversation_id"], Data=d["Data"], model_alias=args.alias,
                   method=args.method, provider=args.provider, model_id=args.model,
                   prompt_version=PROMPT_VERSION, temperature=0.0, max_output_tokens=args.max_tokens,
                   seconds=round(time.time() - t0, 2),
                   called_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   reasoning=json.dumps(reasoning) if reasoning else "")
        hit = usage.get("output_tokens") is not None and usage["output_tokens"] >= args.max_tokens
        answer, actions = clean_response(text, hit_max_tokens=hit)
        errors, info = qa_check(answer, method=args.method, hit_max_tokens=hit)
        row.update(cleaning_version=CLEANING_VERSION, cleaning_actions=";".join(actions),
                   qa_errors=";".join(errors), qa_info=";".join(info),
                   qa_status="error" if errors else "ok", response_raw=text, response=answer)
        row.update({k: v for k, v in usage.items() if k in FIELDS})
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerow(row)
        if errors:
            print(f"  QA ERROR {d['conversation_id']}: {';'.join(errors)} -> xem cột response_raw")
        if n % 10 == 0 or n == len(todo):
            spent = row.get("cost_usd")
            print(f"{n}/{len(todo)} done; last cost {spent}; last: {row['response'][:80]}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
