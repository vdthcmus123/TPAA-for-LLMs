# Project reports

- [`benchmark_results_project_plan.md`](benchmark_results_project_plan.md):
  Vietnamese summary of all completed baseline/moLab metrics, paper comparison,
  limitations, and next steps against the course rubric. Generated from the
  moLab ZIP with `python scripts/generate_benchmark_report.py --bundle <zip>`.
- [`baseline/`](baseline/): normalized MRBench audit, fixed split manifest,
  per-dimension metrics, test predictions, and baseline summary.

- [`mrbench_molab_outputs.zip`](mrbench_molab_outputs.zip): moLab output bundle of
  the MRBench run (960 generated responses, 2,549 Qwen judgements). Regenerate the
  report with `python scripts/generate_benchmark_report.py --bundle reports/mrbench_molab_outputs.zip`.

Large model checkpoints stay outside Git. The report stores the input ZIP SHA-256
for provenance.
