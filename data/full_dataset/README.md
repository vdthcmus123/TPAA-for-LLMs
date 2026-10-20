# Full math tutor dataset draft (D1–D4)

This draft contains **184 contexts and 184 tutor-response drafts**, with 46 contexts in each math domain. Each context has one student turn describing a misconception or an attempted answer and one context-specific tutor response draft. The schema follows the project's MRBench-based context and response fields. Revision full-d1-d4-draft-0.3 adds 24 context-response pairs to the previous 160-example set.

## Domain and grade-band allocation

- D1: Number, operations, and quantitative reasoning.
- D2: Algebra, functions, calculus, and linear algebra.
- D3: Geometry, measurement, and trigonometry.
- D4: Statistics, probability, discrete mathematics, and mathematical modeling.
- Every domain includes pre-K, primary, lower-secondary, upper-secondary, and tertiary examples; exact counts are in `quality_summary.json`.

## Files

- `contexts.csv`: one row per context.
- `tutor_response_drafts.csv`: one response draft per context; all eight MRBench annotation fields are blank.
- `dataset_draft.jsonl`: nested form combining each context and its response draft.
- `context_overlap_audit.csv`: exact text and nearest-neighbor checks against MRBench V1 and within this new set.
- `response_overlap_audit.csv`: exact text and nearest-neighbor checks against MRBench V1 tutor responses and within this new set.
- `quality_summary.json`: counts, audit thresholds, status, and release requirements.

## Provenance and review status

These are AI-assisted candidate examples. The tutor responses are written in a human-tutor style but have **not** been authored or approved by a human tutor. All 184 pairs were reviewed for mathematical consistency, a clear misconception, response-to-context alignment, actionable guidance, and topical distinction; 14 inherited draft rows were corrected during this pass. This is an AI review, not teacher sign-off. The team still needs to confirm mathematical correctness, pedagogical and age fit, and novelty. The annotation fields are empty; these drafts are not gold labels. Team attribution and a release license have not yet been decided.

## Duplicate screening

The audit compares all candidate context texts with all 192 MRBench V1 dialogue histories, and all candidate tutor drafts with all available MRBench V1 tutor responses (the source file contains 1,589 response entries). It records exact normalized matches and applies TF-IDF character n-gram screening. Context flags: source raw similarity 0.35, source number-normalized similarity 0.42, within-new-set similarity 0.62. Response flags: source similarity 0.55, within-new-set similarity 0.85. These thresholds are project screening rules, not paper-reported metrics. The current audit found 0 exact context matches and 0 exact tutor-response matches against MRBench; there are 0 internal context pairs and 0 internal response pairs at the project's near-duplicate thresholds. 7 context source-neighbor flags received AI review and were judged lexically similar but mathematically distinct (see `context_overlap_audit.csv`); team sign-off is still pending. Exact text checks and lexical scores cannot prove semantic uniqueness on their own.
