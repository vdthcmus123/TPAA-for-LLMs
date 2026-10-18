# MRBench V1 baseline results

## Data and split audit

| Check | Result |
|---|---:|
| Dialogue objects in current GitHub V1 JSON | 192 |
| Unique `conversation_id`s | 188 |
| Response-level rows | 1,589 |
| Fully labeled rows per taxonomy dimension | 1,589 |
| Source counts in paper | 60 Bridge / 132 MathDial |
| Source counts in current release | 53 Bridge / 139 MathDial |
| Train | 150 unique IDs / 1,275 response rows |
| Test | 38 unique IDs / 314 response rows |
| Train source mix | 39 Bridge / 111 MathDial IDs |
| Test source mix | 10 Bridge / 28 MathDial IDs |

Four Bridge `conversation_id`s are repeated in the release with the same
history but different tutor-response variants. They are retained as distinct
response rows and assigned together to one split. The paper's reported counts
and the current V1 file do not match; the baseline uses the released JSON as-is
and records the discrepancy in `data_audit.json`. The provided `Split` field is
not used because it places all MathDial dialogue objects in `test` and is
therefore confounded with source.

## Models

- **Naive Bayes:** word-level TF-IDF, unigrams and bigrams, MultinomialNB,
  `alpha=0.3`.
- **Random Forest + Word2Vec:** 100-dimensional Skip-Gram with Negative
  Sampling vectors, trained only on training text; mean context/response
  vectors; 300-tree Random Forest with balanced subsampling.
- **BERT:** Google compact BERT checkpoint (2 layers, hidden size 128),
  fine-tuned for 3 epochs with eight classification heads. For long contexts,
  input construction retains the most recent dialogue tokens and the tutor
  response.

The same source-aware split and 314 test responses are used for each method.
Reported results are from one fixed split (seed 42), without hyperparameter
search; treat them as an initial baseline, not a stable estimate of expected
performance.

## Macro-F1 by dimension

| Dimension | Naive Bayes | RF + Word2Vec | BERT |
|---|---:|---:|---:|
| Mistake identification | 0.3122 | **0.3757** | 0.2929 |
| Mistake location | **0.4740** | 0.4483 | 0.4109 |
| Revealing the answer | 0.3057 | **0.4075** | 0.3358 |
| Providing guidance | 0.3482 | **0.4007** | 0.3277 |
| Actionability | 0.4129 | **0.4199** | 0.3676 |
| Coherence | 0.2975 | **0.3129** | 0.2975 |
| Tutor tone | 0.2949 | **0.3663** | 0.3174 |
| Human-likeness | 0.3076 | 0.3076 | 0.3076 |
| **Mean across dimensions** | **0.3441** | **0.3799** | **0.3322** |

The mean here is an unweighted descriptive average across eight tasks, not a
metric reported by the paper. Macro-F1 is the primary comparison because the
labels are imbalanced. For example, the release contains only one `Offensive`
label, so that class cannot support a meaningful estimate of tutor-tone
performance. Inspect `metrics_<model>.json` for accuracy, weighted-F1, and
per-class support; inspect `predictions_<model>.csv` for errors.

## Interpretation

On this split, Random Forest + Word2Vec has the highest mean macro-F1, while
Naive Bayes leads on mistake location. BERT does not win this initial run. This
is not evidence that one model family is generally superior: the test set has
only 38 independent conversation IDs, several rare labels, and just one split.
Use grouped cross-validation or repeated source-aware splits before drawing
strong conclusions. These classification scores are not directly comparable
with the paper's DAMR or Annotation Correlation metrics.
