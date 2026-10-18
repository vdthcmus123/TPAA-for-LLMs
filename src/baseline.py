#!/usr/bin/env python
"""Audit and train response-level baselines on MRBench V1."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "MRBench_V1.json"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "baseline"
LOCAL_ARTIFACTS_DIR = ROOT / ".cache" / "local_artifacts"

TARGET_LABELS: dict[str, tuple[str, ...]] = {
    "Mistake_Identification": ("Yes", "To some extent", "No"),
    "Mistake_Location": ("Yes", "To some extent", "No"),
    "Revealing_of_the_Answer": (
        "Yes (and the answer is correct)",
        "Yes (but the answer is incorrect)",
        "No",
    ),
    "Providing_Guidance": ("Yes", "To some extent", "No"),
    "Actionability": ("Yes", "To some extent", "No"),
    "Coherence": ("Yes", "To some extent", "No"),
    "Tutor_Tone": ("Encouraging", "Neutral", "Offensive"),
    "Humanlikeness": ("Yes", "To some extent", "No"),
}

PAPER_COUNTS = {
    "dialogues": 192,
    "dialogues_by_source": {"Bridge": 60, "MathDial": 132},
    "responses": 1596,
}

ANNOTATION_KEYS = {
    "mistakeidentification": "Mistake_Identification",
    "mistakelocation": "Mistake_Location",
    "revealingoftheanswer": "Revealing_of_the_Answer",
    "providingguidance": "Providing_Guidance",
    "actionability": "Actionability",
    "coherence": "Coherence",
    "tutortone": "Tutor_Tone",
    "humanlikeness": "Humanlikeness",
    "humanness": "Humanlikeness",
}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.I)


def _canonical_key(key: str) -> str | None:
    return ANNOTATION_KEYS.get(re.sub(r"[^a-z]", "", key.casefold()))


def normalize_label(target: str, value: Any) -> str | None:
    """Map release spelling/casing to the paper's canonical label schema."""
    if value is None:
        return None
    folded = re.sub(r"\s+", " ", str(value).strip().casefold())
    if target == "Revealing_of_the_Answer":
        if folded in {"no", "no answer", "does not reveal"}:
            return "No"
        if folded.startswith("yes"):
            if "incorrect" in folded or "wrong" in folded:
                return "Yes (but the answer is incorrect)"
            if "correct" in folded or folded == "yes":
                return "Yes (and the answer is correct)"
        return None
    if target == "Tutor_Tone":
        return {"encouraging": "Encouraging", "neutral": "Neutral", "offensive": "Offensive"}.get(folded)
    return {"yes": "Yes", "to some extent": "To some extent", "no": "No"}.get(folded)


def load_raw(path: Path = RAW_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Download the public MRBench V1 JSON "
            "from the URL in data/raw/README.md."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("MRBench V1 must be a JSON list of dialogue objects.")
    return data


def flatten_dataset(data: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    seen_dialogue_ids: set[str] = set()
    duplicate_dialogue_ids: list[str] = []
    source_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    model_counts: collections.Counter[str] = collections.Counter()

    for dialogue in data:
        conversation_id = str(dialogue.get("conversation_id", "")).strip()
        if not conversation_id:
            malformed.append({"reason": "missing conversation_id"})
            continue
        if conversation_id in seen_dialogue_ids:
            duplicate_dialogue_ids.append(conversation_id)
        seen_dialogue_ids.add(conversation_id)

        source = str(dialogue.get("Data", "Unknown"))
        source_counts[source] += 1
        split_counts[str(dialogue.get("Split", "missing"))] += 1
        history = str(dialogue.get("conversation_history", "")).strip()
        responses = dialogue.get("anno_llm_responses") or {}
        if not isinstance(responses, dict):
            malformed.append({"conversation_id": conversation_id, "reason": "responses not a map"})
            continue

        for tutor, response_obj in responses.items():
            response_obj = response_obj or {}
            response_text = str(response_obj.get("response", "")).strip()
            annotation = response_obj.get("annotation") or {}
            canonical = {target: None for target in TARGET_LABELS}
            for raw_key, raw_value in annotation.items():
                target = _canonical_key(str(raw_key))
                if target:
                    canonical[target] = normalize_label(target, raw_value)
            missing = [target for target, value in canonical.items() if value is None]
            if not history or not response_text or missing:
                malformed.append(
                    {
                        "conversation_id": conversation_id,
                        "tutor": str(tutor),
                        "reason": "missing content or annotation",
                        "missing_targets": missing,
                        "has_history": bool(history),
                        "has_response": bool(response_text),
                    }
                )
            rows.append(
                {
                    "example_id": f"{conversation_id}::{tutor}",
                    "conversation_id": conversation_id,
                    "source": source,
                    "provided_split": str(dialogue.get("Split", "missing")),
                    "topic": str(dialogue.get("Topic", "")),
                    "tutor": str(tutor),
                    "history": history,
                    "response": response_text,
                    "text": f"{history}\nTutor response: {response_text}",
                    "ground_truth_solution": str(dialogue.get("Ground_Truth_Solution", "")),
                    **canonical,
                }
            )
            model_counts[str(tutor)] += 1

    frame = pd.DataFrame(rows)
    actual_by_source = {key: int(value) for key, value in sorted(source_counts.items())}
    annotation_counts = {target: int(frame[target].notna().sum()) for target in TARGET_LABELS}
    label_distributions = {
        target: {str(label): int(count) for label, count in frame[target].value_counts(dropna=False).items()}
        for target in TARGET_LABELS
    }
    split_source = collections.Counter(
        (str(item.get("Split", "missing")), str(item.get("Data", "Unknown"))) for item in data
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dialogue in data:
        key = str(dialogue.get("conversation_id", "")).strip()
        if key:
            groups[key].append(dialogue)
    repeated_groups = []
    unique_source_counts: collections.Counter[str] = collections.Counter()
    for conversation_id, group in groups.items():
        unique_source_counts[str(group[0].get("Data", "Unknown"))] += 1
        if len(group) > 1:
            tutor_names = set().union(
                *(set((item.get("anno_llm_responses") or {}).keys()) for item in group)
            )
            repeated_groups.append(
                {
                    "conversation_id": conversation_id,
                    "objects": len(group),
                    "sources": sorted({str(item.get("Data", "Unknown")) for item in group}),
                    "provided_splits": sorted({str(item.get("Split", "missing")) for item in group}),
                    "same_history": len({str(item.get("conversation_history", "")) for item in group}) == 1,
                    "tutors_with_different_responses": sorted(
                        tutor
                        for tutor in tutor_names
                        if len(
                            {
                                json.dumps((item.get("anno_llm_responses") or {}).get(tutor), sort_keys=True)
                                for item in group
                            }
                            - {"null"}
                        )
                        > 1
                    ),
                }
            )
    audit = {
        "source_file": str(RAW_PATH.relative_to(ROOT)),
        "paper_reported": PAPER_COUNTS,
        "release_observed": {
            "dialogue_objects": len(data),
            "unique_conversation_ids": len(groups),
            "dialogues_by_source": actual_by_source,
            "unique_conversation_ids_by_source": dict(sorted(unique_source_counts.items())),
            "response_rows": len(frame),
            "tutor_responses_by_name": dict(sorted(model_counts.items())),
            "provided_split_dialogues": dict(sorted(split_counts.items())),
            "provided_split_by_source": {
                f"{split}|{source}": int(count) for (split, source), count in sorted(split_source.items())
            },
            "annotation_rows_by_dimension": annotation_counts,
            "label_distribution_including_missing": label_distributions,
            "dialogue_ids_unique": len(seen_dialogue_ids) == len(data),
            "duplicate_dialogue_ids": duplicate_dialogue_ids,
            "repeated_id_groups": repeated_groups,
            "malformed_response_rows": malformed,
        },
        "differences_from_paper": {
        "dialogue_objects": len(data) - PAPER_COUNTS["dialogues"],
        "unique_conversation_ids": len(groups) - PAPER_COUNTS["dialogues"],
            "responses": len(frame) - PAPER_COUNTS["responses"],
            "dialogues_by_source": {
                source: actual_by_source.get(source, 0) - expected
                for source, expected in PAPER_COUNTS["dialogues_by_source"].items()
            },
        },
        "normalization_notes": [
            "Raw humanlikeness key is normalized to Humanlikeness.",
            "Revealing-of-answer variants are mapped to the three paper labels.",
            "Missing/unknown labels are retained as null and excluded only for that target.",
        ],
    }
    return frame, audit


def prepare() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, audit = flatten_dataset(load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "mrbench_v1_responses.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, audit


def source_aware_split(frame: pd.DataFrame, test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Split dialogue IDs within each source, never splitting a conversation."""
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("--test-fraction must be greater than 0 and less than 0.5.")
    rng = np.random.default_rng(seed)
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for source in sorted(frame["source"].dropna().unique()):
        ids = np.array(sorted(frame.loc[frame.source == source, "conversation_id"].unique()))
        if len(ids) < 2:
            raise ValueError(f"Need at least 2 dialogue IDs for source {source!r}.")
        shuffled = rng.permutation(ids)
        n_test = max(1, int(round(len(ids) * test_fraction)))
        test_ids.update(str(value) for value in shuffled[:n_test])
        train_ids.update(str(value) for value in shuffled[n_test:])
    if train_ids & test_ids:
        raise AssertionError("Conversation leakage: train/test dialogue IDs overlap.")
    return train_ids, test_ids


def write_split_manifest(frame: pd.DataFrame, train_ids: set[str], test_ids: set[str], seed: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = frame[frame.conversation_id.isin(train_ids)]
    test_rows = frame[frame.conversation_id.isin(test_ids)]
    manifest = {
        "seed": seed,
        "method": "source-aware 80/20 split of conversation IDs; all tutor responses stay with their dialogue",
        "train_dialogues": int(len(train_ids)),
        "test_dialogues": int(len(test_ids)),
        "train_responses": int(len(train_rows)),
        "test_responses": int(len(test_rows)),
        "train_dialogues_by_source": {
            str(key): int(value) for key, value in train_rows.groupby("source").conversation_id.nunique().items()
        },
        "test_dialogues_by_source": {
            str(key): int(value) for key, value in test_rows.groupby("source").conversation_id.nunique().items()
        },
        "train_ids": sorted(train_ids),
        "test_ids": sorted(test_ids),
    }
    (REPORTS_DIR / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def label_ids(target: str, values: Iterable[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(TARGET_LABELS[target])}
    return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)


def score_target(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    labels = list(range(len(TARGET_LABELS[target])))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(TARGET_LABELS[target]),
        zero_division=0,
        output_dict=True,
    )
    return {
        "target": target,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": report,
    }


def save_results(model_name: str, scored: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "target": item["target"],
                "n_test": item["n_test"],
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                "weighted_f1": item["weighted_f1"],
            }
            for item in scored
        ]
    )
    summary.to_csv(REPORTS_DIR / f"metrics_{model_name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(REPORTS_DIR / f"predictions_{model_name}.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / f"metrics_{model_name}.json").write_text(
        json.dumps({"model": model_name, "dimensions": scored}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _prediction_rows(target: str, test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "example_id": row.example_id,
            "conversation_id": row.conversation_id,
            "source": row.source,
            "tutor": row.tutor,
            "target": target,
            "gold": TARGET_LABELS[target][int(gold)],
            "prediction": TARGET_LABELS[target][int(guess)],
        }
        for row, gold, guess in zip(test.itertuples(), y_true, y_pred)
    ]


def run_nb(train: pd.DataFrame, test: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    results, predictions = [], []
    for target in TARGET_LABELS:
        tr, te = train[train[target].notna()], test[test[target].notna()]
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True),
            MultinomialNB(alpha=0.3),
        )
        y_train, y_test = label_ids(target, tr[target]), label_ids(target, te[target])
        model.fit(tr.text.tolist(), y_train)
        pred = model.predict(te.text.tolist())
        results.append(score_target(target, y_test, pred))
        predictions.extend(_prediction_rows(target, te, y_test, pred))
        print(f"NB {target}: macro-F1={results[-1]['macro_f1']:.4f}")
    save_results("nb", results, predictions)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def train_sgns(
    sentences: list[list[str]], vector_size: int, window: int, min_count: int,
    negatives: int, epochs: int, seed: int,
) -> tuple[dict[str, int], np.ndarray]:
    """Train a compact Skip-Gram Negative Sampling (Word2Vec) model on train text only."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    counts = collections.Counter(token for sentence in sentences for token in sentence)
    vocab = [token for token, count in counts.most_common() if count >= min_count]
    if len(vocab) < 2:
        raise ValueError("Word2Vec vocabulary has fewer than two tokens.")
    word_to_id = {word: index for index, word in enumerate(vocab)}
    centers: list[int] = []
    contexts: list[int] = []
    for sentence in sentences:
        ids = [word_to_id[token] for token in sentence if token in word_to_id]
        for index, center in enumerate(ids):
            for other in range(max(0, index - window), min(len(ids), index + window + 1)):
                if other != index:
                    centers.append(center)
                    contexts.append(ids[other])
    if not centers:
        raise ValueError("Word2Vec produced no context pairs.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    center_ids = torch.tensor(centers, dtype=torch.long)
    context_ids = torch.tensor(contexts, dtype=torch.long)
    noise = np.asarray([counts[word] ** 0.75 for word in vocab], dtype=np.float64)
    noise /= noise.sum()
    noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
    input_embeddings = nn.Embedding(len(vocab), vector_size).to(device)
    output_embeddings = nn.Embedding(len(vocab), vector_size).to(device)
    nn.init.uniform_(input_embeddings.weight, -0.5 / vector_size, 0.5 / vector_size)
    nn.init.zeros_(output_embeddings.weight)
    optimizer = torch.optim.Adam(list(input_embeddings.parameters()) + list(output_embeddings.parameters()), lr=0.003)
    rng = np.random.default_rng(seed)
    batch_size = 2048
    pair_count = len(center_ids)
    input_embeddings.train()
    output_embeddings.train()
    for epoch in range(epochs):
        order = rng.permutation(pair_count)
        loss_sum, batch_count = 0.0, 0
        for start in range(0, pair_count, batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            center_batch, context_batch = center_ids[indices].to(device), context_ids[indices].to(device)
            negative_batch = torch.multinomial(
                noise_tensor, num_samples=len(center_batch) * negatives, replacement=True
            ).view(len(center_batch), negatives)
            positive_score = (input_embeddings(center_batch) * output_embeddings(context_batch)).sum(dim=1)
            negative_score = torch.bmm(
                output_embeddings(negative_batch), input_embeddings(center_batch).unsqueeze(2)
            ).squeeze(2)
            loss = -F.logsigmoid(positive_score).mean() - F.logsigmoid(-negative_score).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
        print(f"Word2Vec epoch {epoch + 1}/{epochs}: pairs={pair_count}, loss={loss_sum / max(batch_count, 1):.4f}")
    return word_to_id, input_embeddings.weight.detach().cpu().numpy().astype(np.float32)


def mean_word_vectors(contexts: list[str], responses: list[str], word_to_id: dict[str, int], vectors: np.ndarray) -> np.ndarray:
    features = np.zeros((len(contexts), vectors.shape[1]), dtype=np.float32)
    for index, (context, response) in enumerate(zip(contexts, responses)):
        ids = [word_to_id[token] for token in tokenize(context + " " + response) if token in word_to_id]
        if ids:
            features[index] = vectors[ids].mean(axis=0)
    return features


def run_rf(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> None:
    from sklearn.ensemble import RandomForestClassifier

    sentences = [tokenize(value) for value in train.history.tolist()]
    sentences.extend(tokenize(value) for value in train.response.tolist())
    word_to_id, word_vectors = train_sgns(
        sentences, args.w2v_dim, args.w2v_window, args.w2v_min_count,
        args.w2v_negatives, args.w2v_epochs, args.seed,
    )
    x_train = mean_word_vectors(train.history.tolist(), train.response.tolist(), word_to_id, word_vectors)
    x_test = mean_word_vectors(test.history.tolist(), test.response.tolist(), word_to_id, word_vectors)
    results, predictions = [], []
    for target in TARGET_LABELS:
        tr_mask, te_mask = train[target].notna().to_numpy(), test[target].notna().to_numpy()
        tr, te = train.loc[tr_mask], test.loc[te_mask]
        y_train, y_test = label_ids(target, tr[target]), label_ids(target, te[target])
        model = RandomForestClassifier(
            n_estimators=args.rf_trees, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed,
        )
        model.fit(x_train[tr_mask], y_train)
        pred = model.predict(x_test[te_mask])
        results.append(score_target(target, y_test, pred))
        predictions.extend(_prediction_rows(target, te, y_test, pred))
        print(f"RF+Word2Vec {target}: macro-F1={results[-1]['macro_f1']:.4f}")
    save_results("rf_word2vec", results, predictions)


def _bert_encode_pairs(tokenizer: Any, rows: pd.DataFrame, max_length: int) -> dict[str, Any]:
    import torch

    input_ids, attention_masks, token_type_ids = [], [], []
    special_count = tokenizer.num_special_tokens_to_add(pair=True)
    for context, response in zip(rows.history.tolist(), rows.response.tolist()):
        context_ids = tokenizer.encode(str(context), add_special_tokens=False)
        response_ids = tokenizer.encode(str(response), add_special_tokens=False)
        available = max_length - special_count
        if len(response_ids) > available // 2:
            response_ids = response_ids[: available // 2]
        context_budget = max(1, available - len(response_ids))
        # Keep recent turns, including the student's mistake, for long histories.
        context_ids = context_ids[-context_budget:]
        # Transformers 5.x removed the old tokenizer helper methods. Construct the
        # standard BERT pair layout explicitly: [CLS] context [SEP] response [SEP].
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None or tokenizer.pad_token_id is None:
            raise ValueError("The selected checkpoint must provide BERT CLS/SEP/PAD token IDs.")
        ids = [cls_id] + context_ids + [sep_id] + response_ids + [sep_id]
        types = [0] * (len(context_ids) + 2) + [1] * (len(response_ids) + 1)
        if len(ids) > max_length:
            # Preserve all of the candidate tutor response and trim the oldest
            # context tokens if decoding/tokenizer behavior exceeded the budget.
            overflow = len(ids) - max_length
            context_ids = context_ids[overflow:]
            ids = [cls_id] + context_ids + [sep_id] + response_ids + [sep_id]
            types = [0] * (len(context_ids) + 2) + [1] * (len(response_ids) + 1)
        mask = [1] * len(ids)
        padding = max_length - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        attention_masks.append(mask + [0] * padding)
        token_type_ids.append(types + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    }


def run_bert(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BERT checkpoint {args.bert_model!r} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, local_files_only=args.offline)
    encoder = AutoModel.from_pretrained(args.bert_model, local_files_only=args.offline)

    class MultiTaskBert(nn.Module):
        def __init__(self, model: nn.Module, hidden_size: int):
            super().__init__()
            self.encoder = model
            self.heads = nn.ModuleDict(
                {name: nn.Linear(hidden_size, len(labels)) for name, labels in TARGET_LABELS.items()}
            )

        def forward(self, input_ids, attention_mask, token_type_ids):
            output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            pooled = output.last_hidden_state[:, 0]
            return {name: head(pooled) for name, head in self.heads.items()}

    hidden_size = getattr(encoder.config, "hidden_size", None) or getattr(encoder.config, "dim")
    model = MultiTaskBert(encoder, int(hidden_size)).to(device)
    train_features = _bert_encode_pairs(tokenizer, train, args.max_length)
    test_features = _bert_encode_pairs(tokenizer, test, args.max_length)
    train_labels = {
        name: torch.tensor(label_ids(name, train[name]), dtype=torch.long) for name in TARGET_LABELS
    }
    dataset = TensorDataset(
        train_features["input_ids"], train_features["attention_mask"], train_features["token_type_ids"],
        *[train_labels[name] for name in TARGET_LABELS],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.bert_learning_rate)
    model.train()
    for epoch in range(args.bert_epochs):
        total_loss, steps = 0.0, 0
        for batch in loader:
            ids, mask, types, *label_batch = [item.to(device) for item in batch]
            logits = model(ids, mask, types)
            loss = sum(
                nn.functional.cross_entropy(logits[name], y)
                for name, y in zip(TARGET_LABELS, label_batch)
            ) / len(TARGET_LABELS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        print(f"BERT epoch {epoch + 1}/{args.bert_epochs}: loss={total_loss / max(steps, 1):.4f}")

    model.eval()
    test_dataset = TensorDataset(
        test_features["input_ids"], test_features["attention_mask"], test_features["token_type_ids"]
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    collected: dict[str, list[int]] = {name: [] for name in TARGET_LABELS}
    with torch.no_grad():
        for ids, mask, types in test_loader:
            output = model(ids.to(device), mask.to(device), types.to(device))
            for name in TARGET_LABELS:
                collected[name].extend(output[name].argmax(dim=1).cpu().tolist())
    results, predictions = [], []
    for target in TARGET_LABELS:
        y_test = label_ids(target, test[target])
        pred = np.asarray(collected[target], dtype=np.int64)
        results.append(score_target(target, y_test, pred))
        predictions.extend(_prediction_rows(target, test, y_test, pred))
        print(f"BERT {target}: macro-F1={results[-1]['macro_f1']:.4f}")
    save_results("bert", results, predictions)
    LOCAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_name": args.bert_model, "labels": TARGET_LABELS},
        LOCAL_ARTIFACTS_DIR / "bert_checkpoint.pt",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="normalize MRBench and write data audit")
    run = subparsers.add_parser("run", help="run one or all supervised baselines")
    run.add_argument("--model", choices=("nb", "rf", "bert", "all"), default="all")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--test-fraction", type=float, default=0.2)
    run.add_argument("--w2v-dim", type=int, default=100)
    run.add_argument("--w2v-window", type=int, default=2)
    run.add_argument("--w2v-min-count", type=int, default=1)
    run.add_argument("--w2v-negatives", type=int, default=5)
    run.add_argument("--w2v-epochs", type=int, default=5)
    run.add_argument("--rf-trees", type=int, default=300)
    run.add_argument("--bert-model", default="google/bert_uncased_L-2_H-128_A-2")
    run.add_argument("--bert-epochs", type=int, default=3)
    run.add_argument("--bert-learning-rate", type=float, default=2e-5)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--offline", action="store_true", help="do not download a missing model checkpoint")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    try:
        frame, audit = prepare()
        if args.command == "audit":
            observed = audit["release_observed"]
            print(f"Dialogue objects: {observed['dialogue_objects']}")
            print(f"Unique conversation IDs: {observed['unique_conversation_ids']}")
            print(f"Response rows: {observed['response_rows']}")
            print(f"Sources: {observed['dialogues_by_source']}")
            print(f"Repeated conversation IDs: {len(observed['repeated_id_groups'])}")
            print(f"Malformed response rows: {len(observed['malformed_response_rows'])}")
            print(f"Audit written to {REPORTS_DIR / 'data_audit.json'}")
            return 0

        train_ids, test_ids = source_aware_split(frame, args.test_fraction, args.seed)
        write_split_manifest(frame, train_ids, test_ids, args.seed)
        train = frame[frame.conversation_id.isin(train_ids)].reset_index(drop=True)
        test = frame[frame.conversation_id.isin(test_ids)].reset_index(drop=True)
        target_columns = list(TARGET_LABELS)
        if train[target_columns].isna().any().any() or test[target_columns].isna().any().any():
            raise ValueError("One or more target labels are missing; inspect data_audit.json first.")
        print(
            f"Conversation split: train={len(train_ids)} dialogues/{len(train)} responses, "
            f"test={len(test_ids)} dialogues/{len(test)} responses"
        )
        selected = ["nb", "rf", "bert"] if args.model == "all" else [args.model]
        for model_name in selected:
            if model_name == "nb":
                run_nb(train, test)
            elif model_name == "rf":
                run_rf(train, test, args)
            else:
                run_bert(train, test, args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
