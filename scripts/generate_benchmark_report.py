#!/usr/bin/env python
"""Build the project report from the moLab output bundle and local baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = {
    "Mistake_Identification": "Nhận diện lỗi",
    "Mistake_Location": "Xác định vị trí lỗi",
    "Revealing_of_the_Answer": "Có tiết lộ đáp án",
    "Providing_Guidance": "Cung cấp hướng dẫn",
    "Actionability": "Tính hành động",
    "Coherence": "Mạch lạc",
    "Tutor_Tone": "Giọng điệu gia sư",
    "Humanlikeness": "Tính tự nhiên như người",
}
DIMENSION_ORDER = list(DIMENSIONS)
TUTOR_ORDER = [
    "Novice", "Expert", "Llama-3.1-8B", "Phi3", "Gemini", "Sonnet",
    "Mistral", "GPT-4", "Llama-3.1-405B",
]
MODEL_KEYS = [
    ("llama31_8b", "paper_prompt"),
    ("phi3_mini", "paper_prompt"),
    ("mistral_7b", "paper_prompt"),
    ("qwen35_9b", "zero_shot"),
    ("qwen35_9b", "cot"),
]
MODEL_LABELS = {
    ("llama31_8b", "paper_prompt"): "Llama 3.1 8B",
    ("phi3_mini", "paper_prompt"): "Phi-3 Mini",
    ("mistral_7b", "paper_prompt"): "Mistral 7B",
    ("qwen35_9b", "zero_shot"): "Qwen3.5 Zero-shot",
    ("qwen35_9b", "cot"): "Qwen3.5 CoT",
}


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


def load_bundle(path: Path) -> tuple[dict[str, pd.DataFrame], dict, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    names = [
        "paper_vs_current_release_damr.csv", "qwen38_ac_vs_paper.csv",
        "qwen38_generated_damr.csv", "qwen38_generated_damr_vs_paper.csv",
        "qwen38_release_damr_vs_paper.csv", "qwen38_judgements.csv",
        "multimodel_fig2_responses.csv",
    ]
    with ZipFile(path) as archive:
        manifest = json.load(archive.open("MANIFEST.json"))
        for item in manifest["files"]:
            payload = archive.read(item["name"])
            if len(payload) != item["size_bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Bundle manifest mismatch for {item['name']}")
        frames = {Path(name).stem: pd.read_csv(archive.open(name)) for name in names}
    return frames, manifest, digest


def build_report(bundle_path: Path) -> str:
    frames, manifest, digest = load_bundle(bundle_path)
    paper_release = frames["paper_vs_current_release_damr"]
    ac = frames["qwen38_ac_vs_paper"]
    generated = frames["qwen38_generated_damr"]
    generated_vs_paper = frames["qwen38_generated_damr_vs_paper"]
    release = frames["qwen38_release_damr_vs_paper"]
    judgements = frames["qwen38_judgements"]
    responses = frames["multimodel_fig2_responses"]
    baseline = {
        key: pd.read_csv(ROOT / "reports" / "baseline" / f"metrics_{key}.csv")
        for key in ["nb", "rf_word2vec", "bert"]
    }

    baseline_rows = []
    for dimension, label in DIMENSIONS.items():
        cells = []
        for key in ["nb", "rf_word2vec", "bert"]:
            row = baseline[key].set_index("target").loc[dimension]
            cells.append(f"{row.macro_f1:.3f} / {row.accuracy:.3f} / {row.weighted_f1:.3f}")
        baseline_rows.append([label, *cells])
    baseline_summary = []
    for key, name in [("nb", "Naive Bayes"), ("rf_word2vec", "RF + Word2Vec"), ("bert", "BERT nhỏ")]:
        values = baseline[key]
        baseline_summary.append([
            name, f"{values.macro_f1.mean():.4f}", f"{values.accuracy.mean():.4f}",
            f"{values.weighted_f1.mean():.4f}",
        ])

    generated = generated.copy()
    generated["weighted_hits"] = generated.qwen_judge_DAMR_pct * generated.n_qwen_judged
    pooled = generated.groupby(
        ["model_alias", "model_name", "method", "method_name", "dimension"], as_index=False
    ).agg(weighted_hits=("weighted_hits", "sum"), n=("n_qwen_judged", "sum"))
    pooled["damr"] = pooled.weighted_hits / pooled.n
    pivot = pooled.pivot(index="dimension", columns=["model_alias", "method"], values="damr")
    generated_rows = [
        [DIMENSIONS[d], *[f"{pivot.loc[d, key]:.2f}%" for key in MODEL_KEYS]]
        for d in DIMENSION_ORDER
    ]
    config_columns = [
        "model_alias", "model_name", "method", "model_id", "model_revision", "dtype",
        "do_sample", "prompt_version", "enable_thinking", "max_input_tokens", "max_new_tokens",
    ]
    model_configs = responses[config_columns].drop_duplicates().sort_values(["model_alias", "method"])
    condition_names = {
        "paper_prompt": "Paper Figure 2 prompt",
        "zero_shot": "Zero-shot (thinking off)",
        "cot": "CoT (thinking on)",
    }
    config_rows = [[
        row.model_name, row.model_id, row.model_revision,
        condition_names[row.method],
        f"{row.dtype}; do_sample={row.do_sample}; thinking={row.enable_thinking}",
        int(row.max_input_tokens), int(row.max_new_tokens),
    ] for row in model_configs.itertuples(index=False)]
    source_rows = []
    for alias, method in MODEL_KEYS:
        selected = generated[(generated.model_alias == alias) & (generated.method == method)]
        for source in ["Bridge", "MathDial"]:
            rows = selected[selected.source == source]
            source_rows.append([
                MODEL_LABELS[(alias, method)], source, int(rows.n_qwen_judged.iloc[0]),
                f"{rows.qwen_judge_DAMR_pct.mean():.2f}%",
            ])

    release_detail = paper_release.merge(
        release[[
            "tutor", "dimension", "n_qwen_judged", "qwen_judge_DAMR_pct",
            "qwen_minus_paper_pp", "qwen_minus_release_human_pp",
        ]], on=["tutor", "dimension"], validate="one_to_one",
    )
    release_detail["tutor_order"] = release_detail.tutor.map({name: i for i, name in enumerate(TUTOR_ORDER)})
    release_detail["dimension_order"] = release_detail.dimension.map({name: i for i, name in enumerate(DIMENSION_ORDER)})
    release_detail = release_detail.sort_values(["tutor_order", "dimension_order"])
    release_rows = [[
        row.tutor, DIMENSIONS[row.dimension], int(row.n_release), f"{row.paper_DAMR_pct:.2f}%",
        f"{row.release_DAMR_pct:.2f}%", f"{row.qwen_judge_DAMR_pct:.2f}%",
        f"{row.release_minus_paper_pp:+.2f}", f"{row.qwen_minus_paper_pp:+.2f}",
        f"{row.qwen_minus_release_human_pp:+.2f}",
    ] for row in release_detail.itertuples(index=False)]

    ac = ac.copy()
    ac["tutor_order"] = ac.tutor.map({name: i for i, name in enumerate(TUTOR_ORDER)})
    ac["dimension_order"] = ac.dimension.map({name: i for i, name in enumerate(DIMENSION_ORDER)})
    ac = ac.sort_values(["tutor_order", "dimension_order"])
    ac_rows = [[
        row.tutor, DIMENSIONS[row.dimension], int(row.n_paired), f"{row.qwen_pearson_AC:.3f}",
        f"{row['paper_AC_Prometheus2']:.3f}", f"{row['paper_AC_Llama-3.1-8B critic']:.3f}",
    ] for _, row in ac.iterrows()]

    current_release_summary = []
    for tutor, rows in paper_release.groupby("tutor"):
        current_release_summary.append([
            tutor, f"{rows.paper_DAMR_pct.mean():.2f}%", f"{rows.release_DAMR_pct.mean():.2f}%",
            f"{rows.release_minus_paper_pp.mean():+.2f}",
            f"{rows.release_minus_paper_pp.abs().mean():.2f}",
        ])
    tutor_rank = {name: i for i, name in enumerate(TUTOR_ORDER)}
    current_release_summary.sort(key=lambda row: tutor_rank.get(row[0], 100))

    generated_comparisons = []
    shared = generated_vs_paper.dropna(subset=["paper_human_DAMR_pct"])
    for (model, method), rows in shared.groupby(["model_name", "method_name"], sort=False):
        generated_comparisons.append([
            f"{model} — {method}", f"{rows.qwen_judge_DAMR_pct.mean():.2f}%",
            f"{rows.paper_human_DAMR_pct.mean():.2f}%",
            f"{rows.qwen_judge_minus_paper_pp.mean():+.2f}",
            f"{rows.qwen_judge_minus_paper_pp.abs().mean():.2f}",
        ])
    shared = shared.copy()
    shared["dimension_order"] = shared.dimension.map({name: i for i, name in enumerate(DIMENSION_ORDER)})
    shared = shared.sort_values(["model_alias", "method", "dimension_order"])
    generated_detail_rows = [[
        row.model_name, row.method_name, DIMENSIONS[row.dimension], int(row.n_qwen_judged),
        f"{row.qwen_judge_DAMR_pct:.2f}%", f"{row.paper_human_DAMR_pct:.2f}%",
        f"{row.qwen_judge_minus_paper_pp:+.2f}",
    ] for row in shared.itertuples(index=False)]

    ac_dimension = ac.groupby("dimension").agg(
        qwen=("qwen_pearson_AC", "mean"), paper_prometheus=("paper_AC_Prometheus2", "mean"),
        paper_llama=("paper_AC_Llama-3.1-8B critic", "mean"),
    ).reindex(DIMENSION_ORDER)
    ac_dimension_rows = [[
        DIMENSIONS[name], f"{row.qwen:.3f}", f"{row.paper_prometheus:.3f}", f"{row.paper_llama:.3f}"
    ] for name, row in ac_dimension.iterrows()]
    ac_tutor = ac.groupby("tutor").agg(
        qwen=("qwen_pearson_AC", "mean"), paper_prometheus=("paper_AC_Prometheus2", "mean"),
        paper_llama=("paper_AC_Llama-3.1-8B critic", "mean"), n=("n_paired", "min"),
    ).reindex(TUTOR_ORDER)
    ac_tutor_rows = [[
        tutor, int(row.n), f"{row.qwen:.3f}", f"{row.paper_prometheus:.3f}", f"{row.paper_llama:.3f}"
    ] for tutor, row in ac_tutor.iterrows()]

    generated_macro = pooled.groupby(["model_alias", "method"]).damr.mean()
    judge_release = int((judgements.evaluation_set == "release_ac").sum())
    judge_generated = int((judgements.evaluation_set == "generated_damr").sum())
    source_counts = generated.groupby("source").n_qwen_judged.first().to_dict()
    qwen_ac = ac.qwen_pearson_AC
    qwen_release_mean = release.qwen_judge_DAMR_pct.mean()
    qwen_paper_mean = release.paper_DAMR_pct.mean()
    qwen_paper_mae = release.qwen_minus_paper_pp.abs().mean()
    qwen_human_delta = release.qwen_minus_release_human_pp.mean()
    qwen_human_mae = release.qwen_minus_release_human_pp.abs().mean()
    parse_count = int(judgements.parse_ok.sum())
    judge_hours = judgements.judge_seconds.sum() / 3600
    today = pd.Timestamp.now().strftime("%d/%m/%Y")

    return f'''# MRBench: kết quả thực nghiệm, đối chiếu paper và kế hoạch đồ án

**Ngày chốt số liệu:** {today}  
**ZIP đầu vào:** `{bundle_path.name}`  
**SHA-256 ZIP:** `{digest}`. Manifest có {len(manifest["files"])} file và đã xác minh checksum/kích thước.  
**Nguyên tắc:** phân biệt nhãn người trong release với nhãn do Qwen judge dự đoán; chỉ gọi là tái lập gần khi dữ liệu, checkpoint hoặc judge khác paper.

## 1. Tóm tắt

Repo đã chạy ba baseline phân loại tám nhãn MRBench: Naive Bayes + TF-IDF, Random Forest + Word2Vec SGNS, và BERT nhỏ đa nhiệm. Cả ba dùng cùng test set 314 response, chia theo hội thoại. Macro-F1 trung bình tám chiều lần lượt là **0.344, 0.380 và 0.332**. Đây là bài toán dự đoán nhãn, khác với DAMR và AC của paper.

Notebook moLab sinh **{len(responses)} response** trên 192 hội thoại với năm cấu hình (ba mô hình theo prompt Figure 2 và Qwen3.5-9B Zero-shot/CoT), sau đó Qwen3.8-27B chấm **{len(judgements)} response**: {judge_release} response gốc để tính AC và {judge_generated} response mới để tính DAMR proxy. Có {parse_count}/{len(judgements)} judgement parse thành công.

DAMR tính lại từ nhãn người của release nhìn chung khá gần Table 3 ở phần lớn tutor nhưng lệch đáng kể ở Gemini và một số chiều của Expert/Novice. Qwen judge có Pearson AC trung bình mô tả **{qwen_ac.mean():.3f}** trên 72 ô tutor×dimension, so với **{ac.paper_AC_Prometheus2.mean():.3f}** của Prometheus2 và **{ac['paper_AC_Llama-3.1-8B critic'].mean():.3f}** của Llama critic trong các bảng paper. Đây chưa phải so sánh cùng điều kiện. DAMR trên response mới hiện do Qwen judge chấm, chưa có xác nhận người.

## 2. Dataset, thiết lập và phạm vi táي lập

| Hạng mục | Thiết lập / số liệu |
| --- | --- |
| MRBench V1 trong repo/ZIP | 192 dialogue objects; 139 MathDial, 53 Bridge; 188 conversation ID duy nhất; 1.589 response gốc có nhãn |
| Paper | 132 MathDial + 60 Bridge; 192 hội thoại; 1.596 response (7 LLM × 192 + Expert × 192 + Novice × 60) |
| Sai khác release | So với paper, MathDial nhiều hơn 7 và Bridge ít hơn 7; Novice hiện có 53 response thay vì 60. Bốn ID Bridge lặp với history giống nhau nhưng response variant khác nhau; baseline giữ response và gom cùng split |
| Baseline split | Seed 42; train 150 conversation IDs / 1.275 response; test 38 IDs / 314 response; giữ nguyên hội thoại trong một split và phân tầng nguồn |
| Response sinh mới | 960 = 5 cấu hình × 192 dialogue; `do_sample=False`; BF16; Qwen3.5 thinking off cho Zero-shot, thinking on cho CoT |
| Judge | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 2.549 judgement; checkpoint theo response |
| Tài nguyên moLab | Ảnh phiên chạy hiển thị 4 CPU, 32 GiB RAM, RTX Pro 6000. Tổng `judge_seconds` cộng dồn khoảng {judge_hours:.1f} giờ, không phải wall-clock của phiên |

Model ID, revision và cấu hình generation được lưu trong từng dòng CSV. Các revision dưới đây là commit Hugging Face được notebook dùng ở lần chạy này:

{markdown_table(["Model", "Hugging Face ID", "Revision", "Condition", "dtype / sampling", "max input tokens", "max new tokens"], config_rows)}

Paper mô tả bốn annotator đã huấn luyện và coi human labels là gold. So sánh trong báo cáo dùng JSON release hiện tại, không phải snapshot gốc đã được xác minh trùng từng dòng với paper.

## 3. Baseline phân loại

Input của mỗi ví dụ là conversation history và tutor response; output là dự đoán riêng cho tám dimension. Ô ghi **Macro-F1 / Accuracy / Weighted-F1**. Macro-F1 là metric chính do lệch lớp.

{markdown_table(["Dimension", "Naive Bayes + TF-IDF", "RF + Word2Vec", "BERT nhỏ"], baseline_rows)}

Trung bình không trọng số qua tám chiều:

{markdown_table(["Mô hình", "Macro-F1 mean", "Accuracy mean", "Weighted-F1 mean"], baseline_summary)}

RF + Word2Vec dẫn đầu mean macro-F1 trong split này (0.3799), NB tốt nhất ở Mistake Location; BERT chưa vượt baseline cổ điển. Đây mới là một split cố định; một số lớp hiếm (Offensive chỉ có một nhãn trong audit), nên không khái quát hóa mạnh. Paper không báo cáo các metric phân loại này; không đối chiếu F1/accuracy trực tiếp với DAMR hoặc AC.

Thiết lập: NB dùng TF-IDF unigram/bigram và MultinomialNB (`alpha=0.3`); RF dùng Word2Vec Skip-Gram with Negative Sampling 100 chiều học trên train, mean pooling và Random Forest 300 cây; BERT dùng `google/bert_uncased_L-2_H-128_A-2`, tám classification heads, fine-tune 3 epoch. Tham số đầy đủ, predictions và split manifest nằm trong `reports/baseline/`.

## 4. DAMR nhãn người: release so với paper Table 3

Paper định nghĩa DAMR là tỷ lệ response đạt nhãn mong muốn: Yes cho nhận diện/vị trí lỗi, hướng dẫn, actionability, coherence và human-likeness; No cho tiết lộ đáp án; Encouraging cho tone. Bảng này tính lại DAMR từ nhãn người sẵn có trong release. MAE là sai số tuyệt đối trung bình tám chiều.

{markdown_table(["Tutor", "Paper mean 8 chiều", "Release human mean", "Delta signed (pp)", "MAE (pp)"], current_release_summary)}

Trên toàn bộ 72 ô, release trung bình cao hơn paper **{paper_release.release_minus_paper_pp.mean():+.2f} pp**, MAE **{paper_release.release_minus_paper_pp.abs().mean():.2f} pp**. Gemini lệch trung bình +22.92 pp; riêng Expert ở Tutor Tone lệch -75.00 pp, còn Novice có MAE 7.64 pp và chỉ có 53 mẫu hiện tại thay vì 60. Đây là khác biệt đủ lớn để ưu tiên kiểm tra snapshot, join tutor/response, mã hóa tone và nhãn trước khi diễn giải như khác biệt thực chất về tutor.

## 5. DAMR do Qwen judge chấm

### 5.1 Response gốc đã có nhãn người

Qwen chấm {judge_release} response gốc và so với gold label hiện có. Trung bình unweighted 72 ô Qwen-judged DAMR là **{qwen_release_mean:.2f}%**, so với **{qwen_paper_mean:.2f}%** ở Table 3; signed difference **{release.qwen_minus_paper_pp.mean():+.2f} pp**, MAE **{qwen_paper_mae:.2f} pp**. So với nhãn người cùng release, signed difference là **{qwen_human_delta:+.2f} pp**, nhưng MAE vẫn **{qwen_human_mae:.2f} pp**. Mean gần 0 che lỗi theo chiều: Tutor Tone MAE 28.72 pp, Mistake Identification 17.57 pp; Revealing Answer MAE 1.39 pp.

### 5.2 960 response mới

Các điểm là tỷ lệ đạt nhãn mong muốn do Qwen3.8-27B phán đoán; chưa có human gold. Ba model đầu trùng họ với model paper và dùng prompt Figure 2. Qwen3.5 là phần mở rộng. Gộp Bridge/MathDial có trọng số theo số mẫu của từng nguồn ({int(source_counts.get('Bridge', 0))} Bridge, {int(source_counts.get('MathDial', 0))} MathDial).

{markdown_table(["Dimension", "Llama 3.1 8B", "Phi-3 Mini", "Mistral 7B", "Qwen3.5 Zero-shot", "Qwen3.5 CoT"], generated_rows)}

So sánh trung bình tám chiều với paper human DAMR Table 3 (chỉ ba model tương ứng):

{markdown_table(["Model / condition", "Qwen-judged mean", "Paper human mean", "Delta mean (pp)", "MAE 8 dims (pp)"], generated_comparisons)}

So sánh chi tiết từng dimension của ba model có trong Table 3. Model sinh response là cùng họ/checkpoint gần tương ứng; cột generated được Qwen judge chấm còn paper là human DAMR:

<details><summary>Mở bảng 24 ô generated DAMR so với paper</summary>

{markdown_table(["Model", "Condition", "Dimension", "N", "Qwen-judged generated %", "Paper human %", "Delta (pp)"], generated_detail_rows)}

</details>

Llama 3.1 8B gần paper nhất về macro mean (-1.04 pp), nhưng MAE theo chiều 13.28 pp. Phi-3 Mini cao hơn paper +7.29 pp (MAE 15.10 pp); Mistral thấp hơn -43.49 pp (MAE 51.95 pp). Qwen judge khác annotator gold và checkpoint/decoding paper không đầy đủ, nên không kết luận trực tiếp tốt hơn/xấu hơn.

Qwen3.5 Zero-shot đạt mean 70.96%, CoT 72.33% (+1.37 pp). CoT tăng tỷ lệ đạt “không tiết lộ đáp án” 15.10 pp nhưng giảm Mistake Identification 3.65 pp và Mistake Location 4.17 pp. Đây là mô tả trên cùng 192 dialogue; chưa kiểm định ý nghĩa thống kê hoặc xác nhận nhãn người.

Mean DAMR theo nguồn và tám dimensions:

{markdown_table(["Model / condition", "Source", "N", "Mean DAMR"], source_rows)}

Zero-shot tốt hơn trên Bridge (77.36%) so với MathDial (68.53%); CoT đạt 71.70% trên Bridge và 72.57% trên MathDial. Luôn báo cáo theo nguồn vì độ dài/bối cảnh và độ khó khác nhau.

## 6. Annotation Correlation (AC)

Paper dùng Pearson giữa critic annotations và human annotations, báo riêng từng tutor×dimension cho Prometheus2 và Llama-3.1-8B critic ở Tables 5–6. Notebook tính Pearson tương tự cho Qwen3.8 so với human labels hiện tại. Paper không nêu cách mã hóa nhãn phân loại thành số; notebook dùng ordinal 1–3 theo thứ tự nhãn trong rubric, nên so sánh là gần đúng. Pearson trên ordinal không đo agreement tuyệt đối và không thay thế Cohen/Fleiss kappa.

Trên 72 ô, Qwen AC mean={qwen_ac.mean():.3f}, median={qwen_ac.median():.3f}; {(qwen_ac > 0).sum()}/72 ô dương, min={qwen_ac.min():.3f}, max={qwen_ac.max():.3f}. Mean số học các ô paper trong CSV là Prometheus2 {ac.paper_AC_Prometheus2.mean():.3f}, Llama critic {ac['paper_AC_Llama-3.1-8B critic'].mean():.3f}. Đây không phải chỉ số gộp do paper công bố; số mẫu, Qwen judge và mã hóa khác làm hạn chế so sánh.

Mean theo dimension:

{markdown_table(["Dimension", "Qwen3.8 r", "Prometheus2 paper r", "Llama critic paper r"], ac_dimension_rows)}

Mean theo tutor (N là số cặp hiện tại, Novice=53; paper dùng 60):

{markdown_table(["Tutor", "N", "Qwen3.8 r", "Prometheus2 paper r", "Llama critic paper r"], ac_tutor_rows)}

Qwen AC cao nhất ở Revealing Answer (mean r=0.840), thấp nhất Humanlikeness (mean r=0.100). Cần kiểm tra nhãn mù/human agreement theo chiều, đặc biệt Tone và Humanlikeness.

## 7. Việc tiếp theo theo rubric giảng viên

| Ưu tiên | Yêu cầu | Việc cần làm | Đầu ra |
| --- | --- | --- | --- |
| 1 | Bắt buộc: đọc và trình bày paper | Trình bày bài toán/input/output, tám chiều và desired labels, Bridge/MathDial, prompt Figure 2, human annotation, DAMR/AC, kết quả và giới hạn; nêu lệch release | Slide, script, citation |
| 2 | Bắt buộc: thuật toán NLP cơ bản | Giữ NB, RF+Word2Vec, BERT; bổ sung grouped cross-validation hoặc nhiều seed, per-class metrics/confusion matrix/error analysis | Kết quả ổn định, phân tích lỗi |
| 3 | Bắt buộc: phương pháp LLM mới | Qwen3.5 Zero-shot/CoT đã chạy local HF; mô tả đây không phải API. Khóa revision, prompt, decoding và giới hạn token | Bảng LLM tái lập và cấu hình |
| 4 | Bắt buộc: dataset mới 250 mẫu/người | Chốt unit là dialogue hay response; tiêu chí thu thập hội thoại toán có lỗi/confusion; lưu nguồn, quyền sử dụng và loại bỏ PII | Dataset mới, data card, pipeline |
| 5 | Bắt buộc: gán nhãn | Viết rubric/examples; pilot 20–30 mẫu; ít nhất hai người gán độc lập một phần dữ liệu; đo Cohen/Fleiss kappa và phân xử. LLM chỉ hỗ trợ gợi ý, người xác nhận gold | Guideline, labels, agreement, adjudication log |
| 6 | Tự chọn: đánh giá phương pháp trên data mới | Group split theo dialogue/source; chạy cùng baseline và Qwen Zero-shot/CoT; báo tám dimension, support và lỗi | MRBench vs data nhóm |
| 7 | Tự chọn: thêm phương pháp paper | Nếu khả thi, chạy Figure 6 critic prompt bằng Prometheus2 và Llama-3.1-8B critic như paper; hoặc so sánh Qwen judge với hai annotator trên cùng mẫu. Báo Pearson cùng agreement tuyệt đối | Reliability study |

Nếu nhóm có bốn sinh viên, mục tiêu 250 mẫu/người tương ứng 1.000 mẫu sau gộp; cần thống nhất trước một “mẫu” là dialogue hay tutor response để không tính lẫn hai đơn vị.

### Lộ trình thao tác

1. Đóng băng MRBench V1 bằng hash, ghi rõ 139/53 và 1.589 rows; kiểm tra sai khác Gemini/Expert/Novice, không sửa dữ liệu để khớp paper.  
2. Audit judge bằng mẫu phân tầng theo tutor/source/dimension; ưu tiên Tone, Humanlikeness và Mistake Identification; gán mù độc lập rồi adjudicate.  
3. Xây dataset mới và guideline trước khi chạy model; pilot 20–30 mẫu; xác định cách tính 250 mẫu mỗi người và phát hiện mẫu trùng.  
4. Chốt group split; dùng validation chọn tham số và khóa test; chạy các baseline và Qwen trên đúng cùng test examples.  
5. Hoàn tất slide/báo cáo với kết quả theo từng dimension, phân tích lỗi định tính và giới hạn; tách nhãn người khỏi nhãn LLM judge.

## 8. Giới hạn

- DAMR của response mới chỉ do một Qwen judge tạo. `parse_ok=True` xác nhận schema parse được, không xác nhận nhãn đúng; không gọi đây là human DAMR.  
- Release chênh bảy mẫu với paper, đặc biệt Novice 53 thay vì 60; không hoàn toàn cùng tập mẫu.  
- Llama/Phi/Mistral chạy với Hugging Face checkpoint theo prompt Figure 2; paper không công bố đủ revisions/template/decoding để tái lập bit-for-bit.  
- Qwen AC dùng ordinal encoding 1–3 do notebook chọn; mapping số của paper không được nêu tường minh.  
- Baseline chỉ có một split seed 42, test 38 conversation IDs và nhiều lớp hiếm; chưa có cross-validation/kiểm định ý nghĩa.  
- Zero-shot–CoT hiện chỉ là mô tả; nên bootstrap theo conversation ID nếu cần lượng hóa bất định.

## 9. Phụ lục A — DAMR từng tutor × dimension

`Release` là DAMR tái tính từ nhãn người; `Qwen judge` là dự đoán Qwen trên cùng response; deltas là điểm phần trăm. Novice có N=53 hiện tại, paper Table 3 dùng 60.

<details><summary>Mở bảng 72 ô DAMR</summary>

{markdown_table(["Tutor", "Dimension", "N hiện tại", "Paper human %", "Release human %", "Qwen judge %", "Release−paper pp", "Qwen−paper pp", "Qwen−human pp"], release_rows)}

</details>

## 10. Phụ lục B — Pearson AC từng tutor × dimension

N là số cặp nhãn trên release hiện tại. Paper critic values lấy từ Tables 5–6; Qwen AC tính trên human labels hiện hành.

<details><summary>Mở bảng 72 ô AC</summary>

{markdown_table(["Tutor", "Dimension", "N paired", "Qwen3.8 r", "Prometheus2 paper r", "Llama critic paper r"], ac_rows)}

</details>

## Nguồn và provenance

- [Paper gốc: Maurya et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.57/), Sections 5.1–5.4, Table 3, Tables 5–6 và Appendix Figure 6. Paper mô tả 60 Bridge + 132 MathDial, 1.596 responses, bốn annotator; DAMR là tỷ lệ nhãn mong muốn, AC là Pearson giữa critic và human labels.
- [Repo upstream UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation).
- ZIP moLab được dùng khi sinh báo cáo; không đưa ZIP có raw responses/judgements vào Git để repo nhẹ và tránh lưu output lớn.
- Baseline details và audit: `reports/baseline/`; mã: `src/baseline.py`.
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Path to mrbench_all_outputs.zip")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "benchmark_results_project_plan.md"
    )
    args = parser.parse_args()
    text = build_report(args.bundle.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
