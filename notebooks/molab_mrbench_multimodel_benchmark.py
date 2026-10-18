import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    intro = mo.md(r"""
    # MRBench: tutor baselines + Qwen3.8-27B evaluation

    Chạy trên cùng các hội thoại MRBench V1: **Llama 3.1 8B, Phi-3 Mini và Mistral 7B** dùng prompt tutor ở Figure 2 của paper; **Qwen3.5-9B** chạy hai điều kiện Zero-shot và CoT để so sánh.

    Ba model đầu nằm trong nhóm model tutor của paper; Qwen3.5 Zero-shot/CoT là phần mở rộng đã sinh response. Notebook này dùng **Qwen3.8-27B chạy local trên GPU** làm LLM judge để gán đủ tám nhãn cho response mới và tính Qwen-judged DAMR. Đồng thời, Qwen chấm lại response trong MRBench release để tính Pearson AC với nhãn người đã phát hành.

    **Lưu ý diễn giải:** DAMR của response mới do Qwen chấm là *Qwen-judged DAMR*, không phải human DAMR như Table 3. AC chỉ tính hợp lệ trên các response gốc có nhãn người tương ứng; response mới chưa có nhãn người nên không thể tính paper-style AC cho chúng nếu chưa có human gold. Paper không nêu đầy đủ cách mã hóa nhãn categorical dùng cho Pearson; notebook ghi rõ quy ước mã hóa thứ bậc và xem AC so với Table 5–6 là tái lập gần đúng.

    ## Chạy trên moLab

    1. Mở [moLab](https://molab.marimo.io/notebooks) và tải/mở file marimo `.py` này.
    2. Trong **notebook specs**, bật NVIDIA RTX PRO 6000 Blackwell. Notebook kiểm tra GPU và cảnh báo nếu phát hiện card khác.
    3. Model tải từ Hugging Face; cần Internet. Notebook tự cài/cập nhật `transformers` và `accelerate` trong runtime nếu cần. Llama 3.1 8B là model gated: chấp nhận license trên Hugging Face. Nếu chạy Llama, điền token trong `HF_TOKEN = ""`.
    4. Mặc định chạy đủ 192 hội thoại: 192 response cho mỗi model paper-prompt và 384 response cho Qwen3.5 (tổng 960 lượt sinh). Sửa `MAX_DIALOGUES = 12` để chạy pilot.
    5. Notebook dùng BF16, greedy decoding và checkpoint revision cố định. Phi-3 Mini checkpoint được chọn là bản 4K-Instruct; notebook giới hạn input để chừa chỗ cho response trong cửa sổ ngữ cảnh 4K.
    6. Có thể tải toàn bộ thư mục kết quả thành một ZIP. Khi mở phiên moLab mới, tải ZIP đó lên ở mục **Khôi phục checkpoint từ ZIP** trước khi bấm Run/resume; notebook khôi phục CSV checkpoint rồi bỏ qua response đã hoàn tất.

    Ba model paper-prompt chỉ nhận template Figure 2, không thêm CoT. Với Mistral, system prompt được ghép vào nội dung instruction vì định dạng Instruct của checkpoint dùng một khối `[INST]`. Qwen3.5 Zero-shot tắt thinking; CoT thêm chỉ dẫn suy luận và bật thinking. Qwen3.8-27B judge chạy chế độ instruct (thinking off), xuất nhãn JSON, lưu từng response vào checkpoint CSV và có thể tiếp tục sau khi runtime ngắt.
    """)
    intro
    return (mo,)


@app.cell
def _(mo):
    import hashlib
    import json
    import os
    import re
    import subprocess as package_installer
    import sys as runtime_sys
    import time
    from pathlib import Path
    import importlib.metadata as package_metadata

    import pandas as pd
    import requests
    import torch

    try:
        installed_transformers = package_metadata.version("transformers")
        transformers_version = tuple(
            int(part) for part in installed_transformers.split(".")[:2]
        )
    except package_metadata.PackageNotFoundError:
        transformers_version = (0, 0)

    install_packages = transformers_version < (5, 0)
    try:
        package_metadata.version("accelerate")
    except package_metadata.PackageNotFoundError:
        install_packages = True

    if install_packages:
        print("Đang tự cài/cập nhật Transformers 5+ và accelerate trong notebook...")
        try:
            package_installer.check_call(
                [
                    runtime_sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "transformers",
                    "accelerate",
                ]
            )
        except Exception as install_exception:
            raise RuntimeError(
                "Notebook không cài được Transformers/accelerate. Kiểm tra Internet "
                "của runtime rồi chạy lại cell. Không cần cài package thủ công."
            ) from install_exception

    import accelerate
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Chưa bật GPU trong moLab notebook specs.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    is_blackwell_6000 = "6000" in gpu_name and "blackwell" in gpu_name.casefold()
    if not is_blackwell_6000:
        mo.md(
            f"⚠️ Phát hiện **{gpu_name}** ({gpu_vram_gb:.1f} GiB VRAM), "
            "không khớp tên RTX PRO 6000 Blackwell. Tiếp tục nếu GPU thay thế "
            "có đủ VRAM và hỗ trợ BF16."
        )
    else:
        mo.md(f"GPU sẵn sàng: **{gpu_name}**, {gpu_vram_gb:.1f} GiB VRAM.")

    DATA_URL = (
        "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/"
        "main/MRBench/MRBench_V1.json"
    )
    EXPECTED_DATA_SHA256 = (
        "77a33ab81ca9c773d22faee2be7a7bb43b203a06497a5daa35d715f9ada3a6f3"
    )
    HF_TOKEN = ""  # Dán token vào giữa dấu nháy, ví dụ: "hf_..."
    OUT_DIR = Path.cwd() / "mrbench_multimodel_outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH = OUT_DIR / "MRBench_V1.json"
    RELEASE_DAMR_FILE = OUT_DIR / "paper_vs_current_release_damr.csv"

    return (
        AutoModelForMultimodalLM,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        accelerate,
        package_installer,
        runtime_sys,
        DATA_PATH,
        DATA_URL,
        EXPECTED_DATA_SHA256,
        HF_TOKEN,
        OUT_DIR,
        RELEASE_DAMR_FILE,
        hashlib,
        json,
        pd,
        re,
        requests,
        time,
        torch,
    )


@app.cell
def _(mo):
    resume_archive_upload = mo.ui.file(
        filetypes=[".zip"],
        multiple=False,
        kind="area",
        max_size=500_000_000,
        label="Tải ZIP checkpoint để tiếp tục phiên trước",
    )
    mo.md(
        "## Khôi phục checkpoint từ ZIP\n"
        "Ở phiên moLab mới, tải ZIP kết quả đã lưu ở đây. File generation và Qwen judge "
        "được phục hồi vào `mrbench_multimodel_outputs`; sau đó bấm Run/resume như bình thường. "
        "Nếu chưa có ZIP, để trống và notebook sẽ bắt đầu từ đầu."
    )
    resume_archive_upload
    return (resume_archive_upload,)


@app.cell
def _(OUT_DIR, hashlib, json, resume_archive_upload, mo):
    import io as resume_zip_io
    import zipfile as resume_zipfile

    resume_archive_revision = ""
    if resume_archive_upload.value:
        resume_archive = resume_archive_upload.value[0]
        resume_archive_bytes = resume_archive.contents
        resume_archive_revision = hashlib.sha256(resume_archive_bytes).hexdigest()
        try:
            with resume_zipfile.ZipFile(resume_zip_io.BytesIO(resume_archive_bytes)) as archive:
                if "MANIFEST.json" not in archive.namelist():
                    raise ValueError("ZIP thiếu MANIFEST.json; hãy dùng ZIP xuất từ notebook này.")
                manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
                if manifest.get("format") != "mrbench-output-bundle-v1":
                    raise ValueError("Phiên bản ZIP không được hỗ trợ; hãy xuất lại ZIP từ notebook mới nhất.")

                manifest_entries = manifest.get("files", [])
                if not isinstance(manifest_entries, list):
                    raise ValueError("Danh sách files trong manifest không hợp lệ.")
                restored_payloads = []
                seen_names = set()
                total_uncompressed_bytes = 0
                for entry in manifest_entries:
                    filename = str(entry.get("name", ""))
                    if (
                        not filename
                        or filename in {".", "..", "MANIFEST.json"}
                        or "\\" in filename
                        or any(part in {"", ".", ".."} for part in filename.split("/"))
                        or ":" in filename
                        or "\x00" in filename
                        or filename.casefold().endswith(".zip")
                    ):
                        raise ValueError(f"Tên file không hợp lệ trong manifest: {filename!r}")
                    if filename in seen_names:
                        raise ValueError(f"Tên file bị lặp trong manifest: {filename}")
                    seen_names.add(filename)
                    if filename not in archive.namelist():
                        raise ValueError(f"ZIP thiếu file được khai báo trong manifest: {filename}")
                    zip_info = archive.getinfo(filename)
                    total_uncompressed_bytes += zip_info.file_size
                    if total_uncompressed_bytes > 500_000_000:
                        raise ValueError("Tổng dung lượng giải nén vượt giới hạn 500 MB.")
                    payload = archive.read(filename)
                    if len(payload) != int(entry.get("size_bytes", -1)):
                        raise ValueError(f"Kích thước file không khớp: {filename}")
                    checksum = hashlib.sha256(payload).hexdigest()
                    if checksum != entry.get("sha256"):
                        raise ValueError(f"Checksum không khớp: {filename}")
                    restored_payloads.append((filename, payload))

            restored_files = []
            for filename, payload in restored_payloads:
                target_path = OUT_DIR.joinpath(*filename.split("/"))
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(payload)
                restored_files.append(filename)

            resume_archive_status = mo.md(
                f"✅ Đã khôi phục **{len(restored_files)} file** từ `{resume_archive.name}`. "
                "Các response đã có checkpoint sẽ được bỏ qua khi chạy tiếp."
            )
        except Exception as resume_error:
            resume_archive_revision = ""
            resume_archive_status = mo.md(
                f"❌ Không khôi phục được ZIP: `{type(resume_error).__name__}: {resume_error}`"
            )
    else:
        resume_archive_status = mo.md("Chưa tải ZIP checkpoint; có thể bắt đầu benchmark từ đầu.")

    resume_archive_status
    return (resume_archive_revision,)


@app.cell
def _(
    DATA_PATH,
    DATA_URL,
    EXPECTED_DATA_SHA256,
    hashlib,
    json,
    mo,
    pd,
    requests,
    resume_archive_revision,
):
    _ = resume_archive_revision  # Bảo đảm ZIP được khôi phục trước khi nạp dataset.
    if not DATA_PATH.exists():
        response = requests.get(DATA_URL, timeout=120)
        response.raise_for_status()
        DATA_PATH.write_bytes(response.content)

    data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if data_sha256.lower() != EXPECTED_DATA_SHA256:
        raise RuntimeError(
            f"MRBench V1 SHA-256 không khớp: {data_sha256}; "
            f"đang chờ {EXPECTED_DATA_SHA256}. Kiểm tra lại snapshot dữ liệu."
        )

    dialogues = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(dialogues, list):
        raise ValueError("MRBench_V1.json phải là một JSON list.")

    source_counts = pd.Series([str(d.get("Data", "Unknown")) for d in dialogues]).value_counts()
    mo.md(
        f"Đã nạp **{len(dialogues)}** hội thoại; phân bố nguồn: "
        f"{source_counts.to_dict()}. SHA-256: `{data_sha256}`. "
        "Paper báo cáo 60 Bridge + 132 MathDial; release snapshot này có thể "
        "khác phân bố đó, nên notebook giữ nguyên dữ liệu và ghi nhận khác biệt."
    )
    return data_sha256, dialogues, source_counts


@app.cell
def _(RELEASE_DAMR_FILE, dialogues, json, mo, pd, re):
    DIMENSIONS = [
        "Mistake_Identification",
        "Mistake_Location",
        "Revealing_of_the_Answer",
        "Providing_Guidance",
        "Actionability",
        "Coherence",
        "Tutor_Tone",
        "Humanlikeness",
    ]
    DESIRED = {
        "Mistake_Identification": "Yes",
        "Mistake_Location": "Yes",
        "Revealing_of_the_Answer": "No",
        "Providing_Guidance": "Yes",
        "Actionability": "Yes",
        "Coherence": "Yes",
        "Tutor_Tone": "Encouraging",
        "Humanlikeness": "Yes",
    }
    PAPER_TABLE3 = {
        "Novice": [43.33, 16.67, 80.00, 11.67, 1.67, 50.00, 90.00, 35.00],
        "Expert": [76.04, 63.02, 90.62, 67.19, 76.04, 79.17, 92.19, 87.50],
        "Llama-3.1-8B": [80.21, 54.69, 73.96, 45.31, 42.71, 80.73, 19.79, 93.75],
        "Phi3": [28.65, 26.04, 73.96, 17.71, 11.98, 39.58, 45.31, 52.08],
        "Gemini": [63.02, 39.58, 67.71, 37.50, 42.71, 56.77, 21.88, 68.23],
        "Sonnet": [85.42, 69.79, 94.79, 59.38, 60.94, 88.54, 54.69, 96.35],
        "Mistral": [93.23, 73.44, 86.46, 63.54, 70.31, 86.98, 15.10, 95.31],
        "GPT-4": [94.27, 84.38, 53.12, 76.04, 46.35, 90.17, 37.50, 89.62],
        "Llama-3.1-405B": [94.27, 84.38, 80.73, 77.08, 74.48, 91.67, 16.15, 90.62],
    }
    TUTOR_KEYS = {
        "Gemini": "Gemini",
        "Phi3": "Phi3",
        "Llama318B": "Llama-3.1-8B",
        "Llama31405B": "Llama-3.1-405B",
        "Mistral": "Mistral",
        "Expert": "Expert",
        "GPT4": "GPT-4",
        "Sonnet": "Sonnet",
        "Novice": "Novice",
    }

    def canonical_dimension(raw_key):
        key = re.sub(r"[^a-z]", "", str(raw_key).casefold())
        return {
            "mistakeidentification": "Mistake_Identification",
            "mistakelocation": "Mistake_Location",
            "revealingoftheanswer": "Revealing_of_the_Answer",
            "providingguidance": "Providing_Guidance",
            "actionability": "Actionability",
            "coherence": "Coherence",
            "tutortone": "Tutor_Tone",
            "humanlikeness": "Humanlikeness",
            "humanness": "Humanlikeness",
        }.get(key)

    def normalize_label(dimension, value):
        if value is None or pd.isna(value):
            return None
        text = re.sub(r"\s+", " ", str(value).strip().casefold())
        if dimension == "Revealing_of_the_Answer":
            if text in {"no", "no answer", "does not reveal"}:
                return "No"
            if text.startswith("yes"):
                if "incorrect" in text or "wrong" in text:
                    return "Yes (but the answer is incorrect)"
                if "correct" in text or text == "yes":
                    return "Yes (and the answer is correct)"
            return None
        if dimension == "Tutor_Tone":
            return {
                "encouraging": "Encouraging",
                "neutral": "Neutral",
                "offensive": "Offensive",
            }.get(text)
        return {"yes": "Yes", "to some extent": "To some extent", "no": "No"}.get(text)

    release_rows = []
    for dialogue_idx, dialogue in enumerate(dialogues):
        for raw_tutor, obj in (dialogue.get("anno_llm_responses") or {}).items():
            normalized = {}
            for raw_dim, raw_value in (obj.get("annotation") or {}).items():
                dim = canonical_dimension(raw_dim)
                if dim:
                    normalized[dim] = normalize_label(dim, raw_value)
            release_rows.append(
                {
                    "dialogue_index": dialogue_idx,
                    "sample_id": f"release:{dialogue_idx}:{TUTOR_KEYS.get(raw_tutor, raw_tutor)}",
                    "conversation_id": str(dialogue.get("conversation_id", "")),
                    "source": str(dialogue.get("Data", "Unknown")),
                    "topic": str(dialogue.get("Topic", "")),
                    "history": str(dialogue.get("conversation_history", "")),
                    "tutor": TUTOR_KEYS.get(raw_tutor, raw_tutor),
                    "response": str(obj.get("response", "")),
                    **{dim: normalized.get(dim) for dim in DIMENSIONS},
                }
            )
    release_df = pd.DataFrame(release_rows)

    release_scores = []
    for tutor, group in release_df.groupby("tutor", sort=False):
        for dim in DIMENSIONS:
            labels = group[dim].dropna()
            release_scores.append(
                {
                    "tutor": tutor,
                    "dimension": dim,
                    "release_DAMR_pct": (
                        100 * (labels == DESIRED[dim]).mean() if len(labels) else float("nan")
                    ),
                    "n_release": int(len(labels)),
                }
            )
    release_damr = pd.DataFrame(release_scores)
    paper_damr = pd.DataFrame(
        [
            {"tutor": tutor, "dimension": dim, "paper_DAMR_pct": score}
            for tutor, scores in PAPER_TABLE3.items()
            for dim, score in zip(DIMENSIONS, scores)
        ]
    )
    paper_vs_release = paper_damr.merge(release_damr, on=["tutor", "dimension"], how="left")
    paper_vs_release["release_minus_paper_pp"] = (
        paper_vs_release["release_DAMR_pct"] - paper_vs_release["paper_DAMR_pct"]
    )
    paper_vs_release.to_csv(RELEASE_DAMR_FILE, index=False)
    mo.md("## Frozen MRBench V1 baseline\nDAMR được tính lại từ response và annotation đã có trong release. Đây là mốc đối chiếu dữ liệu gốc; không phải điểm của Qwen.")
    paper_vs_release
    return (
        DESIRED,
        DIMENSIONS,
        PAPER_TABLE3,
        TUTOR_KEYS,
        canonical_dimension,
        normalize_label,
        paper_vs_release,
        release_df,
    )


@app.cell
def _(OUT_DIR, mo):
    MAX_DIALOGUES = None  # None = đủ 192; đặt 12 để chạy pilot.
    SEED = 42
    MAX_INPUT_TOKENS_DEFAULT = 8192
    MAX_INPUT_TOKENS_PHI3 = 3500  # Chừa tối thiểu 384 tokens cho context window 4K.
    MAX_NEW_TOKENS_PAPER = 384
    MAX_NEW_TOKENS_ZERO_SHOT = 384
    MAX_NEW_TOKENS_COT = 1536

    MODEL_SPECS = {
        "llama31_8b": {
            "display_name": "Llama 3.1 8B",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
            "family": "causal",
            "methods": ("paper_prompt",),
            "trust_remote_code": False,
            "max_input_tokens": MAX_INPUT_TOKENS_DEFAULT,
        },
        "phi3_mini": {
            "display_name": "Phi-3 Mini 4K",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "revision": "f39ac1d28e925b323eae81227eaba4464caced4e",
            "family": "causal",
            "methods": ("paper_prompt",),
            "trust_remote_code": False,
            "max_input_tokens": MAX_INPUT_TOKENS_PHI3,
        },
        "mistral_7b": {
            "display_name": "Mistral 7B Instruct v0.2",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "revision": "b9cbf4f579dff85096d3528ab81b46ceacf99bdd",
            "family": "causal",
            "methods": ("paper_prompt",),
            "trust_remote_code": False,
            "max_input_tokens": MAX_INPUT_TOKENS_DEFAULT,
        },
        "qwen35_9b": {
            "display_name": "Qwen3.5 9B",
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "cc5442c03a5c0bff0bd4c6888d9a40029c637733",
            "family": "qwen_multimodal",
            "methods": ("zero_shot", "cot"),
            "trust_remote_code": False,
            "max_input_tokens": MAX_INPUT_TOKENS_DEFAULT,
        },
    }
    MODEL_ALIASES_TO_RUN = ["llama31_8b", "phi3_mini", "mistral_7b", "qwen35_9b"]
    METHOD_NAMES = {
        "paper_prompt": "Paper Figure 2 prompt",
        "zero_shot": "Zero-shot (thinking off)",
        "cot": "CoT (explicit reasoning + thinking on)",
    }
    GENERATION_FILE = OUT_DIR / "multimodel_fig2_responses.csv"
    ANNOTATION_TEMPLATE_FILE = OUT_DIR / "human_annotation_template.csv"
    ANNOTATED_CSV = None  # Sau khi gán nhãn, ví dụ: "mrbench_multimodel_outputs/human_annotated_responses.csv"
    mo.md(
        "## Cấu hình thí nghiệm\n"
        "Mặc định chạy đủ ba model dùng Figure 2 và Qwen3.5 ở hai điều kiện. "
        "Sửa `MODEL_ALIASES_TO_RUN` để chọn model, hoặc đặt `MAX_DIALOGUES = 12` cho pilot. "
        "Các model paper chỉ chạy `paper_prompt`; Qwen mới chạy Zero-shot/CoT."
    )
    return (
        ANNOTATED_CSV,
        ANNOTATION_TEMPLATE_FILE,
        GENERATION_FILE,
        MAX_DIALOGUES,
        MAX_NEW_TOKENS_COT,
        MAX_NEW_TOKENS_PAPER,
        MAX_NEW_TOKENS_ZERO_SHOT,
        METHOD_NAMES,
        MODEL_ALIASES_TO_RUN,
        MODEL_SPECS,
        SEED,
    )


@app.cell
def _(
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    DATA_PATH,
    EXPECTED_DATA_SHA256,
    GENERATION_FILE,
    HF_TOKEN,
    MAX_DIALOGUES,
    MAX_NEW_TOKENS_COT,
    MAX_NEW_TOKENS_PAPER,
    MAX_NEW_TOKENS_ZERO_SHOT,
    MODEL_ALIASES_TO_RUN,
    MODEL_SPECS,
    resume_archive_revision,
    SEED,
    data_sha256,
    dialogues,
    mo,
    pd,
    re,
    run_button,
    time,
    torch,
):
    _ = resume_archive_revision  # Đọc lại generation CSV sau khi nạp ZIP checkpoint.
    if not run_button.value:
        generations = (
            pd.read_csv(GENERATION_FILE)
            if GENERATION_FILE.exists()
            else pd.DataFrame()
        )
        mo.md(
            "Bấm Run multimodel benchmark để chạy hoặc tiếp tục. "
            "Notebook bỏ qua các cặp dialogue/model/method đã ghi vào CSV."
        )
    else:
        def benchmark_frame(items, max_dialogues, seed):
            frame = pd.DataFrame(
                [
                    {
                        "dialogue_index": idx,
                        "conversation_id": str(item.get("conversation_id", "")),
                        "source": str(item.get("Data", "Unknown")),
                        "topic": str(item.get("Topic", "")),
                        "history": str(item.get("conversation_history", "")),
                    }
                    for idx, item in enumerate(items)
                ]
            )
            if max_dialogues is None or max_dialogues >= len(frame):
                return frame
            target = max(1, int(max_dialogues))
            groups = list(frame.groupby("source", sort=False))
            sampled = []
            for _, group in groups:
                take = min(len(group), max(1, target * len(group) // len(frame)))
                sampled.append(group.sample(n=take, random_state=seed))
            result = pd.concat(sampled).drop_duplicates("dialogue_index")
            if len(result) < target:
                remaining = frame.loc[~frame.dialogue_index.isin(result.dialogue_index)]
                needed = min(target - len(result), len(remaining))
                result = pd.concat([result, remaining.sample(n=needed, random_state=seed)])
            return result.sort_values("dialogue_index").head(target).reset_index(drop=True)

        def build_messages(raw_dialogue, method):
            source = str(raw_dialogue.get("Data", ""))
            history = str(raw_dialogue.get("conversation_history", "")).strip()
            if source == "Bridge":
                system = (
                    "You are an experienced elementary school math teacher, and you are going "
                    "to respond to a student's mistake in a useful and caring way"
                )
                topic = str(raw_dialogue.get("Topic", "")).strip()
                user = (
                    f"The problem your student is solving is on the topic: {topic}\n"
                    f" Conversation History: {history}"
                )
                assistant_prefix = (
                    "Tutor response (maximum one sentence that is most appropriate given "
                    "topic and conversation history):"
                )
            else:
                system = (
                    "You are an experienced middle school math teacher, and you are going "
                    "to respond to a student's mistake in a useful and caring way"
                )
                user = (
                    "The conversation history of the problem your student is solving is: "
                    f"{history}"
                )
                assistant_prefix = (
                    "Tutor response (maximum one sentence that aligns most appropriately "
                    "with conversation history):"
                )
            if method == "cot":
                system += (
                    " Before writing the tutor response, reason through the student's work "
                    "step by step and decide on the most useful next pedagogical move. "
                    "Keep that reasoning private; return only the requested tutor response."
                )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return messages, assistant_prefix

        def expected_method_settings(alias, method):
            spec = MODEL_SPECS[alias]
            if method == "cot":
                max_new_tokens = MAX_NEW_TOKENS_COT
                prompt_version = "mrbench_fig2_plus_cot_v1"
                enable_thinking = True
            elif method == "zero_shot":
                max_new_tokens = MAX_NEW_TOKENS_ZERO_SHOT
                prompt_version = "mrbench_fig2_zero_shot_v1"
                enable_thinking = False
            else:
                max_new_tokens = MAX_NEW_TOKENS_PAPER
                prompt_version = "mrbench_fig2_paper_prompt_v1"
                enable_thinking = False
            return {
                "model_id": spec["model_id"],
                "model_revision": spec["revision"],
                "data_sha256": EXPECTED_DATA_SHA256,
                "dtype": "bfloat16",
                "do_sample": False,
                "prompt_version": prompt_version,
                "enable_thinking": enable_thinking,
                "max_input_tokens": spec["max_input_tokens"],
                "max_new_tokens": max_new_tokens,
            }

        if not MODEL_ALIASES_TO_RUN:
            raise ValueError("MODEL_ALIASES_TO_RUN đang rỗng; hãy chọn ít nhất một model.")
        unknown_aliases = set(MODEL_ALIASES_TO_RUN) - set(MODEL_SPECS)
        if unknown_aliases:
            raise ValueError(f"Model alias không tồn tại: {sorted(unknown_aliases)}")
        if len(set(MODEL_ALIASES_TO_RUN)) != len(MODEL_ALIASES_TO_RUN):
            raise ValueError("MODEL_ALIASES_TO_RUN có alias bị lặp.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "GPU đang dùng không báo hỗ trợ BF16; cần RTX PRO 6000 Blackwell "
                "hoặc GPU tương đương."
            )

        old = pd.read_csv(GENERATION_FILE) if GENERATION_FILE.exists() else pd.DataFrame()
        done = set()
        if not old.empty:
            required_columns = {
                "dialogue_index",
                "model_alias",
                "method",
                "model_id",
                "model_revision",
                "data_sha256",
                "dtype",
                "do_sample",
                "prompt_version",
                "enable_thinking",
                "max_input_tokens",
                "max_new_tokens",
            }
            missing = required_columns - set(old.columns)
            if missing:
                raise RuntimeError(
                    f"Output CSV thiếu cột {sorted(missing)}; đổi tên file kết quả "
                    "trước khi bắt đầu benchmark."
                )
            keys = list(
                zip(
                    old["dialogue_index"].astype(int),
                    old["model_alias"].astype(str),
                    old["method"].astype(str),
                )
            )
            if len(keys) != len(set(keys)):
                raise RuntimeError("Output CSV có khóa dialogue/model/method bị trùng.")
            for record in old.to_dict("records"):
                alias = str(record["model_alias"])
                method = str(record["method"])
                if alias not in MODEL_SPECS or method not in MODEL_SPECS[alias]["methods"]:
                    raise RuntimeError(
                        f"Output CSV chứa cấu hình ngoài notebook hiện tại: {alias}/{method}."
                    )
                expected = expected_method_settings(alias, method)
                for field, value in expected.items():
                    actual = record[field]
                    if field in {"do_sample", "enable_thinking"}:
                        actual_value = str(actual).casefold() in {"true", "1"}
                        expected_value = bool(value)
                    elif field in {"max_input_tokens", "max_new_tokens"}:
                        actual_value = int(actual)
                        expected_value = int(value)
                    else:
                        actual_value = str(actual)
                        expected_value = str(value)
                    if actual_value != expected_value:
                        raise RuntimeError(
                            f"Output CSV có {field} khác cấu hình cho {alias}/{method}. "
                            "Đổi tên file output để tránh trộn kết quả."
                        )
            done = set(keys)

        sample = benchmark_frame(dialogues, MAX_DIALOGUES, SEED)
        runs_completed = 0

        def load_checkpoint(alias):
            spec = MODEL_SPECS[alias]
            auth_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
            common = {
                "revision": spec["revision"],
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                **auth_kwargs,
            }
            if spec["family"] == "qwen_multimodal":
                decoder = AutoProcessor.from_pretrained(
                    spec["model_id"], revision=spec["revision"], **auth_kwargs
                )
                model = AutoModelForMultimodalLM.from_pretrained(
                    spec["model_id"], **common
                )
            else:
                remote_kwargs = (
                    {"trust_remote_code": True}
                    if spec["trust_remote_code"]
                    else {}
                )
                tokenizer_kwargs = {
                    "revision": spec["revision"],
                    **auth_kwargs,
                    **remote_kwargs,
                }
                decoder = AutoTokenizer.from_pretrained(
                    spec["model_id"], **tokenizer_kwargs
                )
                model = AutoModelForCausalLM.from_pretrained(
                    spec["model_id"], **{**common, **remote_kwargs}
                )
            model.eval()
            resolved_revision = getattr(model.config, "_commit_hash", None)
            resolved_revision = str(resolved_revision or spec["revision"])
            if resolved_revision != spec["revision"]:
                del model, decoder
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"{alias}: checkpoint resolve thành {resolved_revision}, "
                    f"trong khi notebook yêu cầu {spec['revision']}."
                )
            return model, decoder, resolved_revision

        def encode_prompt(alias, decoder, messages, assistant_prefix, method):
            spec = MODEL_SPECS[alias]
            if spec["family"] == "qwen_multimodal":
                encoded = decoder.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=(method == "cot"),
                )
                tokenizer = decoder.tokenizer
            else:
                chat_messages = messages
                if alias == "mistral_7b":
                    chat_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"{messages[0]['content']}\n\n{messages[1]['content']}\n"
                                f"{assistant_prefix}"
                            ),
                        }
                    ]
                encoded = decoder.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                tokenizer = decoder
            if alias != "mistral_7b":
                prefix_ids = tokenizer(
                    assistant_prefix,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"]
                encoded["input_ids"] = torch.cat(
                    [encoded["input_ids"], prefix_ids], dim=-1
                )
                encoded["attention_mask"] = torch.cat(
                    [
                        encoded["attention_mask"],
                        torch.ones_like(prefix_ids),
                    ],
                    dim=-1,
                )
            return encoded.to("cuda:0")

        def decode_response(alias, decoder, generated_ids):
            is_qwen = MODEL_SPECS[alias]["family"] == "qwen_multimodal"
            text = decoder.decode(
                generated_ids,
                skip_special_tokens=not is_qwen,
            ).strip()
            if "</think>" in text:
                text = text.rsplit("</think>", 1)[-1].strip()
            for stop in ("<|im_end|>", "<|endoftext|>", "<|fim_suffix|>"):
                if stop in text:
                    text = text.split(stop, 1)[0].strip()
            return re.sub(r"<\|[^>]+\|>", "", text).strip()

        for alias in MODEL_ALIASES_TO_RUN:
            spec = MODEL_SPECS[alias]
            alias_sample = [
                row
                for row in sample.itertuples(index=False)
                if any(
                    (int(row.dialogue_index), alias, method) not in done
                    for method in spec["methods"]
                )
            ]
            if not alias_sample:
                print(f"Bỏ qua {alias}: mọi response trong phạm vi chạy đã có.")
                continue

            if alias == "llama31_8b" and not HF_TOKEN:
                raise RuntimeError(
                    "Llama 3.1 8B là model gated. Hãy chấp nhận license trên Hugging Face "
                    "và điền HF_TOKEN = \"hf_...\" trong cell cấu hình của notebook."
                )
            try:
                model, decoder, model_revision = load_checkpoint(alias)
            except Exception as load_exception:
                if "accelerate" in str(load_exception).casefold():
                    raise RuntimeError(
                        "Notebook đã thử tự cài `accelerate` nhưng model loader vẫn "
                        "không dùng được package này. Kiểm tra kết nối Internet, "
                        "restart runtime rồi chạy lại notebook."
                    ) from load_exception
                if alias == "llama31_8b":
                    raise RuntimeError(
                        "Không tải được Llama 3.1 8B. Kiểm tra HF_TOKEN, quyền truy cập "
                        "model gated và license Hugging Face."
                    ) from load_exception
                raise

            if alias == "phi3_mini" and not callable(getattr(model, "generate", None)):
                raise RuntimeError(
                    "Phi-3 đang được nạp bằng custom code cũ, không hỗ trợ `.generate()` "
                    "trên phiên bản Transformers hiện tại. Dùng notebook mới nhất với "
                    "trust_remote_code=False cho Phi-3."
                )

            try:
                for row in alias_sample:
                    raw_dialogue = dialogues[int(row.dialogue_index)]
                    for method in spec["methods"]:
                        key = (int(row.dialogue_index), alias, method)
                        if key in done:
                            continue
                        messages, assistant_prefix = build_messages(raw_dialogue, method)
                        encoded = encode_prompt(
                            alias, decoder, messages, assistant_prefix, method
                        )
                        input_length = int(encoded["input_ids"].shape[-1])
                        max_input_tokens = int(spec["max_input_tokens"])
                        if input_length > max_input_tokens:
                            raise ValueError(
                                f"{alias} dialogue {row.dialogue_index} có {input_length} "
                                f"input tokens, vượt giới hạn {max_input_tokens}. "
                                "Notebook không tự cắt lịch sử hội thoại."
                            )
                        settings = expected_method_settings(alias, method)
                        started = time.time()
                        with torch.inference_mode():
                            output_ids = model.generate(
                                **encoded,
                                max_new_tokens=settings["max_new_tokens"],
                                do_sample=False,
                                num_beams=1,
                                use_cache=True,
                            )
                        generated_ids = output_ids[0, input_length:]
                        answer = decode_response(alias, decoder, generated_ids)
                        record = {
                            "dialogue_index": int(row.dialogue_index),
                            "conversation_id": str(row.conversation_id),
                            "source": str(row.source),
                            "topic": str(row.topic),
                            "history": str(row.history),
                            "model_alias": alias,
                            "model_name": spec["display_name"],
                            "method": method,
                            **settings,
                            "generated_tokens_including_reasoning": int(
                                generated_ids.shape[-1]
                            ),
                            "generation_seconds": round(time.time() - started, 3),
                            "response": answer,
                        }
                        pd.DataFrame([record]).to_csv(
                            GENERATION_FILE,
                            mode="a",
                            header=not GENERATION_FILE.exists(),
                            index=False,
                            encoding="utf-8",
                        )
                        done.add(key)
                        runs_completed += 1
                        if runs_completed % 10 == 0:
                            print(
                                f"Đã sinh mới {runs_completed} response; "
                                f"gần nhất: {alias}/{method}, dialogue {row.dialogue_index}."
                            )
            finally:
                del model, decoder
                torch.cuda.empty_cache()

        generations = (
            pd.read_csv(GENERATION_FILE)
            if GENERATION_FILE.exists()
            else pd.DataFrame()
        )
        mo.md(
            f"Hoàn tất/tiếp tục benchmark. File kết quả: {GENERATION_FILE}. "
            f"Lượt sinh mới ở lần này: {runs_completed}."
        )
    return (generations,)


@app.cell
def _(MAX_DIALOGUES, mo):
    run_button = mo.ui.run_button(label="Run multimodel benchmark")
    mo.md(
        f"**Cỡ chạy hiện tại:** `MAX_DIALOGUES = {MAX_DIALOGUES}` "
        "(None nghĩa là toàn bộ dữ liệu). Mặc định: 3 model theo prompt Figure 2 + "
        "Qwen3.5 Zero-shot/CoT.\n\n"
    )
    run_button
    return (run_button,)


@app.cell
def _(
    ANNOTATION_TEMPLATE_FILE,
    DIMENSIONS,
    GENERATION_FILE,
    METHOD_NAMES,
    generations,
    mo,
    normalize_label,
    pd,
):
    if generations.empty:
        mo.md("Chưa có response. Dùng nút Run multimodel benchmark ở phía trên.")
        display_df = pd.DataFrame()
        annotation_template = pd.DataFrame()
    else:
        display_df = generations.groupby(
            ["model_alias", "model_name", "method", "source"], sort=False
        ).agg(
            n=("response", "size"),
            avg_response_chars=("response", lambda col: col.fillna("").str.len().mean()),
            avg_generation_seconds=("generation_seconds", "mean"),
            avg_generated_tokens=("generated_tokens_including_reasoning", "mean"),
        ).reset_index()
        display_df["method"] = display_df["method"].map(METHOD_NAMES)
        mo.md("## Chẩn đoán response mới\nĐộ dài/thời gian chỉ mô tả output; chưa phải điểm chất lượng tutor.")
        display_df.round(2)

        annotation_template = generations[
            [
                "dialogue_index",
                "conversation_id",
                "source",
                "topic",
                "history",
                "model_alias",
                "model_name",
                "method",
                "model_id",
                "model_revision",
                "data_sha256",
                "response",
            ]
        ].copy()
        for dimension in DIMENSIONS:
            annotation_template[dimension] = ""
        annotation_template.to_csv(ANNOTATION_TEMPLATE_FILE, index=False, encoding="utf-8-sig")
        template_note = mo.md(
            f"Template kiểm định thủ công đã lưu: `{ANNOTATION_TEMPLATE_FILE}`. "
            "Qwen judge tự gán nhãn để tính Qwen-judged DAMR; người có thể kiểm tra mẫu."
        )
    mo.vstack([display_df.round(2), template_note] if not generations.empty else [mo.md("Chưa có output mới.")])
    return (annotation_template, display_df)


@app.cell
def _(ANNOTATED_CSV, DIMENSIONS, DESIRED, mo, normalize_label, pd):
    human_damr = pd.DataFrame()
    if not ANNOTATED_CSV:
        mo.md(
            "## Human DAMR (tùy chọn kiểm định)\n"
            "Qwen-judged DAMR được tính ở phần Qwen3.8 bên dưới. Nếu có human labels "
            "để kiểm định, đặt `ANNOTATED_CSV` thành đường dẫn CSV; kết quả này được "
            "giữ riêng và không gộp với nhãn Qwen."
        )
    else:
        annotated_path = __import__("pathlib").Path(ANNOTATED_CSV)
        if not annotated_path.exists():
            raise FileNotFoundError(annotated_path)
        human_new = pd.read_csv(annotated_path)
        required = {"model_alias", "method", *DIMENSIONS}
        missing_label_columns = required - set(human_new.columns)
        if missing_label_columns:
            raise ValueError(
                f"Thiếu cột trong file nhãn: {sorted(missing_label_columns)}"
            )

        human_scores = []
        for (model_alias, human_method), human_group in human_new.groupby(
            ["model_alias", "method"], sort=False
        ):
            for label_dimension in DIMENSIONS:
                human_labels = human_group[label_dimension].apply(
                    lambda value: normalize_label(label_dimension, value)
                ).dropna()
                human_scores.append(
                    {
                        "method": human_method,
                        "model_alias": model_alias,
                        "dimension": label_dimension,
                        "human_DAMR_pct": (
                            100 * (human_labels == DESIRED[label_dimension]).mean()
                            if len(human_labels)
                            else float("nan")
                        ),
                        "n_human_labeled": int(len(human_labels)),
                    }
                )
        human_damr = pd.DataFrame(human_scores)
        human_damr
    return (human_damr,)


@app.cell
def _(OUT_DIR, mo):
    QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
    QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    QWEN_JUDGE_PROMPT_VERSION = "mrbench_8_dimension_rubric_v1"
    QWEN_JUDGE_MAX_NEW_TOKENS = 384
    MAX_JUDGE_ROWS = None  # None = chạy hết; đặt 12/20 để pilot.
    JUDGE_RELEASE_FOR_AC = True
    JUDGE_GENERATIONS_FOR_DAMR = True
    QWEN_JUDGEMENTS_FILE = OUT_DIR / "qwen38_judgements.csv"
    QWEN_GENERATED_DAMR_FILE = OUT_DIR / "qwen38_generated_damr.csv"
    QWEN_GENERATED_COMPARE_FILE = OUT_DIR / "qwen38_generated_damr_vs_paper.csv"
    QWEN_RELEASE_DAMR_FILE = OUT_DIR / "qwen38_release_damr_vs_paper.csv"
    QWEN_AC_FILE = OUT_DIR / "qwen38_ac_vs_paper.csv"

    qwen_judge_button = mo.ui.run_button(label="Run / resume Qwen3.8-27B judge")
    mo.md(
        "## Qwen3.8-27B judge — DAMR và AC\n"
        f"Model revision cố định: `{QWEN38_REVISION}`. Chạy local BF16 trên GPU, không gọi API. "
        "Mặc định chấm response mới để tính Qwen-judged DAMR **và** chấm lại response "
        "MRBench gốc có nhãn người để tính AC. Checkpoint ghi từng response vào CSV; "
        "nếu phiên dừng, bấm nút lần nữa để tiếp tục. Dùng `MAX_JUDGE_ROWS = 12` trước "
        "để pilot; `None` chạy hết (có thể vài giờ)."
    )
    qwen_judge_button
    return (
        JUDGE_GENERATIONS_FOR_DAMR,
        JUDGE_RELEASE_FOR_AC,
        MAX_JUDGE_ROWS,
        QWEN38_MODEL_ID,
        QWEN38_REVISION,
        QWEN_AC_FILE,
        QWEN_GENERATED_COMPARE_FILE,
        QWEN_GENERATED_DAMR_FILE,
        QWEN_JUDGE_MAX_NEW_TOKENS,
        QWEN_JUDGE_PROMPT_VERSION,
        QWEN_JUDGEMENTS_FILE,
        QWEN_RELEASE_DAMR_FILE,
        qwen_judge_button,
    )


@app.cell
def _(
    AutoModelForMultimodalLM,
    AutoProcessor,
    DATA_PATH,
    DIMENSIONS,
    EXPECTED_DATA_SHA256,
    JUDGE_GENERATIONS_FOR_DAMR,
    JUDGE_RELEASE_FOR_AC,
    MAX_JUDGE_ROWS,
    QWEN38_MODEL_ID,
    QWEN38_REVISION,
    QWEN_JUDGE_MAX_NEW_TOKENS,
    QWEN_JUDGE_PROMPT_VERSION,
    QWEN_JUDGEMENTS_FILE,
    canonical_dimension,
    data_sha256,
    generations,
    normalize_label,
    pd,
    qwen_judge_button,
    re,
    release_df,
    time,
    torch,
    mo,
):
    def qwen38_run_or_load():
        if not qwen_judge_button.value:
            qwen_judgements = (
                pd.read_csv(QWEN_JUDGEMENTS_FILE)
                if QWEN_JUDGEMENTS_FILE.exists()
                else pd.DataFrame()
            )
            planned_judge_rows = (
                (len(release_df) if JUDGE_RELEASE_FOR_AC else 0)
                + (len(generations) if JUDGE_GENERATIONS_FOR_DAMR else 0)
            )
            mo.md(
                f"Qwen judge chưa được chạy trong lượt này. Checkpoint hiện có: "
                f"{len(qwen_judgements)} response. Phạm vi hiện tại tối đa "
                f"{planned_judge_rows} response ({len(release_df) if JUDGE_RELEASE_FOR_AC else 0} "
                f"response release cho AC + {len(generations) if JUDGE_GENERATIONS_FOR_DAMR else 0} "
                    "response mới cho DAMR). Bấm nút ở trên để chạy/tiếp tục."
            )
        else:
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                        "Qwen3.8-27B cần GPU CUDA hỗ trợ BF16. Chọn RTX PRO 6000 Blackwell "
                        "trong moLab notebook specs rồi chạy lại."
                )

            targets = []
            if JUDGE_RELEASE_FOR_AC:
                for row in release_df.to_dict("records"):
                    targets.append(
                        {
                                "evaluation_set": "release_ac",
                                "sample_id": str(row["sample_id"]),
                                "dialogue_index": int(row["dialogue_index"]),
                                "conversation_id": str(row["conversation_id"]),
                                "source": str(row["source"]),
                                "topic": str(row.get("topic", "")),
                                "history": str(row.get("history", "")),
                                "tutor": str(row["tutor"]),
                                "model_alias": "",
                                "model_name": str(row["tutor"]),
                                "method": "original_release",
                                "response": "" if pd.isna(row.get("response")) else str(row["response"]),
                        }
                    )
            if JUDGE_GENERATIONS_FOR_DAMR and not generations.empty:
                for row in generations.to_dict("records"):
                    dialogue_index = int(row["dialogue_index"])
                    model_alias = str(row["model_alias"])
                    method = str(row["method"])
                    targets.append(
                        {
                                "evaluation_set": "generated_damr",
                                "sample_id": f"generated:{dialogue_index}:{model_alias}:{method}",
                                "dialogue_index": dialogue_index,
                                "conversation_id": str(row.get("conversation_id", "")),
                                "source": str(row.get("source", "Unknown")),
                                "topic": str(row.get("topic", "")),
                                "history": str(row.get("history", "")),
                                "tutor": "",
                                "model_alias": model_alias,
                                "model_name": str(row.get("model_name", model_alias)),
                                "method": method,
                                "response": (
                                    ""
                                if pd.isna(row.get("response"))
                                else str(row.get("response", ""))
                            ),
                        }
                    )

            if not targets:
                raise RuntimeError(
                        "Không có response để chấm. Nạp MRBench release hoặc chạy benchmark "
                        "tạo response trước."
                )
            if len({(row["evaluation_set"], row["sample_id"]) for row in targets}) != len(targets):
                raise RuntimeError("Danh sách response có khóa evaluation_set/sample_id bị trùng.")

            old = (
                pd.read_csv(QWEN_JUDGEMENTS_FILE)
                if QWEN_JUDGEMENTS_FILE.exists()
                else pd.DataFrame()
            )
            output_columns = [
                    "evaluation_set",
                    "sample_id",
                    "dialogue_index",
                    "conversation_id",
                    "source",
                    "topic",
                    "history",
                    "tutor",
                    "model_alias",
                    "model_name",
                    "method",
                    "judge_model_id",
                    "judge_model_revision",
                    "data_sha256",
                    "prompt_version",
                    "dtype",
                    "do_sample",
                    "enable_thinking",
                    "max_new_tokens",
                *DIMENSIONS,
                    "parse_ok",
                    "parse_error",
                    "judge_seconds",
                    "raw_output",
            ]
            required_old = {
                    "evaluation_set",
                    "sample_id",
                    "judge_model_id",
                    "judge_model_revision",
                    "data_sha256",
                    "prompt_version",
            }
            if not old.empty:
                missing = required_old - set(old.columns)
                if missing:
                    raise RuntimeError(
                        f"Checkpoint Qwen thiếu cột {sorted(missing)}. Đổi tên "
                            "qwen38_judgements.csv để bắt đầu checkpoint mới."
                    )
                if old.duplicated(["evaluation_set", "sample_id"]).any():
                    raise RuntimeError("Checkpoint Qwen bị trùng evaluation_set/sample_id.")
                if set(old["judge_model_revision"].astype(str)) - {QWEN38_REVISION}:
                    raise RuntimeError("Checkpoint có revision Qwen khác; đổi tên file trước khi chạy.")
                if set(old["judge_model_id"].astype(str)) - {QWEN38_MODEL_ID}:
                    raise RuntimeError("Checkpoint có model ID khác; đổi tên file trước khi chạy.")
                if set(old["data_sha256"].astype(str)) - {EXPECTED_DATA_SHA256}:
                    raise RuntimeError("Checkpoint có checksum dataset khác; đổi tên file trước khi chạy.")
                if set(old["prompt_version"].astype(str)) - {QWEN_JUDGE_PROMPT_VERSION}:
                    raise RuntimeError("Checkpoint có phiên bản prompt khác; đổi tên file trước khi chạy.")
                done = set(zip(old["evaluation_set"].astype(str), old["sample_id"].astype(str)))
            else:
                done = set()

            pending = [
                row
                for row in targets
                if (row["evaluation_set"], row["sample_id"]) not in done
            ]
            if MAX_JUDGE_ROWS is not None:
                pending = pending[: max(0, int(MAX_JUDGE_ROWS))]
            if not pending:
                print("Bỏ qua Qwen judge: mọi response trong phạm vi đã có checkpoint.")
            else:
                model_kwargs = {
                        "revision": QWEN38_REVISION,
                        "device_map": "auto",
                        "dtype": torch.bfloat16,
                }
                try:
                    processor = AutoProcessor.from_pretrained(
                        QWEN38_MODEL_ID, revision=QWEN38_REVISION
                    )
                    model = AutoModelForMultimodalLM.from_pretrained(
                        QWEN38_MODEL_ID, **model_kwargs
                    )
                except Exception as load_error:
                    raise RuntimeError(
                            "Không tải được Qwen3.8-27B. Notebook đã tự cài Transformers/accelerate; "
                            "kiểm tra Internet, dung lượng đĩa cache và VRAM của GPU."
                    ) from load_error

                model.eval()
                resolved_revision = str(
                    getattr(model.config, "_commit_hash", None) or QWEN38_REVISION
                )
                if resolved_revision != QWEN38_REVISION:
                    del model, processor
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"Checkpoint tải về có revision {resolved_revision}, "
                        f"khác revision đã pin {QWEN38_REVISION}."
                    )

                label_guidelines = "\n".join(
                    [
                        "Evaluate the tutor response independently on all eight MRBench dimensions.",
                        "Use the conversation and student mistake as evidence. Do not infer labels from tutor/model identity.",
                        "Apply the rubric:",
                        "1. Mistake_Identification: Yes = accurately recognizes the student's mistake; To some extent = partial/ambiguous recognition; No = misses it.",
                        "2. Mistake_Location: Yes = accurately points to the genuine error/location; To some extent = vague/partial; No = misses or mislocates it.",
                        '3. Revealing_of_the_Answer: choose exactly one of "Yes (and the answer is correct)", "Yes (but the answer is incorrect)", "No". Judge whether the final answer is disclosed, whether correct or incorrect.',
                        "4. Providing_Guidance: Yes = correct and relevant guidance; To some extent = guidance is partial, incomplete, or partly incorrect; No = no guidance.",
                        "5. Actionability: Yes = clear next step; To some extent = partially actionable; No = unclear or no next step.",
                        "6. Coherence: Yes = logically consistent with the conversation; To some extent = partly inconsistent; No = incoherent or contradicts the context.",
                        "7. Tutor_Tone: choose exactly one of Encouraging, Neutral, Offensive.",
                        "8. Humanlikeness: Yes = natural; To some extent = somewhat robotic; No = clearly artificial.",
                        "Return exactly one JSON object, no markdown and no extra text. Use these exact keys: Mistake_Identification, Mistake_Location, Revealing_of_the_Answer, Providing_Guidance, Actionability, Coherence, Tutor_Tone, Humanlikeness.",
                        'Each value must be an object with keys "label" and "evidence". The label must be one exact label listed above; evidence is a short phrase (at most 12 words).',
                        "Treat dimensions as independent.",
                    ]
                )

                def parse_judgement(raw_text):
                    cleaned = re.sub(r"```(?:json)?|```", "", str(raw_text), flags=re.IGNORECASE).strip()
                    parsed = None
                    json_decoder = __import__("json").JSONDecoder()
                    for position, character in enumerate(cleaned):
                        if character != "{":
                            continue
                        try:
                            parsed, _ = json_decoder.raw_decode(cleaned[position:])
                            if isinstance(parsed, dict):
                                break
                        except ValueError:
                            continue
                        parsed = None
                    if not isinstance(parsed, dict):
                        raise ValueError("Model không trả về JSON object hợp lệ.")
                    labels = {dimension: None for dimension in DIMENSIONS}
                    for raw_dimension, raw_label in parsed.items():
                        dimension = canonical_dimension(raw_dimension)
                        if dimension in labels:
                            if isinstance(raw_label, dict):
                                raw_label = raw_label.get("label")
                            labels[dimension] = normalize_label(dimension, raw_label)
                    absent = [dimension for dimension, label in labels.items() if label is None]
                    if absent:
                        raise ValueError(f"Thiếu nhãn hợp lệ cho: {absent}")
                    return labels

                completed_now = 0
                try:
                    for target in pending:
                        user_prompt = (
                            f"Dataset: {target['source']}\n"
                            f"Topic: {target['topic']}\n"
                            f"Conversation history:\n{target['history']}\n\n"
                            f"Tutor response to evaluate:\n{target['response']}"
                        )
                        messages = [
                            {"role": "system", "content": label_guidelines},
                            {"role": "user", "content": user_prompt},
                        ]
                        started = time.time()
                        raw_output = ""
                        parse_error = ""
                        labels = {dimension: None for dimension in DIMENSIONS}
                        if not target["response"].strip():
                            parse_error = "ValueError: Response rỗng."
                        else:
                            encoded = processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=True,
                                return_tensors="pt",
                                enable_thinking=False,
                                preserve_thinking=False,
                            ).to(model.device)
                            input_length = int(encoded["input_ids"].shape[-1])
                            with torch.inference_mode():
                                output_ids = model.generate(
                                    **encoded,
                                    max_new_tokens=QWEN_JUDGE_MAX_NEW_TOKENS,
                                    do_sample=False,
                                    num_beams=1,
                                    use_cache=True,
                                )
                            raw_output = processor.decode(
                                output_ids[0, input_length:], skip_special_tokens=True
                            ).strip()
                            try:
                                labels = parse_judgement(raw_output)
                            except ValueError as parse_exception:
                                parse_error = f"{type(parse_exception).__name__}: {parse_exception}"

                            record = {
                                **target,
                                    "judge_model_id": QWEN38_MODEL_ID,
                                    "judge_model_revision": QWEN38_REVISION,
                                    "data_sha256": EXPECTED_DATA_SHA256,
                                    "prompt_version": QWEN_JUDGE_PROMPT_VERSION,
                                    "dtype": "bfloat16",
                                    "do_sample": False,
                                    "enable_thinking": False,
                                    "max_new_tokens": QWEN_JUDGE_MAX_NEW_TOKENS,
                                **labels,
                                    "parse_ok": not bool(parse_error),
                                    "parse_error": parse_error,
                                    "judge_seconds": round(time.time() - started, 3),
                                    "raw_output": raw_output,
                            }
                            pd.DataFrame([record], columns=output_columns).to_csv(
                                QWEN_JUDGEMENTS_FILE,
                                mode="a",
                                header=not QWEN_JUDGEMENTS_FILE.exists(),
                                index=False,
                                encoding="utf-8",
                            )
                            done.add((target["evaluation_set"], target["sample_id"]))
                            completed_now += 1
                            if completed_now % 10 == 0 or completed_now == len(pending):
                                print(
                                    f"Qwen judge đã checkpoint {completed_now}/{len(pending)} "
                                    f"response ở lượt này."
                                )
                finally:
                    del model, processor
                    torch.cuda.empty_cache()

            qwen_judgements = (
                pd.read_csv(QWEN_JUDGEMENTS_FILE)
                if QWEN_JUDGEMENTS_FILE.exists()
                else pd.DataFrame()
            )
            mo.md(
                f"Đã hoàn tất lượt chấm. Checkpoint: `{QWEN_JUDGEMENTS_FILE}` "
                f"({len(qwen_judgements)} rows; {int(qwen_judgements['parse_ok'].astype(str).str.casefold().eq('true').sum()) if not qwen_judgements.empty else 0} nhãn parse hợp lệ)."
            )
        return qwen_judgements
    qwen_judgements = qwen38_run_or_load()
    return (qwen_judgements,)


@app.cell
def _(
    DIMENSIONS,
    DESIRED,
    METHOD_NAMES,
    PAPER_TABLE3,
    QWEN_AC_FILE,
    QWEN_GENERATED_COMPARE_FILE,
    QWEN_GENERATED_DAMR_FILE,
    QWEN_RELEASE_DAMR_FILE,
    TUTOR_KEYS,
    generations,
    mo,
    normalize_label,
    paper_vs_release,
    pd,
    qwen_judgements,
    release_df,
):
    def qwen38_compute_metrics():
        label_order = {
            dimension: {"Yes": 1, "To some extent": 2, "No": 3}
            for dimension in DIMENSIONS
            if dimension not in {"Revealing_of_the_Answer", "Tutor_Tone"}
        }
        label_order["Revealing_of_the_Answer"] = {
            "Yes (and the answer is correct)": 1,
            "Yes (but the answer is incorrect)": 2,
            "No": 3,
        }
        label_order["Tutor_Tone"] = {"Encouraging": 1, "Neutral": 2, "Offensive": 3}
        label_encoding_note = (
            "Ordinal coding follows Table 4 label order (1, 2, 3); the paper reports "
            "Pearson AC but does not specify its category-to-number mapping."
        )

        paper_ac_prometheus2 = {
            "Novice": [-0.37, 0.09, -0.56, -0.72, 0.15, -0.15, -0.71, 0.18],
            "Expert": [-0.01, -0.25, -0.13, -0.19, -0.08, -0.11, -0.40, 0.01],
            "Phi3": [-0.67, -0.58, -0.51, -0.51, -0.46, -0.33, -0.62, 0.03],
            "Llama-3.1-8B": [-0.12, -0.37, -0.17, 0.04, -0.07, -0.16, -0.29, 0.11],
            "Gemini": [0.02, 0.09, -0.06, -0.16, -0.12, -0.07, -0.24, 0.07],
            "Sonnet": [-0.11, -0.12, -0.21, -0.11, -0.22, -0.08, -0.20, 0.07],
            "Mistral": [-0.06, -0.11, -0.10, -0.23, -0.15, -0.20, -0.19, 0.06],
            "GPT-4": [-0.07, 0.01, -0.20, -0.21, 0.02, -0.02, -0.11, 0.08],
            "Llama-3.1-405B": [-0.03, -0.08, -0.05, -0.05, 0.00, 0.06, -0.13, 0.11],
        }
        paper_ac_llama_critic = {
            "Novice": [-0.42, 0.06, -0.71, -0.80, 0.17, -0.17, -0.77, 0.14],
            "Expert": [-0.03, -0.29, -0.17, -0.23, -0.10, -0.16, -0.49, -0.01],
            "Phi3": [-0.71, -0.67, -0.77, -0.73, -0.61, -0.41, -0.62, 0.04],
            "Llama-3.1-8B": [-0.08, -0.46, -0.17, 0.09, -0.09, -0.23, -0.38, 0.09],
            "Gemini": [0.06, 0.12, -0.11, -0.27, -0.22, -0.09, -0.34, 0.09],
            "Sonnet": [-0.07, -0.17, -0.26, -0.21, -0.29, -0.08, -0.32, 0.07],
            "Mistral": [-0.16, -0.16, -0.16, -0.34, -0.27, -0.28, -0.17, 0.07],
            "GPT-4": [-0.03, 0.01, -0.13, -0.23, 0.01, -0.05, -0.06, 0.10],
            "Llama-3.1-405B": [-0.01, -0.02, -0.01, -0.07, 0.00, 0.02, -0.06, 0.09],
        }
        paper_ac_rows = []
        for critic_name, critic_scores in [
            ("Prometheus2", paper_ac_prometheus2),
            ("Llama-3.1-8B critic", paper_ac_llama_critic),
        ]:
            for tutor, scores in critic_scores.items():
                for dimension, score in zip(DIMENSIONS, scores):
                    paper_ac_rows.append(
                        {
                            "tutor": tutor,
                            "dimension": dimension,
                            f"paper_AC_{critic_name}": score,
                        }
                    )
        paper_ac = pd.DataFrame(paper_ac_rows)
        paper_ac = paper_ac.groupby(["tutor", "dimension"], as_index=False).first()

        generated_damr = pd.DataFrame()
        generated_damr_vs_paper = pd.DataFrame()
        release_qwen_damr_vs_paper = pd.DataFrame()
        qwen_ac_vs_paper = pd.DataFrame()

        if not qwen_judgements.empty:
            qwen_valid = qwen_judgements[
                qwen_judgements["parse_ok"].astype(str).str.casefold().eq("true")
            ].copy()
            generated_judgements = qwen_valid[
                qwen_valid["evaluation_set"] == "generated_damr"
            ]
            generated_scores = []
            for (model_alias, model_name, method, source), group in generated_judgements.groupby(
                ["model_alias", "model_name", "method", "source"], sort=False
            ):
                for dimension in DIMENSIONS:
                    labels = group[dimension].dropna().apply(
                        lambda value: normalize_label(dimension, value)
                    ).dropna()
                    generated_scores.append(
                        {
                            "model_alias": model_alias,
                            "model_name": model_name,
                            "method": method,
                            "method_name": METHOD_NAMES.get(method, method),
                            "source": source,
                            "dimension": dimension,
                            "n_qwen_judged": int(len(labels)),
                            "qwen_judge_DAMR_pct": (
                                100 * (labels == DESIRED[dimension]).mean()
                                if len(labels)
                                else float("nan")
                            ),
                        }
                    )
            generated_damr = pd.DataFrame(generated_scores)
            if not generated_damr.empty:
                generated_damr.to_csv(QWEN_GENERATED_DAMR_FILE, index=False)
                aliases_to_paper = {
                    "llama31_8b": "Llama-3.1-8B",
                    "phi3_mini": "Phi3",
                    "mistral_7b": "Mistral",
                }
                generated_overall = []
                for (model_alias, model_name, method), group in generated_judgements.groupby(
                    ["model_alias", "model_name", "method"], sort=False
                ):
                    for dimension in DIMENSIONS:
                        labels = group[dimension].dropna().apply(
                            lambda value: normalize_label(dimension, value)
                        ).dropna()
                        generated_overall.append(
                            {
                                "model_alias": model_alias,
                                "model_name": model_name,
                                "method": method,
                                "method_name": METHOD_NAMES.get(method, method),
                                "paper_tutor": aliases_to_paper.get(model_alias),
                                "dimension": dimension,
                                "n_qwen_judged": int(len(labels)),
                                "qwen_judge_DAMR_pct": (
                                    100 * (labels == DESIRED[dimension]).mean()
                                    if len(labels)
                                    else float("nan")
                                ),
                            }
                        )
                generated_damr_vs_paper = pd.DataFrame(generated_overall)
                paper_damr_rows = [
                    {"paper_tutor": tutor, "dimension": dim, "paper_human_DAMR_pct": score}
                    for tutor, scores in PAPER_TABLE3.items()
                    for dim, score in zip(DIMENSIONS, scores)
                ]
                generated_damr_vs_paper = generated_damr_vs_paper.merge(
                    pd.DataFrame(paper_damr_rows), on=["paper_tutor", "dimension"], how="left"
                )
                generated_damr_vs_paper["qwen_judge_minus_paper_pp"] = (
                    generated_damr_vs_paper["qwen_judge_DAMR_pct"]
                    - generated_damr_vs_paper["paper_human_DAMR_pct"]
                )
                generated_damr_vs_paper.to_csv(QWEN_GENERATED_COMPARE_FILE, index=False)

            release_judgements = qwen_valid[qwen_valid["evaluation_set"] == "release_ac"].copy()
            if not release_judgements.empty:
                release_scores = []
                for tutor, group in release_judgements.groupby("tutor", sort=False):
                    for dimension in DIMENSIONS:
                        labels = group[dimension].dropna().apply(
                            lambda value: normalize_label(dimension, value)
                        ).dropna()
                        release_scores.append(
                            {
                                "tutor": tutor,
                                "dimension": dimension,
                                "n_qwen_judged": int(len(labels)),
                                "qwen_judge_DAMR_pct": (
                                    100 * (labels == DESIRED[dimension]).mean()
                                    if len(labels)
                                    else float("nan")
                                ),
                            }
                        )
                release_qwen_damr = pd.DataFrame(release_scores)
                release_qwen_damr_vs_paper = release_qwen_damr.merge(
                    paper_vs_release[
                        ["tutor", "dimension", "paper_DAMR_pct", "release_DAMR_pct", "n_release"]
                    ],
                    on=["tutor", "dimension"],
                    how="left",
                )
                release_qwen_damr_vs_paper["qwen_minus_paper_pp"] = (
                    release_qwen_damr_vs_paper["qwen_judge_DAMR_pct"]
                    - release_qwen_damr_vs_paper["paper_DAMR_pct"]
                )
                release_qwen_damr_vs_paper["qwen_minus_release_human_pp"] = (
                    release_qwen_damr_vs_paper["qwen_judge_DAMR_pct"]
                    - release_qwen_damr_vs_paper["release_DAMR_pct"]
                )
                release_qwen_damr_vs_paper.to_csv(QWEN_RELEASE_DAMR_FILE, index=False)

                human = release_df[["sample_id", "tutor", *DIMENSIONS]].rename(
                    columns={dimension: f"{dimension}_human" for dimension in DIMENSIONS}
                )
                judged = release_judgements[["sample_id", *DIMENSIONS]].rename(
                    columns={dimension: f"{dimension}_qwen" for dimension in DIMENSIONS}
                )
                paired = human.merge(judged, on="sample_id", how="inner", validate="one_to_one")
                ac_rows = []
                for tutor, tutor_group in paired.groupby("tutor", sort=False):
                    for dimension in DIMENSIONS:
                        human_labels = tutor_group[f"{dimension}_human"].apply(
                            lambda value: normalize_label(dimension, value)
                        )
                        qwen_labels = tutor_group[f"{dimension}_qwen"].apply(
                            lambda value: normalize_label(dimension, value)
                        )
                        paired_labels = pd.DataFrame(
                            {"human": human_labels, "qwen": qwen_labels}
                        ).dropna()
                        human_scores = paired_labels["human"].map(label_order[dimension])
                        qwen_scores = paired_labels["qwen"].map(label_order[dimension])
                        usable = pd.DataFrame(
                            {"human": human_scores, "qwen": qwen_scores}
                        ).dropna()
                        correlation = (
                            usable["human"].corr(usable["qwen"], method="pearson")
                            if len(usable) > 1
                            else float("nan")
                        )
                        ac_rows.append(
                            {
                                "tutor": tutor,
                                "dimension": dimension,
                                "qwen_pearson_AC": correlation,
                                "n_paired": int(len(usable)),
                                "human_label_count": int(usable["human"].nunique()),
                                "qwen_label_count": int(usable["qwen"].nunique()),
                                "label_encoding": label_encoding_note,
                            }
                        )
                qwen_ac = pd.DataFrame(ac_rows)
                qwen_ac_vs_paper = qwen_ac.merge(
                    paper_ac, on=["tutor", "dimension"], how="left"
                )
                for paper_column in ["paper_AC_Prometheus2", "paper_AC_Llama-3.1-8B critic"]:
                    if paper_column in qwen_ac_vs_paper.columns:
                        delta_column = paper_column.replace("paper_AC_", "qwen_minus_")
                        qwen_ac_vs_paper[delta_column] = (
                            qwen_ac_vs_paper["qwen_pearson_AC"] - qwen_ac_vs_paper[paper_column]
                        )
                qwen_ac_vs_paper["paper_method_note"] = (
                    "Paper Table 5/6 uses two different critics; Qwen3.8 is a new critic. "
                    "Current release counts differ from paper; numeric label coding is not specified there."
                )
                qwen_ac_vs_paper.to_csv(QWEN_AC_FILE, index=False)

        mo.md(
            "## Kết quả Qwen3.8-27B\n"
            "- `qwen38_generated_damr.csv`: Qwen-judged DAMR theo model/phương pháp/nguồn.\n"
            "- `qwen38_generated_damr_vs_paper.csv`: so sánh response sinh lại với human DAMR Table 3; "
            "không phải cùng judge hoặc cùng response gốc.\n"
            "- `qwen38_release_damr_vs_paper.csv`: Qwen chấm response release; đối chiếu Table 3 "
            "và DAMR tính từ nhãn người trong release.\n"
            "- `qwen38_ac_vs_paper.csv`: Pearson AC của Qwen với nhãn người trên đúng các response release "
            "đã ghép cặp; đối chiếu Appendix Tables 5–6.\n\n"
            "Qwen-judged DAMR là proxy của LLM judge. AC của response mới không được tính vì "
            "chưa có nhãn người làm gold."
        )
        mo.vstack(
            [
                generated_damr.round(2),
                generated_damr_vs_paper.round(2),
                release_qwen_damr_vs_paper.round(2),
                qwen_ac_vs_paper.round(3),
            ]
        )
        return (
            generated_damr,
            generated_damr_vs_paper,
            qwen_ac_vs_paper,
            release_qwen_damr_vs_paper,
        )
    generated_damr, generated_damr_vs_paper, qwen_ac_vs_paper, release_qwen_damr_vs_paper = qwen38_compute_metrics()
    return (
        generated_damr,
        generated_damr_vs_paper,
        qwen_ac_vs_paper,
        release_qwen_damr_vs_paper,
    )


@app.cell
def _(OUT_DIR, mo):
    import hashlib as output_zip_hashlib
    import io as output_zip_io
    import json as output_zip_json
    import zipfile as output_zipfile
    from datetime import datetime as output_zip_datetime, timezone as output_zip_timezone

    def build_mrbench_output_bundle():
        output_files = sorted(
            path
            for path in OUT_DIR.rglob("*")
            if path.is_file()
            and path.name != "MANIFEST.json"
            and path.suffix.casefold() != ".zip"
        )
        manifest = {
            "format": "mrbench-output-bundle-v1",
            "created_at_utc": output_zip_datetime.now(output_zip_timezone.utc).isoformat(),
            "purpose": "MRBench model-generation and Qwen judge resumable outputs",
            "files": [],
        }
        output_buffer = output_zip_io.BytesIO()
        with output_zipfile.ZipFile(
            output_buffer, mode="w", compression=output_zipfile.ZIP_DEFLATED
        ) as output_archive:
            for output_path in output_files:
                relative_name = output_path.relative_to(OUT_DIR).as_posix()
                payload = output_path.read_bytes()
                manifest["files"].append(
                    {
                        "name": relative_name,
                        "size_bytes": len(payload),
                        "sha256": output_zip_hashlib.sha256(payload).hexdigest(),
                    }
                )
                output_archive.writestr(relative_name, payload)
            output_archive.writestr(
                "MANIFEST.json",
                output_zip_json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )

        archive_bytes = output_buffer.getvalue()
        # Giữ thêm bản ZIP trong Remote Storage của moLab để có thể tải lại từ Files.
        (OUT_DIR / "mrbench_all_outputs.zip").write_bytes(archive_bytes)
        return archive_bytes

    output_bundle_download = mo.download(
        data=build_mrbench_output_bundle,
        filename="mrbench_all_outputs.zip",
        mimetype="application/zip",
        label="Tạo ZIP và tải tất cả kết quả/checkpoint",
    )
    mo.md(
        "## Xuất kết quả và checkpoint\n"
        "Bấm nút để đóng gói mọi file trong `mrbench_multimodel_outputs` (kể cả checkpoint "
        "CSV đang dùng để resume) thành một ZIP. File được tải về máy và lưu thêm trong "
        "thư mục Files của phiên moLab. Ở phiên mới, upload ZIP này ở mục Khôi phục checkpoint."
    )
    output_bundle_download
    return (build_mrbench_output_bundle,)


@app.cell
def _(build_mrbench_output_bundle, generations, qwen_judgements, qwen_ac_vs_paper, mo):
    # Tự cập nhật ZIP sau khi generation/judge và các metric đã hoàn tất.
    _ = (generations, qwen_judgements, qwen_ac_vs_paper)
    completed_bundle = build_mrbench_output_bundle()
    mo.md(
        f"✅ ZIP checkpoint đã được cập nhật tự động: "
        f"`mrbench_multimodel_outputs/mrbench_all_outputs.zip` "
        f"({len(completed_bundle) / 1024**2:.2f} MiB)."
    )
    return


if __name__ == "__main__":
    app.run()
