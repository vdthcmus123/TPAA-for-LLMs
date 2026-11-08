import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    intro = mo.md(r"""
    # TPAA dataset: sinh tutor response bằng model mã nguồn mở

    Notebook sinh response cho **bộ dữ liệu TPAA v1** của nhóm (150 hội thoại định dạng MRBench: 100 kiểu MathDial + 50 kiểu Bridge), dùng đúng cấu hình đã chạy trên MRBench:

    | Model | Điều kiện | Response |
    |---|---|---:|
    | Llama 3.1 8B Instruct | Prompt Figure 2 của paper | 150 |
    | Phi-3 Mini 4K Instruct | Prompt Figure 2 | 150 |
    | Mistral 7B Instruct v0.2 | Prompt Figure 2 | 150 |
    | Qwen3.5 9B | Zero-shot (thinking off) + CoT (thinking on) | 300 |
    | **Tổng** | | **750** |

    Notebook này **chỉ sinh response**. Bộ TPAA chưa có nhãn người, nên phần Qwen judge / DAMR / AC của notebook MRBench được bỏ. Gemini, Sonnet, GPT-4o và Llama-3.1-405B chạy qua OpenRouter bằng `scripts/run_api_tutors.py`, rồi ghép tất cả bằng `scripts/merge_llm_responses.py`.

    ## Chạy trên moLab

    1. Mở [moLab](https://molab.marimo.io/notebooks) và tải file `.py` này lên. Trong **notebook specs**, bật GPU (RTX PRO 6000 Blackwell; card khác cần ≥ 24 GB VRAM và hỗ trợ BF16).
    2. Llama 3.1 8B là model gated: chấp nhận license trên Hugging Face và điền token vào `HF_TOKEN = ""` ở cell cấu hình. **Không commit token lên GitHub.**
    3. Tải file `data/tpaa_dataset/tpaa_mrbench_v1.json` vào ô **Tải dataset TPAA**. Notebook kiểm tra SHA-256 để mọi thành viên chạy trên cùng một phiên bản dữ liệu.
    4. Chạy pilot trước: đặt `MAX_DIALOGUES = 12`, bấm **Run**, đọc vài response. Ổn thì đặt lại `None` rồi bấm Run; các response đã có sẽ được bỏ qua.
    5. Mỗi response được ghi ngay vào CSV checkpoint. Khi xong (hoặc trước khi phiên hết giờ), bấm nút tạo ZIP để tải về. Ở phiên mới, tải ZIP đó vào mục **Khôi phục checkpoint** rồi bấm Run để chạy tiếp.

    Cấu hình sinh giữ nguyên notebook MRBench: BF16, greedy decoding (`do_sample=False`), checkpoint revision cố định, Phi-3 giới hạn input 3.500 token. Với Mistral, system prompt được ghép vào khối `[INST]` vì template của checkpoint không có vai system.

    **Bản v2 (03/11/2026)** sửa các lỗi của lần chạy đầu:
    - **CoT suy luận thật, thời gian có giới hạn:** không điền sẵn đầu lượt trả lời; mở sẵn `<think>`. Ngân sách suy luận `MAX_NEW_TOKENS_COT` (mặc định 1.024 token); hết ngân sách thì chèn câu chốt + `</think>` rồi model trả lời (budget forcing, cột `thinking_budget_forced`). Lưu số token suy luận ở cột `thinking_tokens`.
    - **Chạy tiếp an toàn:** dòng sinh theo cấu hình cũ tự chuyển sang `*_superseded.csv` và được sinh lại; dòng còn hợp lệ được giữ.
    - **Dừng đúng một lượt tutor:** dừng sinh khi model bắt đầu bịa lượt `Student:`/`Tutor response:` hoặc mở đoạn văn thứ hai.
    - **Làm sạch có log:** `response_raw` giữ bản gốc; `response` chỉ sửa hình thức; mọi thay đổi ghi ở `cleaning_actions`.
    - **Cổng QA:** bảng QA phải báo ✅ (0 dòng lỗi) trước khi tải ZIP; chi tiết ở `qa_report.csv`.
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

    EXPECTED_DATA_SHA256 = (
        "bd818191f8b38342b3a2b84c6e2180c1677ab48277813fec28b38ac6dd33aafa"
    )  # SHA-256 của data/tpaa_dataset/tpaa_mrbench_v1.json (bản v1.0, 03/11/2026)
    HF_TOKEN = ""  # Dán token vào giữa dấu nháy, ví dụ: "hf_..."
    OUT_DIR = Path.cwd() / "tpaa_generation_outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH = OUT_DIR / "tpaa_mrbench_v1.json"

    return (
        AutoModelForMultimodalLM,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        accelerate,
        package_installer,
        runtime_sys,
        DATA_PATH,
        EXPECTED_DATA_SHA256,
        HF_TOKEN,
        OUT_DIR,
        hashlib,
        json,
        pd,
        re,
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
        "được phục hồi vào `tpaa_generation_outputs`; sau đó bấm Run/resume như bình thường. "
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
                if manifest.get("format") != "tpaa-output-bundle-v1":
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
def _(mo):
    dataset_upload = mo.ui.file(
        filetypes=[".json"],
        multiple=False,
        kind="area",
        max_size=50_000_000,
        label="Tải data/tpaa_dataset/tpaa_mrbench_v1.json",
    )
    mo.md(
        "## Tải dataset TPAA\n"
        "Kéo thả file `tpaa_mrbench_v1.json` vào đây. Nếu vừa khôi phục ZIP checkpoint "
        "thì dataset đã có sẵn trong thư mục output, không cần tải lại."
    )
    dataset_upload
    return (dataset_upload,)


@app.cell
def _(
    DATA_PATH,
    EXPECTED_DATA_SHA256,
    dataset_upload,
    hashlib,
    json,
    mo,
    pd,
    resume_archive_revision,
):
    _ = resume_archive_revision  # Bảo đảm ZIP được khôi phục trước khi nạp dataset.
    if dataset_upload.value:
        DATA_PATH.write_bytes(dataset_upload.value[0].contents)
    mo.stop(
        not DATA_PATH.exists(),
        mo.md("⏳ Chưa có dataset. Hãy tải `tpaa_mrbench_v1.json` ở ô phía trên."),
    )

    data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if data_sha256.lower() != EXPECTED_DATA_SHA256:
        raise RuntimeError(
            f"SHA-256 của dataset là {data_sha256}, khác bản v1 ({EXPECTED_DATA_SHA256}). "
            "Hãy tải đúng file data/tpaa_dataset/tpaa_mrbench_v1.json. Nếu nhóm cố ý "
            "dựng lại dataset, cập nhật EXPECTED_DATA_SHA256 và dùng OUT_DIR mới để không trộn kết quả."
        )

    dialogues = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(dialogues, list) or not dialogues:
        raise ValueError("Dataset phải là một JSON list khác rỗng.")
    required_keys = {"conversation_id", "conversation_history", "Data", "Topic"}
    missing_keys = [
        d.get("conversation_id", i) for i, d in enumerate(dialogues) if not required_keys <= set(d)
    ]
    if missing_keys:
        raise ValueError(f"Thiếu trường MRBench ở: {missing_keys[:5]}")
    bad_source = sorted({d["Data"] for d in dialogues} - {"MathDial", "Bridge"})
    if bad_source:
        raise ValueError(f"Data phải là MathDial/Bridge để chọn prompt Figure 2, gặp: {bad_source}")
    if len({d["conversation_id"] for d in dialogues}) != len(dialogues):
        raise ValueError("conversation_id bị trùng.")

    source_counts = pd.Series([d["Data"] for d in dialogues]).value_counts()
    mo.md(
        f"✅ Đã nạp **{len(dialogues)}** hội thoại TPAA; phân bố: {source_counts.to_dict()}. "
        f"SHA-256: `{data_sha256}`."
    )
    return data_sha256, dialogues, source_counts


@app.cell
def _(OUT_DIR, mo):
    MAX_DIALOGUES = None  # None = đủ 150 hội thoại; đặt 12 để chạy pilot.
    SEED = 42
    MAX_INPUT_TOKENS_DEFAULT = 8192
    MAX_INPUT_TOKENS_PHI3 = 3500  # Chừa tối thiểu 384 tokens cho context window 4K.
    MAX_NEW_TOKENS_PAPER = 384
    MAX_NEW_TOKENS_ZERO_SHOT = 384
    MAX_NEW_TOKENS_COT = 1024  # NGÂN SÁCH SUY LUẬN của CoT (token trong <think>). Hết ngân sách thì
    # notebook chèn câu chốt + </think> rồi cho model trả lời (budget forcing). Giảm xuống 512 nếu cần nhanh hơn.

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
    GENERATION_FILE = OUT_DIR / "tpaa_open_model_responses_v2.csv"
    mo.md(
        "## Cấu hình thí nghiệm\n"
        "Mặc định chạy đủ ba model dùng Figure 2 và Qwen3.5 ở hai điều kiện (750 response cho 150 hội thoại). "
        "Sửa `MODEL_ALIASES_TO_RUN` để chọn model, hoặc đặt `MAX_DIALOGUES = 12` cho pilot. "
        "Các model paper chỉ chạy `paper_prompt`; Qwen mới chạy Zero-shot/CoT."
    )
    return (
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
def _(mo):
    import re as _re
    CLEANING_VERSION = "tutor_clean_v1"

    # A new dialogue turn invented by the model: "Student:", "Student response:",
    # "Tutor (encouraging):", "Teacher response:" ... not at the very start.
    TURN_TAG_RE = _re.compile(
        r"(?:^|(?<=[\s\"'*(]))(?:\*\*)?(Student|Tutor|Teacher)(?:'s)?(?:\s+(?:response|reply))?"
        r"(?:\s*\([^)]{0,40}\))?(?:\*\*)?\s*:",
        _re.IGNORECASE,
    )
    LEADING_TAG_RE = _re.compile(
        r"^\s*(?:\*\*)?\s*(?:(?:Tutor|Teacher)(?:'s)?(?:\s+(?:response|reply))?|Response|Answer)"
        r"(?:\s*\([^)]{0,80}\))?\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*",
        _re.IGNORECASE,
    )
    REASONING_RE = _re.compile(
        r"^\s*(reasoning|analysis|thought process)\b|^\s*step \d+\s*:|\n\s*(reasoning|analysis)\s*:",
        _re.IGNORECASE,
    )
    SPECIAL_RE = _re.compile(r"<\|[^>]*\|>|</?think>|\[/?INST\]|</?s>")
    META_RE = _re.compile(
        r"\b(the tutor (should|needs|must|can|will)|the (tutor'?s? )?response should|"
        r"the student (is|has|was|needs|made|seems|confus)|student'?s mistake|pedagogical|"
        r"as (a|the) tutor,? I)\b",
        _re.IGNORECASE,
    )
    MARKDOWN_RE = _re.compile(r"\*\*|__|`|^\s*#+\s*|^\s*[-*•]\s+", _re.MULTILINE)
    ITALIC_RE = _re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")


    PREAMBLE_RE = _re.compile(
        r"^\s*(?:the )?student\b|^\s*the correct answer is\b|"
        r"\b(?:a|the) (?:tutor'?s? )?response should\b|\bthe tutor should\b",
        _re.IGNORECASE,
    )


    def drop_preamble(text):
        """Model wrote an analysis paragraph first and the real tutor turn after it
        (labelled "Tutor:" or in quotes): keep the tutor turn, drop the analysis."""
        paras = [p for p in _re.split(r"\n\s*\n", text.strip()) if p.strip()]
        if len(paras) < 2 or not (PREAMBLE_RE.search(paras[0]) or META_RE.search(paras[0])):
            return text, False
        nxt = paras[1].strip()
        if LEADING_TAG_RE.match(nxt) or nxt[:1] in "\"“":
            return "\n\n".join(paras[1:]), True
        return text, False


    def find_new_turn(text):
        """Index where an invented next turn starts (after some real content), or -1."""
        for m in TURN_TAG_RE.finditer(text):
            if text[: m.start()].strip(" \t\n*\"'"):
                return m.start()
        return -1


    def clean_response(raw, hit_max_tokens=False):
        """Return (clean_text, actions). Only formatting is changed."""
        actions = []
        text = str(raw or "")
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
            actions.append("dropped_reasoning_block")
        stripped = SPECIAL_RE.sub("", text)
        if stripped != text:
            actions.append("removed_special_tokens")
            text = stripped
        text = text.strip()
        text, _dropped = drop_preamble(text)
        if _dropped:
            actions.append("dropped_preamble")
        new = LEADING_TAG_RE.sub("", text, count=1)
        if new != text:
            actions.append("stripped_leading_role_tag")
            text = new
        cut = find_new_turn(text)
        if cut >= 0:
            text = text[:cut]
            actions.append("cut_invented_next_turn")
        parts = _re.split(r"\n\s*\n", text.strip(), maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            text = parts[0]
            actions.append("cut_after_first_paragraph")
        new = ITALIC_RE.sub(r"\1", MARKDOWN_RE.sub("", text)).replace("\\*", "*")
        if new != text:
            actions.append("removed_markdown")
            text = new
        if "\n" in text.strip():
            actions.append("joined_lines")
        text = _re.sub(r"\s+", " ", text).strip()
        q = text.strip("\"“”'")
        if q != text and text[:1] in "\"“'" and text[-1:] in "\"”'":
            text = q.strip()
            actions.append("removed_wrapping_quotes")
        if hit_max_tokens and text and text[-1] not in ".?!\"”)":
            last = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
            if last > 0:
                text = text[: last + 1]
                actions.append("trimmed_incomplete_sentence")
        return text, actions


    def qa_check(text, method="", thinking_tokens=None, reasoning_closed=None, hit_max_tokens=False):
        """Return (errors, info). errors must be empty for a usable row."""
        errors, info = [], []
        t = str(text or "")
        words = len(t.split())
        if not t.strip():
            errors.append("empty")
        if SPECIAL_RE.search(t):
            errors.append("special_tokens_left")
        if LEADING_TAG_RE.match(t) or find_new_turn(t) >= 0:
            errors.append("role_tag_left")
        if _re.match(r"^\s*(?:\*\*)?\s*Student(?:'s)?\b[^:\n]{0,30}:", t, _re.IGNORECASE):
            errors.append("response_is_student_turn")
        if "**" in t or "\n" in t:
            errors.append("formatting_left")
        if REASONING_RE.search(t) or (method == "cot" and META_RE.search(t)):
            errors.append("reasoning_leaked")
        elif META_RE.search(t):
            info.append("third_person_meta_comment")
        if t and t[-1] not in ".?!\"”)'":
            (errors if hit_max_tokens else info).append("no_sentence_end")
        number_run = _re.search(r"(?:\b\d+\s*,\s*){12,}", t)
        long_number_run = _re.search(r"(?:\b\d+\s*,\s*){30,}", t)
        if long_number_run or (number_run and hit_max_tokens) or _re.search(
            r"(\b\w+\b(?:\W+\w+){0,3})(?:\W+\1\b){5,}", t, _re.I
        ):
            errors.append("degenerate_repetition")
        elif number_run:
            info.append("long_number_list")
        if method == "cot":
            if not reasoning_closed:
                errors.append("cot_reasoning_not_closed")
            elif (thinking_tokens or 0) < 10:
                errors.append("cot_no_reasoning")
        if words > 80:
            info.append("long_over_80_words")
        if len(_re.findall(r"[.!?](?:\s|$)", t)) > 1:
            info.append("more_than_one_sentence")
        return errors, info

    mo.md(
        "## Làm sạch & kiểm tra response\n"
        "Mỗi response được giữ nguyên bản gốc ở cột `response_raw`. Cột `response` chỉ sửa **hình thức**: "
        "bỏ tiền tố vai `Tutor:`, cắt phần model tự bịa lượt hội thoại tiếp theo, chỉ giữ đoạn đầu, bỏ markdown, "
        "cắt câu dở khi hết token. Không sửa câu chữ. `qa_errors` phải rỗng; `qa_info` chỉ để tham khảo."
    )
    return CLEANING_VERSION, clean_response, find_new_turn, qa_check


@app.cell
def _(
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    CLEANING_VERSION,
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
    clean_response,
    data_sha256,
    dialogues,
    find_new_turn,
    mo,
    pd,
    qa_check,
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
            if method == "cot":
                # Không điền sẵn đầu lượt trả lời: để model suy luận trong <think> trước.
                user = f"{user}\n\n{assistant_prefix}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return messages, assistant_prefix

        def expected_method_settings(alias, method):
            spec = MODEL_SPECS[alias]
            if method == "cot":
                # Tổng trần = ngân sách suy luận + phần trả lời.
                max_new_tokens = MAX_NEW_TOKENS_COT + MAX_NEW_TOKENS_ZERO_SHOT
                prompt_version = f"mrbench_fig2_plus_cot_v3_think{MAX_NEW_TOKENS_COT}"
                enable_thinking = True
            elif method == "zero_shot":
                max_new_tokens = MAX_NEW_TOKENS_ZERO_SHOT
                prompt_version = "mrbench_fig2_zero_shot_v2"
                enable_thinking = False
            else:
                max_new_tokens = MAX_NEW_TOKENS_PAPER
                prompt_version = "mrbench_fig2_paper_prompt_v2"
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
                "stop_rule": "turn_stop_v1",
                "cleaning_version": CLEANING_VERSION,
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
            superseded_index = []
            for row_index, record in zip(old.index, old.to_dict("records")):
                alias = str(record["model_alias"])
                method = str(record["method"])
                if alias not in MODEL_SPECS or method not in MODEL_SPECS[alias]["methods"]:
                    raise RuntimeError(
                        f"Output CSV chứa cấu hình ngoài notebook hiện tại: {alias}/{method}."
                    )
                expected = expected_method_settings(alias, method)
                for field, value in expected.items():
                    actual = record.get(field)
                    if field in {"do_sample", "enable_thinking"}:
                        actual_value = str(actual).casefold() in {"true", "1"}
                        expected_value = bool(value)
                    elif field in {"max_input_tokens", "max_new_tokens"}:
                        actual_value = int(actual) if pd.notna(actual) else None
                        expected_value = int(value)
                    else:
                        actual_value = str(actual)
                        expected_value = str(value)
                    if actual_value != expected_value:
                        superseded_index.append(row_index)
                        break
            if superseded_index:
                # Dòng sinh theo cấu hình cũ: chuyển sang file lưu trữ rồi sinh lại theo cấu hình mới.
                superseded_file = GENERATION_FILE.with_name(GENERATION_FILE.stem + "_superseded.csv")
                old.loc[superseded_index].to_csv(
                    superseded_file, mode="a", header=not superseded_file.exists(),
                    index=False, encoding="utf-8",
                )
                old = old.drop(index=superseded_index)
                old.to_csv(GENERATION_FILE, index=False, encoding="utf-8")
                print(
                    f"Đã chuyển {len(superseded_index)} dòng sinh theo cấu hình cũ sang "
                    f"{superseded_file.name}; các dòng này sẽ được sinh lại."
                )
                keys = list(
                    zip(
                        old["dialogue_index"].astype(int),
                        old["model_alias"].astype(str),
                        old["method"].astype(str),
                    )
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
            if method == "cot":
                tail = tokenizer.decode(encoded["input_ids"][0, -6:], skip_special_tokens=False)
                if "</think>" in tail:
                    raise RuntimeError(
                        "Chat template đã đóng </think> dù enable_thinking=True; CoT sẽ không suy luận."
                    )
                if "<think>" not in tail:
                    think_ids = tokenizer("<think>\n", add_special_tokens=False, return_tensors="pt")["input_ids"]
                    encoded["input_ids"] = torch.cat([encoded["input_ids"], think_ids], dim=-1)
                    encoded["attention_mask"] = torch.cat(
                        [encoded["attention_mask"], torch.ones_like(think_ids)], dim=-1
                    )
                return encoded.to("cuda:0")
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

        from transformers import StoppingCriteria, StoppingCriteriaList

        class TurnStop(StoppingCriteria):
            """Dừng khi model bắt đầu bịa lượt hội thoại mới hoặc mở đoạn văn thứ hai.
            Với CoT, chỉ bắt đầu theo dõi sau khi model đóng </think>."""

            def __init__(self, tokenizer, prompt_len, wait_for_think, think_budget=None):
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.answer_start = None if wait_for_think else 0
                self.think_budget = think_budget
                self.budget_hit = False
                self.fired = False

            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, self.prompt_len:]
                count = int(generated.shape[-1])
                stop = False
                if self.answer_start is None:
                    tail = self.tokenizer.decode(generated[max(0, count - 6):], skip_special_tokens=False)
                    if "</think>" in tail:
                        self.answer_start = count
                    elif self.think_budget is not None and count >= self.think_budget:
                        self.budget_hit = True
                        stop = True
                else:
                    text = self.tokenizer.decode(generated[self.answer_start:], skip_special_tokens=True)
                    stop = find_new_turn(text) >= 0 or bool(re.search(r"\S\s*\n\s*\n\s*\S", text))
                self.fired = self.fired or stop
                return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

        FORCE_CLOSE_THINKING = (
            "\n\nConsidering the limited time, I have to give the tutor response "
            "based on the thinking directly now.\n</think>\n\n"
        )

        def run_generate(model, model_inputs, max_new_tokens, stopper):
            with torch.inference_mode():
                return model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                )

        def generate_once(alias, decoder, model, encoded, method, max_new_tokens):
            is_qwen = MODEL_SPECS[alias]["family"] == "qwen_multimodal"
            tokenizer = decoder.tokenizer if is_qwen else decoder
            input_ids = encoded["input_ids"]
            input_length = int(input_ids.shape[-1])
            if method != "cot":
                stopper = TurnStop(tokenizer, input_length, wait_for_think=False)
                output_ids = run_generate(model, encoded, max_new_tokens, stopper)
                generated_ids = output_ids[0, input_length:]
                return {
                    "raw_text": tokenizer.decode(generated_ids, skip_special_tokens=not is_qwen).strip(),
                    "generated_tokens": int(generated_ids.shape[-1]),
                    "hit_max_tokens": int(generated_ids.shape[-1]) >= max_new_tokens,
                    "reasoning_closed": False,
                    "thinking_tokens": 0,
                    "thinking_budget_forced": False,
                    "stopped_by_rule": bool(stopper.fired),
                }

            think_budget = int(MAX_NEW_TOKENS_COT)
            answer_budget = int(MAX_NEW_TOKENS_ZERO_SHOT)
            stopper = TurnStop(tokenizer, input_length, wait_for_think=True, think_budget=think_budget)
            output_ids = run_generate(model, encoded, think_budget + answer_budget, stopper)
            generated_ids = output_ids[0, input_length:]
            if not stopper.budget_hit:
                answer_tokens = (
                    int(generated_ids.shape[-1]) - stopper.answer_start
                    if stopper.answer_start is not None else 0
                )
                return {
                    "raw_text": tokenizer.decode(generated_ids, skip_special_tokens=False).strip(),
                    "generated_tokens": int(generated_ids.shape[-1]),
                    "hit_max_tokens": int(generated_ids.shape[-1]) >= think_budget + answer_budget,
                    "reasoning_closed": stopper.answer_start is not None,
                    "thinking_tokens": int(
                        stopper.answer_start if stopper.answer_start is not None else generated_ids.shape[-1]
                    ),
                    "thinking_budget_forced": False,
                    "stopped_by_rule": bool(stopper.fired),
                    "answer_tokens": answer_tokens,
                }

            # Hết ngân sách suy luận: chốt </think> rồi để model trả lời dựa trên phần đã nghĩ.
            close_ids = tokenizer(
                FORCE_CLOSE_THINKING, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(output_ids.device)
            continued = torch.cat([output_ids, close_ids], dim=-1)
            added = int(continued.shape[-1]) - input_length
            continued_inputs = {"input_ids": continued, "attention_mask": torch.ones_like(continued)}
            for name, value in encoded.items():
                # Trường phụ cùng độ dài prompt (nếu có): nối thêm 0 cho phần đã sinh.
                if name in {"input_ids", "attention_mask"} or not torch.is_tensor(value):
                    continue
                if value.dim() == 2 and int(value.shape[-1]) == input_length:
                    pad = torch.zeros((value.shape[0], added), dtype=value.dtype, device=value.device)
                    continued_inputs[name] = torch.cat([value, pad], dim=-1)
            answer_stopper = TurnStop(tokenizer, int(continued.shape[-1]), wait_for_think=False)
            final_ids = run_generate(model, continued_inputs, answer_budget, answer_stopper)
            answer_ids = final_ids[0, int(continued.shape[-1]):]
            thinking_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
            return {
                "raw_text": (thinking_text + FORCE_CLOSE_THINKING + answer_text).strip(),
                "generated_tokens": int(generated_ids.shape[-1]) + int(answer_ids.shape[-1]),
                "hit_max_tokens": int(answer_ids.shape[-1]) >= answer_budget,
                "reasoning_closed": True,
                "thinking_tokens": int(generated_ids.shape[-1]),
                "thinking_budget_forced": True,
                "stopped_by_rule": bool(answer_stopper.fired),
            }

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

            alias_seconds = []
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
                        result = generate_once(
                            alias, decoder, model, encoded, method, settings["max_new_tokens"]
                        )
                        answer, cleaning_actions = clean_response(
                            result["raw_text"], hit_max_tokens=result["hit_max_tokens"]
                        )
                        qa_errors, qa_info = qa_check(
                            answer,
                            method=method,
                            thinking_tokens=result["thinking_tokens"],
                            reasoning_closed=result["reasoning_closed"],
                            hit_max_tokens=result["hit_max_tokens"],
                        )
                        if result["thinking_budget_forced"]:
                            qa_info.append("thinking_budget_forced")
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
                            "generated_tokens_including_reasoning": result["generated_tokens"],
                            "thinking_tokens": result["thinking_tokens"],
                            "reasoning_closed": result["reasoning_closed"],
                            "hit_max_tokens": result["hit_max_tokens"],
                            "stopped_by_rule": result["stopped_by_rule"],
                            "thinking_budget_forced": result["thinking_budget_forced"],
                            "generation_seconds": round(time.time() - started, 3),
                            "cleaning_actions": ";".join(cleaning_actions),
                            "qa_errors": ";".join(qa_errors),
                            "qa_info": ";".join(qa_info),
                            "qa_status": "error" if qa_errors else "ok",
                            "response_raw": result["raw_text"],
                            "response": answer,
                        }
                        record_df = pd.DataFrame([record])
                        if GENERATION_FILE.exists():
                            existing_columns = list(pd.read_csv(GENERATION_FILE, nrows=0).columns)
                            if set(record_df.columns) - set(existing_columns):
                                # Cột mới (ví dụ sau khi đổi phiên bản notebook): viết lại file với đủ cột.
                                merged = pd.concat([pd.read_csv(GENERATION_FILE), record_df], ignore_index=True)
                                merged.to_csv(GENERATION_FILE, index=False, encoding="utf-8")
                            else:
                                record_df.reindex(columns=existing_columns).to_csv(
                                    GENERATION_FILE, mode="a", header=False, index=False, encoding="utf-8"
                                )
                        else:
                            record_df.to_csv(GENERATION_FILE, index=False, encoding="utf-8")
                        done.add(key)
                        runs_completed += 1
                        alias_seconds.append(time.time() - started)
                        if runs_completed % 10 == 0:
                            remaining = sum(
                                1 for r in alias_sample for m in spec["methods"]
                                if (int(r.dialogue_index), alias, m) not in done
                            )
                            per_row = sum(alias_seconds) / len(alias_seconds)
                            print(
                                f"Đã sinh mới {runs_completed} response; {alias}/{method} "
                                f"dialogue {row.dialogue_index}: {per_row:.1f} s/response, "
                                f"còn {remaining} lượt của {alias} ≈ {remaining * per_row / 60:.0f} phút; "
                                f"thinking={result['thinking_tokens']} token."
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
    run_button = mo.ui.run_button(label="Run / resume TPAA generation")
    mo.md(
        f"**Cỡ chạy hiện tại:** `MAX_DIALOGUES = {MAX_DIALOGUES}` "
        "(None nghĩa là toàn bộ dữ liệu). Mặc định: 3 model theo prompt Figure 2 + "
        "Qwen3.5 Zero-shot/CoT.\n\n"
    )
    run_button
    return (run_button,)


@app.cell
def _(GENERATION_FILE, METHOD_NAMES, MODEL_ALIASES_TO_RUN, MODEL_SPECS, OUT_DIR, dialogues, generations, mo, pd):
    if generations.empty:
        qa_view = mo.md("Chưa có response. Bấm **Run / resume TPAA generation** ở phía trên.")
    else:
        g = generations.copy()
        g["qa_errors"] = g["qa_errors"].fillna("")
        g["qa_info"] = g["qa_info"].fillna("")
        g["cleaning_actions"] = g["cleaning_actions"].fillna("")
        expected_runs = sum(len(MODEL_SPECS[a]["methods"]) for a in MODEL_ALIASES_TO_RUN)
        expected_rows = expected_runs * len(dialogues)
        summary_df = g.groupby(["model_alias", "method"], sort=False).agg(
            n=("response", "size"),
            qa_error_rows=("qa_status", lambda col: int((col == "error").sum())),
            cleaned_rows=("cleaning_actions", lambda col: int(col.ne("").sum())),
            stopped_by_rule=("stopped_by_rule", lambda col: int(col.astype(str).eq("True").sum())),
            avg_words=("response", lambda col: col.fillna("").str.split().str.len().mean()),
            median_thinking_tokens=("thinking_tokens", "median"),
            avg_seconds=("generation_seconds", "mean"),
            thinking_budget_forced=("qa_info", lambda col: int(col.str.contains("thinking_budget_forced").sum())),
            more_than_one_sentence=("qa_info", lambda col: int(col.str.contains("more_than_one_sentence").sum())),
        ).reset_index()
        summary_df["method"] = summary_df["method"].map(METHOD_NAMES)
        error_rows = g[g["qa_status"] == "error"][
            ["conversation_id", "model_alias", "method", "qa_errors", "response", "response_raw"]
        ]
        cleaned_rows = g[g["cleaning_actions"] != ""][
            ["conversation_id", "model_alias", "method", "cleaning_actions", "response_raw", "response"]
        ]
        qa_report = g[
            ["conversation_id", "model_alias", "method", "qa_status", "qa_errors", "qa_info",
             "cleaning_actions", "thinking_tokens", "reasoning_closed", "hit_max_tokens",
             "stopped_by_rule", "response"]
        ]
        qa_report.to_csv(OUT_DIR / "qa_report.csv", index=False, encoding="utf-8-sig")
        cot = g[g["method"] == "cot"]
        cot_note = (
            f"CoT: trung vị **{int(cot['thinking_tokens'].median())}** token suy luận; "
            f"{int(cot['reasoning_closed'].astype(str).eq('True').sum())}/{len(cot)} lượt đã đóng `</think>`."
            if len(cot) else ""
        )
        complete = len(g) == expected_rows and g.duplicated(["conversation_id", "model_alias", "method"]).sum() == 0
        verdict = (
            "✅ **Đạt:** đủ response và không có lỗi QA. Có thể tải ZIP."
            if complete and error_rows.empty
            else (
                f"⚠️ **Chưa đạt:** {len(g)}/{expected_rows} response, {len(error_rows)} dòng lỗi QA. "
                "Nếu thiếu response, bấm Run để chạy tiếp; nếu có lỗi, xem bảng lỗi bên dưới."
            )
        )
        qa_view = mo.vstack(
            [
                mo.md("## Kiểm tra chất lượng (QA)\n" + verdict + "\n\n" + cot_note),
                summary_df.round(2),
                mo.md(f"**Dòng lỗi QA ({len(error_rows)})** — phải bằng 0:"),
                error_rows if len(error_rows) else mo.md("_Không có._"),
                mo.md(f"**Dòng đã được làm sạch hình thức ({len(cleaned_rows)})** — so sánh bản gốc và bản sạch:"),
                cleaned_rows,
                mo.md(
                    f"Đã lưu `qa_report.csv`. File để ghép vào dataset: `{GENERATION_FILE.name}` "
                    "(cột `response` là bản sạch, `response_raw` là bản gốc)."
                ),
            ]
        )
    qa_view
    return

@app.cell
def _(OUT_DIR, mo):
    import hashlib as output_zip_hashlib
    import io as output_zip_io
    import json as output_zip_json
    import zipfile as output_zipfile
    from datetime import datetime as output_zip_datetime, timezone as output_zip_timezone

    def build_tpaa_output_bundle():
        output_files = sorted(
            path
            for path in OUT_DIR.rglob("*")
            if path.is_file()
            and path.name != "MANIFEST.json"
            and path.suffix.casefold() != ".zip"
        )
        manifest = {
            "format": "tpaa-output-bundle-v1",
            "created_at_utc": output_zip_datetime.now(output_zip_timezone.utc).isoformat(),
            "purpose": "TPAA open-model response generation (resumable checkpoint)",
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
        (OUT_DIR / "tpaa_generation_outputs.zip").write_bytes(archive_bytes)
        return archive_bytes

    output_bundle_download = mo.download(
        data=build_tpaa_output_bundle,
        filename="tpaa_generation_outputs.zip",
        mimetype="application/zip",
        label="Tạo ZIP và tải tất cả kết quả/checkpoint",
    )
    mo.md(
        "## Xuất kết quả và checkpoint\n"
        "Bấm nút để đóng gói mọi file trong `tpaa_generation_outputs` (kể cả checkpoint "
        "CSV đang dùng để resume) thành một ZIP. File được tải về máy và lưu thêm trong "
        "thư mục Files của phiên moLab. Ở phiên mới, upload ZIP này ở mục Khôi phục checkpoint."
    )
    output_bundle_download
    return (build_tpaa_output_bundle,)


@app.cell
def _(build_tpaa_output_bundle, generations, mo):
    # Tự cập nhật ZIP sau mỗi lần generation hoàn tất.
    _ = generations
    completed_bundle = build_tpaa_output_bundle()
    mo.md(
        f"✅ ZIP checkpoint đã được cập nhật tự động: "
        f"`tpaa_generation_outputs/tpaa_generation_outputs.zip` "
        f"({len(completed_bundle) / 1024**2:.2f} MiB)."
    )
    return


if __name__ == "__main__":
    app.run()
