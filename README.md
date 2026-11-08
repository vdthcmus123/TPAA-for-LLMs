# Đánh giá năng lực sư phạm của AI Tutor (TPAA for LLMs)

Đồ án môn CSC15011 dựa trên paper [Unifying AI Tutor Evaluation](https://aclanthology.org/2025.naacl-long.57/) (Maurya et al., NAACL 2025) và benchmark MRBench.

**Bài toán.** Đầu vào là một đoạn hội thoại gia sư–học sinh, trong đó học sinh vừa mắc lỗi toán, và một câu trả lời của tutor. Cần gán 8 nhãn sư phạm cho câu trả lời đó: Mistake Identification, Mistake Location, Revealing of the Answer, Providing Guidance, Actionability, Coherence, Tutor Tone, Humanlikeness.

## Trạng thái

| Mục | Nội dung | Trạng thái |
| --- | --- | --- |
| 1 | Trình bày paper | Báo cáo đối chiếu: `reports/benchmark_results_project_plan.md` |
| 2 | Baseline NLP trên MRBench | **Xong.** Naive Bayes, Word2Vec + Random Forest, BERT (`src/baseline.py` → `reports/baseline/`) |
| 3 | Phương pháp LLM trên cùng tập test | Qwen3.8-27B judge zero-shot đã chấm 1.589 response MRBench (`reports/mrbench_molab_outputs.zip`). **Còn thiếu:** tính macro-F1 trên 314 response test bằng cùng hàm metric của mục 2 |
| 4 | Dataset nhóm tự xây (TPAA v1) | **Xong phần dữ liệu:** 150 hội thoại, 1.550 response, schema MRBench (`data/tpaa_dataset/`). **Còn thiếu:** gán 8 nhãn và thống kê nhãn |
| 5–6 | Tự chọn | Chưa thực hiện. Xem [Các bước tiếp theo](#các-bước-tiếp-theo) |

### TPAA v1

TPAA v1 gồm 150 hội thoại: 100 kiểu MathDial và 50 kiểu Bridge, chia train/val/test = 105/15/30 hội thoại (1.085/155/310 response). Mỗi hội thoại có 10 tutor; riêng Bridge có thêm Novice như trong MRBench.

| Tutor | Nguồn | Response | Số từ TB |
| --- | --- | ---: | ---: |
| `Expert` | Nhóm soạn | 150 | 16,7 |
| `Novice` | Nhóm soạn (chỉ Bridge) | 50 | 6,0 |
| `Llama318B` | Llama 3.1 8B Instruct (moLab) | 150 | 34,9 |
| `Phi3` | Phi-3 Mini (moLab) | 150 | 28,1 |
| `Mistral` | Mistral 7B Instruct (moLab) | 150 | 47,4 |
| `Qwen359B` | Qwen3.5 9B, prompt Figure 2 (moLab) | 150 | 34,9 |
| `Qwen359BCoT` | Qwen3.5 9B, có suy luận (moLab) | 150 | 27,9 |
| `Gemini` | Gemini 2.5 Flash (OpenRouter) | 150 | 23,8 |
| `Llama3370B` | Llama 3.3 70B Instruct (OpenRouter) | 150 | 41,8 |
| `GPT4o` | GPT-4o 2024-11-20 (OpenRouter) | 150 | 24,8 |
| `Sonnet` | Claude Sonnet 5.5 (OpenRouter) | 150 | 38,5 |
| **Tổng** | | **1.550** | |

- Tất cả response LLM đều qua bộ làm sạch và QA chung (`scripts/tutor_cleaning.py`). Kết quả 0 lỗi.
- Llama-3.1-405B mà paper dùng đã ngừng phục vụ trên OpenRouter, nên nhóm thay bằng Llama 3.3 70B.

## Cấu trúc repo

```text
data/
  raw/MRBench_V1.json              MRBench V1 gốc (không sửa) + nguồn tải
  processed/                       Bảng response MRBench đã chuẩn hóa (src/baseline.py tạo ra)
  tpaa_dataset/                    Dataset TPAA v1 của nhóm (README riêng)
    tpaa_mrbench_v1.json           150 hội thoại, chỉ Expert/Novice, nhãn trống (SHA-256 bd818191…)
    tpaa_mrbench_v1_with_llm.json  Bản đầy đủ: + 9 LLM tutor = 1.550 response, nhãn trống  ← dùng bản này
    annotation_sheet_with_llm.csv  Bảng gán nhãn, mỗi dòng một response (1.550 dòng)      ← gán nhãn ở đây
    tpaa_metadata_v1.csv           Đáp án đúng/sai, loại lỗi, nguồn gốc từng hội thoại
    overlap_audit_v1.csv, quality_summary_v1.json, draft_disposition_v0.3.csv
    drafts_v0.3/                   Bản nháp cũ (chỉ cần khi dựng lại dataset)
    generations/
      open_models/                 Kết quả moLab: 750 response của model mở + qa_report + MANIFEST
      api_responses.csv            600 response của 4 model API (có cost_usd, response_raw)
      api_responses_qa_errors_archive.csv   Lần gọi lỗi QA đã được gọi lại (lưu để truy vết)
notebooks/
  molab_tpaa_generation.py         Sinh response TPAA bằng model mở trên moLab (v3)
  molab_mrbench_multimodel_benchmark.py   Lần chạy trên MRBench: sinh response + Qwen judge (nguồn số liệu báo cáo)
reports/
  benchmark_results_project_plan.md   Báo cáo kết quả MRBench + kế hoạch
  baseline/                        Metrics, predictions, split manifest của mục 2
  mrbench_molab_outputs.zip        Output moLab của lần chạy MRBench
scripts/
  build_tpaa_dataset.py            Dựng + kiểm tra dataset TPAA (nội dung ở tpaa_items_*.py, tpaa_verbose_solutions.py,
                                   tpaa_explanations.py, tpaa_expert_revisions.py)
  run_api_tutors.py                Sinh response bằng API (OpenRouter / Gemini / Claude), tự giãn nhịp khi bị 429
  tutor_cleaning.py                Làm sạch + QA response (dùng chung với notebook)
  recheck_qa.py                    Chấm lại QA trên CSV có sẵn (không gọi API)
  summarize_api_runs.py            Tóm tắt số dòng, lỗi QA, chi phí theo tutor
  merge_llm_responses.py           Ghép response vào dataset, đổi tên tutor theo MRBench, tạo bảng gán nhãn
  generate_benchmark_report.py     Dựng lại báo cáo từ ZIP moLab
src/baseline.py                    Audit MRBench + 3 baseline phân loại
```

Những thứ không đưa lên Git: cache và checkpoint Hugging Face (`.cache/`), tài liệu môn học (`course_materials/`), thư mục tạm `_to_delete/`, API key và token.

## Cài đặt

Cần Python 3.10 trở lên. Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Mục 2: Baseline trên MRBench

```powershell
python src/baseline.py audit
python src/baseline.py run --model all      # hoặc --model nb | rf | bert
```

**Mô hình:**
- NB dùng TF-IDF unigram/bigram.
- RF nhận trung bình vector Word2Vec SGNS 100 chiều.
- BERT dùng encoder nhỏ `google/bert_uncased_L-2_H-128_A-2` với 8 classification head.

**Chia dữ liệu:** cả ba dùng cùng split seed 42, chia theo `conversation_id` và cân bằng hai nguồn Bridge/MathDial. Nhờ vậy các response của cùng một hội thoại không bị lọt sang cả train lẫn test. Đây là một lần split cố định, chưa có cross-validation hay tìm siêu tham số.

**Tùy chọn khi chạy:**
- `--offline`: cấm tải mạng, dùng khi checkpoint BERT đã có trong cache.
- `--w2v-epochs 2 --rf-trees 100 --bert-epochs 1`: chỉ để thử nhanh.

Macro-F1 và accuracy của bài toán phân loại này là đại lượng khác với DAMR và AC trong paper, nên không so sánh trực tiếp với nhau.

## Mục 4: Dataset TPAA

Chi tiết ở [`data/tpaa_dataset/README.md`](data/tpaa_dataset/README.md). Toàn bộ quy trình đã chạy xong. Chỉ cần chạy lại khi muốn tái lập:

```powershell
# 1. Dựng lại dataset từ mã nguồn (SHA-256 phải là bd818191...)
python scripts/build_tpaa_dataset.py --mrbench data/raw/MRBench_V1.json `
  --drafts data/tpaa_dataset/drafts_v0.3/dataset_draft.jsonl --out data/tpaa_dataset

# 2. Model mở: chạy notebooks/molab_tpaa_generation.py trên moLab, tải ZIP về generations/open_models/

# 3. Model API qua OpenRouter (key chỉ đặt trong biến môi trường, không commit)
$env:OPENROUTER_API_KEY="sk-or-..."
python scripts/run_api_tutors.py --provider openrouter --model google/gemini-2.5-flash --alias Gemini `
  --reasoning-effort none --dataset data/tpaa_dataset/tpaa_mrbench_v1.json `
  --out data/tpaa_dataset/generations/api_responses.csv
#    Tương tự với meta-llama/llama-3.3-70b-instruct (Llama3370B), openai/gpt-4o-2024-11-20 (GPT4o, thêm --sleep 4),
#    anthropic/claude-sonnet-5.5 (Sonnet, --reasoning-effort low --max-tokens 2000). Chạy lại cùng lệnh thì script tự chạy tiếp.
python scripts/summarize_api_runs.py data/tpaa_dataset/generations/api_responses.csv

# 4. Ghép tất cả và tạo bảng gán nhãn
python scripts/merge_llm_responses.py --dataset data/tpaa_dataset/tpaa_mrbench_v1.json `
  --generations data/tpaa_dataset/generations/open_models/tpaa_open_model_responses_v2.csv `
  --generations data/tpaa_dataset/generations/api_responses.csv `
  --out data/tpaa_dataset/tpaa_mrbench_v1_with_llm.json --sheet data/tpaa_dataset/annotation_sheet_with_llm.csv
```

## Các bước tiếp theo

### Bước 1. Gán 8 nhãn (mục 4, mỗi người ≥ 250 mẫu)

Làm trên `annotation_sheet_with_llm.csv`, dùng đúng tập nhãn của paper:

| Chiều | Câu hỏi khi gán | Nhãn |
| --- | --- | --- |
| Mistake_Identification | Tutor có nhận ra học sinh đang sai/nhầm không? | `Yes` / `To some extent` / `No` |
| Mistake_Location | Tutor có chỉ đúng chỗ sai không? | `Yes` / `To some extent` / `No` |
| Revealing_of_the_Answer | Tutor có nói thẳng đáp án cuối không? | `Yes (and the answer is correct)` / `Yes (but the answer is incorrect)` / `No` |
| Providing_Guidance | Tutor có gợi ý, giải thích đúng và liên quan không? | `Yes` / `To some extent` / `No` |
| Actionability | Học sinh có biết rõ phải làm gì tiếp không? | `Yes` / `To some extent` / `No` |
| Coherence | Câu trả lời có khớp, logic với hội thoại trước đó không? | `Yes` / `To some extent` / `No` |
| Tutor_Tone | Giọng điệu | `Encouraging` / `Neutral` / `Offensive` |
| humanlikeness | Câu trả lời có tự nhiên như người không? | `Yes` / `To some extent` / `No` |

Cách làm đề xuất:
1. **Chia việc theo hội thoại, không theo dòng.** Một người gán trọn các response của một hội thoại, để so sánh các tutor trên cùng ngữ cảnh. Với 4 người, mỗi người khoảng 37 hội thoại (≈ 390 response).
2. **Gán thử trước.** Cả nhóm cùng gán 5 hội thoại, thảo luận những chỗ lệch rồi chốt hướng dẫn.
3. **Gán đôi 20%.** Khoảng 30 hội thoại (≈ 310 response) do hai người gán độc lập, ghi vào cột `annotator_1` / `annotator_2`, để tính Cohen κ cho từng chiều. Nếu lệch thì `adjudicator` quyết.
4. **Tổng hợp.** Gộp nhãn cuối vào `tpaa_mrbench_v1_with_llm.json`, rồi thống kê: phân bố nhãn theo chiều và theo tutor, κ, và DAMR theo tutor như cách paper so sánh các tutor.

### Bước 2. Hoàn thiện mục 3

Tính macro-F1/accuracy của Qwen judge trên đúng 314 response test MRBench (theo `reports/baseline/split_manifest.json`), bằng cùng hàm metric của `src/baseline.py`, rồi đặt cạnh NB/RF/BERT trong một bảng.

### Bước 3. Mục 5–6 (tự chọn)

- **Chạy phương pháp trên TPAA:** áp NB/RF/BERT (huấn luyện trên MRBench) và LLM judge lên tập test TPAA, so với nhãn người để xem mô hình có tổng quát sang dữ liệu mới không.
- **Tái hiện phân tích của paper trên TPAA:** DAMR theo tutor, AC (Pearson) giữa LLM judge và nhãn người. Có thể dùng Prometheus2 làm judge như paper.
- **So sánh các tutor:** ví dụ tỉ lệ nói thẳng đáp án của Expert so với LLM, và CoT so với Zero-shot của Qwen3.5 9B.

### Bước 4. Báo cáo

Ghi rõ:
- Phần nào do AI soạn, phần nào người viết hoặc duyệt.
- Các sai khác so với paper: Llama 3.3 70B thay cho 405B, `GPT4o` khác model `GPT4` của MRBench, judge khác.
- Chi phí và cấu hình sinh, lấy từ `MANIFEST.json` và các cột `prompt_version` / `model_id`.

## Lần chạy trên MRBench (moLab)

`notebooks/molab_mrbench_multimodel_benchmark.py` sinh response bằng prompt Figure 2 cho Llama 3.1 8B, Phi-3 Mini, Mistral 7B và Qwen3.5 9B (Zero-shot/CoT). Sau đó notebook dùng Qwen3.8-27B làm judge cho cả response mới lẫn 1.589 response gốc có nhãn người, để tính AC.

Đây là tái lập gần đúng, vì ba lý do:
- Paper không công bố đủ revision và cấu hình decoding của các tutor gốc.
- Bản release lệch 7 response so với số liệu trong paper.
- Judge khác hai judge mà paper dùng.

Ngoài ra, điều kiện "CoT" trong lần chạy này thực tế không kích hoạt suy luận; lỗi này đã sửa trong notebook TPAA v3. Khi chạy Llama, điền `HF_TOKEN` trong cell cấu hình và **không commit token thật**.

Dựng lại báo cáo từ output đã lưu:

```powershell
python scripts/generate_benchmark_report.py --bundle reports/mrbench_molab_outputs.zip
```

## Dữ liệu và sai khác với paper

Paper báo cáo 192 hội thoại (60 Bridge, 132 MathDial) và 1.596 response. File V1 trong repo có:
- 192 object hội thoại nhưng chỉ 188 `conversation_id` duy nhất.
- 139 hội thoại MathDial và 53 Bridge.
- 1.589 response có nhãn.

Bốn ID Bridge bị lặp, mỗi bản có response khác nhau. Nhóm giữ chúng thành các mẫu riêng nhưng xếp vào cùng một split. Audit chỉ ghi nhận các sai khác, không tự tạo thêm hay xóa mẫu. Xem [audit dữ liệu](reports/baseline/data_audit.json).

MRBench được phát hành theo giấy phép CC BY-SA 4.0, dựa trên MathDial và Bridge. Trước khi chia sẻ bản sao hoặc bản biến đổi dữ liệu, cần kiểm tra yêu cầu ghi công tại [repo upstream](https://github.com/kaushal0494/UnifyingAITutorEvaluation).

## Trích dẫn

```bibtex
@inproceedings{maurya-etal-2025-unifying,
  title = {Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors},
  author = {Maurya, Kaushal Kumar and Srivatsa, Kv Aditya and Petukhova, Kseniia and Kochmar, Ekaterina},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year = {2025},
  pages = {1234--1251},
  doi = {10.18653/v1/2025.naacl-long.57},
  url = {https://aclanthology.org/2025.naacl-long.57/}
}
```
