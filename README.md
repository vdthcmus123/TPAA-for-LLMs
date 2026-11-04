# Đánh giá năng lực sư phạm của AI Tutor (TPAA for LLMs)

Đồ án môn CSC15011 dựa trên paper [Unifying AI Tutor Evaluation](https://aclanthology.org/2025.naacl-long.57/) (Maurya et al., NAACL 2025) và benchmark MRBench.

Bài toán: cho một đoạn hội thoại gia sư–học sinh (học sinh vừa mắc lỗi toán) và một câu trả lời của tutor, gán 8 nhãn sư phạm: Mistake Identification, Mistake Location, Revealing of the Answer, Providing Guidance, Actionability, Coherence, Tutor Tone, Humanlikeness.

## Trạng thái theo yêu cầu đồ án

| Mục | Nội dung | Trạng thái |
| --- | --- | --- |
| 1 | Đọc và trình bày paper | Báo cáo đối chiếu tại `reports/benchmark_results_project_plan.md` |
| 2 | Baseline NLP trên MRBench | Đã chạy Naive Bayes, Word2Vec + Random Forest, BERT (`src/baseline.py`, kết quả ở `reports/baseline/`) |
| 3 | Phương pháp LLM | Qwen3.8-27B judge zero-shot đã chấm 1.589 response MRBench (`reports/mrbench_molab_outputs.zip`); cần tính macro-F1 trên 314 response test để so với mục 2 |
| 4 | Dataset nhóm tự xây | **TPAA v1**: 150 hội thoại định dạng MRBench (100 kiểu MathDial + 50 kiểu Bridge), xem `data/tpaa_dataset/` |
| 4 | Sinh tutor response | Model mở: **xong** (750 response, 0 lỗi QA). Model API (Gemini 2.5 Flash, Claude Sonnet 5.5, GPT-4o, Llama-3.1-405B qua OpenRouter): **chưa chạy** |
| 4 | Gán 8 nhãn | Chưa thực hiện |
| 5–6 | Chạy phương pháp trên data nhóm, tái hiện phương pháp paper | Chưa thực hiện |

## Cấu trúc repo

```text
data/
  raw/MRBench_V1.json            MRBench V1 gốc (không sửa) + nguồn tải
  processed/                     Bảng response MRBench đã chuẩn hóa (src/baseline.py tạo ra)
  tpaa_dataset/                  Dataset TPAA v1 của nhóm (README riêng)
    tpaa_mrbench_v1.json         150 hội thoại, schema MRBench, nhãn để trống
    tpaa_metadata_v1.csv         Đáp án đúng/sai, loại lỗi, nguồn gốc từng hội thoại
    annotation_sheet_v1.csv      Bảng gán nhãn (Expert/Novice)
    overlap_audit_v1.csv, quality_summary_v1.json, draft_disposition_v0.3.csv
    drafts_v0.3/                 Bản nháp D1–D4 cũ (chỉ để dựng lại dataset)
    generations/open_models/     Kết quả moLab: 750 response của 4 model mở + qa_report
notebooks/
  molab_tpaa_generation.py       Sinh response cho TPAA bằng model mở (moLab, bản v3)
  molab_mrbench_multimodel_benchmark.py   Lần chạy trên MRBench: sinh response + Qwen judge (nguồn số liệu trong báo cáo)
reports/
  benchmark_results_project_plan.md   Báo cáo kết quả MRBench + kế hoạch
  baseline/                      Metrics, predictions, split manifest của mục 2
  mrbench_molab_outputs.zip      Output moLab của lần chạy MRBench
scripts/
  build_tpaa_dataset.py          Dựng + kiểm tra dataset TPAA (nội dung ở tpaa_items_*.py, tpaa_verbose_solutions.py, tpaa_explanations.py, tpaa_expert_revisions.py)
  run_api_tutors.py              Sinh response bằng API (OpenRouter / Gemini / Claude)
  tutor_cleaning.py              Làm sạch + QA response (dùng chung với notebook)
  merge_llm_responses.py         Ghép response LLM vào dataset, tạo bảng gán nhãn
  generate_benchmark_report.py   Dựng lại báo cáo từ ZIP moLab
src/baseline.py                  Audit MRBench + 3 baseline phân loại
```

Không đưa lên Git: checkpoint/cache Hugging Face (`.cache/`), tài liệu môn học (`course_materials/`), thư mục tạm `_to_delete/`, API key và token.

## Chạy baseline

Yêu cầu Python 3.10 trở lên. Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/baseline.py audit
python src/baseline.py run --model all
```

Có thể chạy riêng từng mô hình với `--model nb`, `--model rf` hoặc `--model bert`. Lần chạy BERT đầu tiên cần tải checkpoint `google/bert_uncased_L-2_H-128_A-2` từ Hugging Face. Nếu checkpoint đã có trong cache, thêm `--offline` để cấm tải mạng. Các cờ giảm thời gian chạy như `--w2v-epochs 2 --rf-trees 100 --bert-epochs 1` chỉ dành cho thử nhanh.

Mỗi target là một trong tám chiều của taxonomy. NB dùng TF-IDF unigram/bigram; RF nhận trung bình vector Word2Vec SGNS 100 chiều; BERT dùng encoder nhỏ với tám classification heads. Cả ba dùng cùng split seed 42, chia theo `conversation_id` và có cân bằng nguồn Bridge/MathDial để tránh rò rỉ các response của cùng hội thoại giữa train/test. Kết quả là một lần split cố định, chưa có cross-validation hay tìm siêu tham số.

Lệnh `run` ghi metrics, predictions, audit và split manifest vào `reports/baseline/`; checkpoint BERT cục bộ vào `.cache/local_artifacts/`. Không so sánh macro-F1/accuracy của bài toán phân loại này trực tiếp với DAMR hoặc AC trong paper vì đây là các đại lượng khác nhau.

## Dataset TPAA và sinh response

Chi tiết ở [`data/tpaa_dataset/README.md`](data/tpaa_dataset/README.md). Tóm tắt quy trình:

```powershell
# 1. (Tùy chọn) Dựng lại dataset từ mã nguồn; SHA-256 phải là bd818191...
python scripts/build_tpaa_dataset.py --mrbench data/raw/MRBench_V1.json `
  --drafts data/tpaa_dataset/drafts_v0.3/dataset_draft.jsonl --out data/tpaa_dataset

# 2. Model mở: chạy notebooks/molab_tpaa_generation.py trên moLab (đã xong, kết quả ở generations/open_models/)

# 3. Model API qua OpenRouter (API key đặt trong biến môi trường, không commit)
$env:OPENROUTER_API_KEY="..."
python scripts/run_api_tutors.py --provider openrouter --model google/gemini-2.5-flash --alias Gemini `
  --reasoning-effort none --dataset data/tpaa_dataset/tpaa_mrbench_v1.json --out data/tpaa_dataset/generations/api_responses.csv

# 4. Ghép tất cả và tạo bảng gán nhãn
python scripts/merge_llm_responses.py --dataset data/tpaa_dataset/tpaa_mrbench_v1.json `
  --generations data/tpaa_dataset/generations/open_models/tpaa_open_model_responses_v2.csv `
  --generations data/tpaa_dataset/generations/api_responses.csv `
  --out data/tpaa_dataset/tpaa_mrbench_v1_with_llm.json --sheet data/tpaa_dataset/annotation_sheet_with_llm.csv
```

## Lần chạy trên MRBench (moLab)

`notebooks/molab_mrbench_multimodel_benchmark.py` sinh response bằng prompt Figure 2 của paper cho Llama 3.1 8B, Phi-3 Mini, Mistral 7B và Qwen3.5 9B (Zero-shot/CoT), rồi dùng Qwen3.8-27B làm judge cho response mới và cho 1.589 response gốc có nhãn người (để tính AC). Thiết lập này là tái lập gần đúng: paper không công bố đủ revision/decoding của các tutor gốc, release lệch 7 response so với paper, và judge khác hai judge của paper. Điều kiện "CoT" trong lần chạy này thực tế không kích hoạt suy luận (đã sửa trong notebook TPAA v3). Điền `HF_TOKEN` trong cell cấu hình khi chạy Llama; **không commit token thật**.

Dựng lại báo cáo từ output đã lưu:

```powershell
python scripts/generate_benchmark_report.py --bundle reports/mrbench_molab_outputs.zip
```

## Dữ liệu và sai khác với paper

Paper báo cáo 192 hội thoại (60 Bridge, 132 MathDial) và 1.596 response. File V1 hiện tại trong repo có 192 object hội thoại, 188 `conversation_id` duy nhất, 139 MathDial và 53 Bridge; release tạo 1.589 response có nhãn. Bốn ID Bridge lặp với response khác nhau được giữ thành các mẫu riêng nhưng cùng một split. Audit ghi nhận sai khác thay vì tự tạo hoặc xóa mẫu. Hãy xem [audit dữ liệu](reports/baseline/data_audit.json) và báo cáo chính để biết giới hạn so sánh.

MRBench được phát hành từ MathDial và Bridge. Trước khi chia sẻ bản sao hoặc bản biến đổi dữ liệu, kiểm tra giấy phép và yêu cầu ghi công tại [repo upstream](https://github.com/kaushal0494/UnifyingAITutorEvaluation) và [trang ACL Anthology của paper](https://aclanthology.org/2025.naacl-long.57/).

## Trích dẫn

```bibtex
@inproceedings{maurya-etal-2025-unifying,
  title = {Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors},
  author = {Maurya, Kaushal Kumar and Srivatsa, K. V. Aditya and Petukhova, Kseniia and Kochmar, Ekaterina},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2025},
  pages = {1234--1251},
  url = {https://aclanthology.org/2025.naacl-long.57/}
}
```
