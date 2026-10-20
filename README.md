# Đánh giá năng lực sư phạm của AI Tutor

Dự án môn học dựa trên paper [Unifying AI Tutor Evaluation](https://aclanthology.org/2025.naacl-long.57/) và benchmark MRBench. Repo hiện có hai nhánh thực nghiệm: baseline phân loại nhãn của MRBench và notebook sinh câu trả lời/LLM-as-judge chạy trên moLab.

## Trạng thái

| Hạng mục | Trạng thái |
| --- | --- |
| Baseline NLP trên nhãn MRBench | Đã chạy Naive Bayes, Random Forest + Word2Vec, BERT |
| Sinh response với mô hình Hugging Face | Đã chạy Llama 3.1 8B, Phi-3 Mini, Mistral 7B; Qwen3.5 9B Zero-shot và CoT |
| Đánh giá Qwen judge | Đã chấm 2.549 response; có DAMR và Pearson AC |
| Dataset Toán mới D1–D4 | Đã mở rộng lên 184 context (46/domain), mỗi context có một tutor-response draft; toàn bộ cặp đã được AI rà soát nội dung và quét trùng lặp với MRBench V1; exact match = 0, cặp nội bộ vượt ngưỡng = 0; còn 7 cờ gần từ vựng đã được AI xem là khác nhiệm vụ, chờ nhóm xác nhận |
| Đánh giá baseline/LLM trên dataset mới | Chưa thực hiện |

Kết quả và phân tích đầy đủ, gồm các bảng so sánh theo từng chiều, xem tại [báo cáo đồ án](reports/benchmark_results_project_plan.md). Kết quả baseline có thêm file dự đoán và thống kê lớp tại [`reports/baseline/`](reports/baseline/).

Hướng dẫn chi tiết để tạo dataset mới theo MRBench, định nghĩa quota 250 mẫu/người, bốn vùng Toán học từ mầm non/tiểu học đến cao đẳng/đại học, schema, quy trình tạo context/response, gán nhãn tám chiều và kiểm tra chất lượng nằm tại [hướng dẫn thu thập dataset](docs/huong_dan_thu_thap_dataset_mrbench.md).

Dataset Toán hợp nhất hiện có **184 context (46 mỗi miền D1–D4) và 184 response draft** tại [`data/full_dataset/`](data/full_dataset/). Toàn bộ context đã được so với 192 hội thoại MRBench V1, response so với 1.589 tutor response. Audit chuẩn hóa văn bản không tìm thấy cặp context hoặc response trùng nguyên văn; bộ lọc nội bộ không còn cặp vượt ngưỡng đã đặt. Bảy cờ tương đồng từ vựng với MRBench được ghi kèm nearest-neighbor và ghi chú rà soát AI là khác nhiệm vụ; nhóm cần xác nhận lại trước khi phát hành. Toàn bộ 184 cặp đã được rà soát AI về lỗi học sinh, tính nhất quán toán học và mức độ phù hợp của phản hồi; đây chưa phải xác nhận của giáo viên/nhóm. Các phản hồi vẫn là bản nháp có AI hỗ trợ, chưa phải tutor response do người thật viết/duyệt; tám nhãn MRBench vẫn để trống. Nếu quota là 250 dòng response/người thì 184 cặp hiện chưa đạt quota cuối của nhóm bốn thành viên.

## Cấu trúc repo

```text
data/
  raw/                 MRBench V1 JSON và thông tin nguồn
  processed/           Bảng response đã chuẩn hóa
  full_dataset/        184 context D1–D4, tutor-response drafts và duplicate audit
notebooks/
  molab_mrbench_multimodel_benchmark.py   Notebook marimo đang dùng trên moLab
  archive/             Notebook Kaggle cũ, chỉ lưu để tham khảo
reports/
  baseline/            Metrics, predictions, split manifest và audit
  benchmark_results_project_plan.md      Báo cáo kết quả + kế hoạch đồ án
  README.md            Chỉ mục báo cáo
scripts/
  generate_benchmark_report.py            Dựng lại báo cáo từ ZIP moLab
  build_full_dataset.py                   Dựng và audit bộ context D1–D4
src/baseline.py        Mã chạy audit và ba baseline phân loại
```

Checkpoint BERT, cache Hugging Face, token và output ZIP lớn được giữ ngoài Git. Notebook moLab tạo `mrbench_all_outputs.zip` chứa response, judgement và checkpoint để tải xuống/lưu tiếp phiên; ZIP không nằm trong repo này.

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

Để tạo lại báo cáo từ ZIP tải ở moLab:

```powershell
python scripts/generate_benchmark_report.py --bundle "C:\path\to\mrbench_all_outputs.zip"
```

Script xác minh checksum các file trong manifest trước khi tính bảng. Đầu ra mặc định là `reports/benchmark_results_project_plan.md`.

## Notebook moLab

Mở `notebooks/molab_mrbench_multimodel_benchmark.py` trên [moLab marimo](https://molab.marimo.io/notebooks). Notebook tự cài các package cần thiết vào runtime. Để tải Llama 3.1 8B, cần chấp nhận license trên Hugging Face và điền token cá nhân vào `HF_TOKEN = ""` trong cell cấu hình. **Không commit token thật vào GitHub.** Các model công khai khác không cần token.

Notebook dùng prompt Figure 2 của paper để sinh response cho Llama 3.1 8B, Phi-3 Mini và Mistral 7B; chạy thêm Qwen3.5 9B ở Zero-shot và CoT. Sau đó Qwen3.8-27B chạy cục bộ làm judge cho response mới và response đã có nhãn người. Notebook có checkpoint theo response, khôi phục từ ZIP và bỏ qua các response đã hoàn tất khi chạy tiếp. Thiết lập này là near-reproduction: paper không công bố đầy đủ revision checkpoint/decoding của các tutor model gốc, release hiện tại lệch phân bố 7 response so với paper, và judge mới khác hai judge trong paper.

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
