# TPAA dataset v1 — định dạng MRBench

Bộ dữ liệu mới của nhóm cho mục 4 của đồ án. Bộ này **thay thế bản nháp D1–D4 v0.3** (lưu ở `drafts_v0.3/`, chỉ để dựng lại dataset). Định dạng giống hệt `MRBench_V1.json`, nên chạy được ngay với notebook sinh response đa LLM. Cả 8 nhãn đang để trống, chờ nhóm gán.

## Nội dung

| | Kiểu MathDial | Kiểu Bridge | Tổng |
|---|---:|---:|---:|
| Hội thoại (`conversation_id`) | 100 (`TPAA-MD-001…100`) | 50 (`TPAA-BR-001…050`) | **150** |
| Response có sẵn | Expert | Expert + Novice | 200 |
| Response sau khi thêm 7 LLM tutor | 8/hội thoại | 9/hội thoại | **1.250** |
| Split (train/val/test) | 70/10/20 | 35/5/10 | 105/15/30 |

Tỉ lệ MathDial : Bridge ≈ 2 : 1, gần với paper (132 : 60).

## File

| File | Nội dung |
|---|---|
| `tpaa_mrbench_v1.json` | Dataset chính. Cùng key và thứ tự key với MRBench V1: `conversation_id`, `conversation_history`, `Data`, `Split`, `Topic`, `Ground_Truth_Solution`, `anno_llm_responses{tutor:{response, annotation{8 chiều}}}` |
| `tpaa_metadata_v1.csv` | Thông tin truy vết mỗi hội thoại: đáp án đúng, đáp án sai của học sinh, loại lỗi (misconception), bản nháp gốc nếu có, số lượt, số từ |
| `annotation_sheet_v1.csv` | Mỗi dòng là một response cần gán 8 nhãn, có cột cho annotator 1 và 2 cùng người phân xử |
| `overlap_audit_v1.csv` | Láng giềng gần nhất trong MRBench V1 và trong chính bộ này (TF-IDF n-gram ký tự) |
| `quality_summary_v1.json` | Kết quả mọi kiểm tra và thống kê so với MRBench |
| `draft_disposition_v0.3.csv` | Xử lý từng mẫu trong 184 mẫu nháp cũ (viết lại hay loại bỏ, kèm lý do) |

Mã nguồn nằm ở `scripts/`:

| Script | Vai trò |
|---|---|
| `build_tpaa_dataset.py` | Dựng toàn bộ thư mục này |
| `tpaa_items_*.py`, `tpaa_verbose_solutions.py`, `tpaa_explanations.py`, `tpaa_expert_revisions.py` | Nội dung dữ liệu |
| `merge_llm_responses.py` | Ghép response LLM vào dataset |

Lệnh dựng lại:

```bash
python scripts/build_tpaa_dataset.py --mrbench data/raw/MRBench_V1.json \
  --drafts data/tpaa_dataset/drafts_v0.3/dataset_draft.jsonl --out data/tpaa_dataset
```

Thêm `--gsm8k <file.jsonl>` để quét trùng lặp với GSM8K. Bước này chưa chạy vì môi trường dựng không tải được GSM8K.

## Hai kiểu hội thoại

**Kiểu MathDial** (`Data = "MathDial"`)
- Lượt mở đầu cố định giống MathDial: *"Hi, could you please provide a step-by-step solution for the question below? The question is: …"*.
- Tiếp theo là lời giải nhiều bước có lỗi, viết theo văn phong học sinh-LLM dài dòng. Dòng đáp án cuối được nối bằng `\n\u00a0` ngay trong lượt học sinh, giống 119/139 hội thoại MathDial của MRBench.
- Sau đó là 1–3 cặp: giáo viên hỏi, học sinh giải thích.
- Hội thoại dừng ở lượt học sinh vẫn còn lỗi.
- `Topic = "Not Available"`. `Ground_Truth_Solution` gồm các bước giải đúng và dòng đáp án cuối, giống format gốc.

**Kiểu Bridge** (`Data = "Bridge"`)
- Đoạn trích ngắn giữa buổi học, cấp tiểu học/đầu THCS, kết thúc ở câu trả lời sai hoặc nhầm lẫn của học sinh.
- `Topic` dùng lại đúng các nhãn chủ đề có trong MRBench (ví dụ `4.5A.Geometric Lines`).
- `Ground_Truth_Solution = "Not Available"`.

> **Lưu ý:** trường `Data` giữ đúng hai giá trị `MathDial` / `Bridge` vì notebook chọn prompt Figure 2 theo trường này. Ở đây nó có nghĩa là **kiểu định dạng**, không phải nguồn dữ liệu. Toàn bộ dữ liệu là dữ liệu gốc của nhóm, nhận biết qua tiền tố `TPAA-` và file metadata.

## So với MRBench V1

| Thông số | MRBench MathDial | TPAA MathDial | MRBench Bridge | TPAA Bridge |
|---|---:|---:|---:|---:|
| Số lượt trung bình | 5,53 | 5,68 | 4,0 | 3,7 |
| Số từ của ngữ cảnh (trung bình) | 241 | 182 | 25,3 | 23,2 |
| Lượt cuối là học sinh | 95,7% | 100% | 100% | 100% |
| Số từ của Expert | 18,5 | 18,0 | 15,2 | 14,1 |
| Expert là câu hỏi | 68,3% | 70,0% | 52,8% | 62,0% |
| Số từ của Novice | — | — | 9,0 | 6,0 |
| Số chủ đề khác nhau | — | — | 41 | 41 |

Khác biệt còn lại: ngữ cảnh kiểu MathDial ngắn hơn khoảng 25%. Nguyên nhân là đề bài ngắn hơn (32 so với 48 từ) và các lượt giải thích của học sinh ngắn hơn.

## Kiểm tra chất lượng

- Schema khớp MRBench V1, kể cả thứ tự key và 8 key nhãn (`humanlikeness` viết thường như bản gốc).
- 544 phép tính trong lời giải đúng, lời giải sai và lời giải thích được máy tính lại.
  - Lời giải đúng: 0 lỗi.
  - Lời giải học sinh: chỉ 2 lỗi số học, đều là lỗi cố ý (MD-022 đặt sai dấu thập phân, MD-039 tính 5 × 0,8 = 0,40). Mọi lỗi khác của học sinh là lỗi khái niệm.
- 0 hội thoại có đáp án học sinh trùng đáp án đúng. 0 hội thoại kết thúc bằng lượt giáo viên.
- Không trùng lặp:
  - 0 ngữ cảnh và 0 response trùng nguyên văn với MRBench.
  - Độ tương đồng ngữ cảnh cao nhất với MRBench là 0,56 (ngưỡng cờ 0,60).
  - Trong nội bộ bộ dữ liệu, cao nhất 0,69 (ngưỡng 0,70).
  - Không response nào có độ tương đồng ≥ 0,80 với response MRBench.
- Câu hỏi gợi mở của giáo viên được xoay vòng giữa 8 cách diễn đạt. Không còn câu mở đầu template lặp lại như bản nháp cũ.

## Nguồn gốc (cần ghi rõ khi báo cáo)

- Toàn bộ ngữ cảnh, lời giải sai, response Expert và Novice do **AI soạn theo văn phong gia sư người** (Claude, 11/2026), dựa trên các loại lỗi sai trong bản nháp cũ của nhóm.
- 34 mẫu nháp được viết lại thành mẫu mới. Ánh xạ nằm ở cột `derived_from_draft` của file metadata và trong file `draft_disposition_v0.3.csv`.
- Trong paper, response Expert/Novice là **lời người thật**. Vì vậy trước khi phát hành, nhóm nên:
  1. Đọc lại từng hội thoại, sửa hoặc viết lại Expert/Novice theo cách của mình.
  2. Ghi tên người duyệt vào metadata.
  3. Ghi rõ trong báo cáo phần nào AI soạn, phần nào người duyệt hoặc viết.

## Bước tiếp theo

1. **Sinh response.**
   - **Model mở trên moLab:** dùng `notebooks/molab_tpaa_generation.py` (bản v3). Notebook chạy Llama 3.1 8B, Phi-3 Mini, Mistral 7B và Qwen3.5 9B (Zero-shot + CoT có suy luận thật, ngân sách suy luận `MAX_NEW_TOKENS_COT = 1024` token; hết ngân sách thì chốt `</think>` rồi trả lời), tổng cộng 750 response. Nếu chạy tiếp từ một lần chạy cũ, các dòng sinh theo cấu hình cũ tự được chuyển sang `*_superseded.csv` và sinh lại. Tải `tpaa_mrbench_v1.json` vào ô upload; notebook tự kiểm tra SHA-256. Chỉ tải ZIP khi bảng QA báo ✅. Kết quả đã chạy xong (03/11/2026, 750/750 response, 0 lỗi QA) nằm ở `generations/open_models/`: `tpaa_open_model_responses_v2.csv` (`response` là bản sạch, `response_raw` là bản gốc), `qa_report.csv` và `MANIFEST.json`.
   - **Model API qua OpenRouter:** dùng `scripts/run_api_tutors.py --provider openrouter`, mỗi model một lệnh (Gemini 2.5 Flash, Claude Sonnet 5.5, GPT-4o, Llama-3.1-405B). Script ghi chi phí thực tế vào cột `cost_usd` và dùng chung bộ làm sạch/QA (`scripts/tutor_cleaning.py`) với notebook. Ghi kết quả vào `generations/api_responses.csv`.
2. **Ghép response vào dataset.**

   ```bash
   python scripts/merge_llm_responses.py --dataset data/tpaa_dataset/tpaa_mrbench_v1.json \
     --generations data/tpaa_dataset/generations/open_models/tpaa_open_model_responses_v2.csv \
     --generations data/tpaa_dataset/generations/api_responses.csv \
     --out data/tpaa_dataset/tpaa_mrbench_v1_with_llm.json \
     --sheet data/tpaa_dataset/annotation_sheet_with_llm.csv
   ```

   File API chỉ cần các cột `conversation_id`, `tutor` (hoặc `model_alias` + `method`) và `response`. Dòng có `qa_status = error` bị bỏ qua và in ra danh sách, trừ khi thêm `--allow-qa-errors`.
3. **Gán 8 nhãn** trên `annotation_sheet_with_llm.csv`. Dùng đúng tập nhãn của paper:

   | Chiều | Nhãn |
   |---|---|
   | 6 chiều Yes/No | `Yes` / `To some extent` / `No` |
   | Revealing_of_the_Answer | `Yes (and the answer is correct)` / `Yes (but the answer is incorrect)` / `No` |
   | Tutor_Tone | `Encouraging` / `Neutral` / `Offensive` |

   Gán đôi khoảng 20% số hội thoại để tính Cohen κ.
