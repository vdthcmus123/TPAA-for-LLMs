# MRBench: kết quả thực nghiệm, đối chiếu paper và kế hoạch đồ án

**Ngày chốt số liệu:** 18/10/2026  
**ZIP đầu vào:** `mrbench_all_outputs (1).zip`  
**SHA-256 ZIP:** `5201980a848ca231010a61ccddee7cb0e4f791556fb79428f4adba9b543abfb8`. Manifest có 9 file và đã xác minh checksum/kích thước.  
**Nguyên tắc:** phân biệt nhãn người trong release với nhãn do Qwen judge dự đoán; chỉ gọi là tái lập gần khi dữ liệu, checkpoint hoặc judge khác paper.

## 1. Tóm tắt

Repo đã chạy ba baseline phân loại tám nhãn MRBench: Naive Bayes + TF-IDF, Random Forest + Word2Vec SGNS, và BERT nhỏ đa nhiệm. Cả ba dùng cùng test set 314 response, chia theo hội thoại. Macro-F1 trung bình tám chiều lần lượt là **0.344, 0.380 và 0.332**. Đây là bài toán dự đoán nhãn, khác với DAMR và AC của paper.

Notebook moLab sinh **960 response** trên 192 hội thoại với năm cấu hình (ba mô hình theo prompt Figure 2 và Qwen3.5-9B Zero-shot/CoT), sau đó Qwen3.8-27B chấm **2549 response**: 1589 response gốc để tính AC và 960 response mới để tính DAMR proxy. Có 2549/2549 judgement parse thành công.

DAMR tính lại từ nhãn người của release nhìn chung khá gần Table 3 ở phần lớn tutor nhưng lệch đáng kể ở Gemini và một số chiều của Expert/Novice. Qwen judge có Pearson AC trung bình mô tả **0.408** trên 72 ô tutor×dimension, so với **-0.153** của Prometheus2 và **-0.190** của Llama critic trong các bảng paper. Đây chưa phải so sánh cùng điều kiện. DAMR trên response mới hiện do Qwen judge chấm, chưa có xác nhận người.

## 2. Dataset, thiết lập và phạm vi táي lập

| Hạng mục | Thiết lập / số liệu |
| --- | --- |
| MRBench V1 trong repo/ZIP | 192 dialogue objects; 139 MathDial, 53 Bridge; 188 conversation ID duy nhất; 1.589 response gốc có nhãn |
| Paper | 132 MathDial + 60 Bridge; 192 hội thoại; 1.596 response (7 LLM × 192 + Expert × 192 + Novice × 60) |
| Sai khác release | So với paper, MathDial nhiều hơn 7 và Bridge ít hơn 7; Novice hiện có 53 response thay vì 60. Bốn ID Bridge lặp với history giống nhau nhưng response variant khác nhau; baseline giữ response và gom cùng split |
| Baseline split | Seed 42; train 150 conversation IDs / 1.275 response; test 38 IDs / 314 response; giữ nguyên hội thoại trong một split và phân tầng nguồn |
| Response sinh mới | 960 = 5 cấu hình × 192 dialogue; `do_sample=False`; BF16; Qwen3.5 thinking off cho Zero-shot, thinking on cho CoT |
| Judge | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 2.549 judgement; checkpoint theo response |
| Tài nguyên moLab | Ảnh phiên chạy hiển thị 4 CPU, 32 GiB RAM, RTX Pro 6000. Tổng `judge_seconds` cộng dồn khoảng 12.1 giờ, không phải wall-clock của phiên |

Model ID, revision và cấu hình generation được lưu trong từng dòng CSV. Các revision dưới đây là commit Hugging Face được notebook dùng ở lần chạy này:

| Model | Hugging Face ID | Revision | Condition | dtype / sampling | max input tokens | max new tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 | Paper Figure 2 prompt | bfloat16; do_sample=False; thinking=False | 8192 | 384 |
| Mistral 7B Instruct v0.2 | mistralai/Mistral-7B-Instruct-v0.2 | b9cbf4f579dff85096d3528ab81b46ceacf99bdd | Paper Figure 2 prompt | bfloat16; do_sample=False; thinking=False | 8192 | 384 |
| Phi-3 Mini 4K | microsoft/Phi-3-mini-4k-instruct | f39ac1d28e925b323eae81227eaba4464caced4e | Paper Figure 2 prompt | bfloat16; do_sample=False; thinking=False | 3500 | 384 |
| Qwen3.5 9B | Qwen/Qwen3.5-9B | cc5442c03a5c0bff0bd4c6888d9a40029c637733 | CoT (thinking on) | bfloat16; do_sample=False; thinking=True | 8192 | 1536 |
| Qwen3.5 9B | Qwen/Qwen3.5-9B | cc5442c03a5c0bff0bd4c6888d9a40029c637733 | Zero-shot (thinking off) | bfloat16; do_sample=False; thinking=False | 8192 | 384 |

Paper mô tả bốn annotator đã huấn luyện và coi human labels là gold. So sánh trong báo cáo dùng JSON release hiện tại, không phải snapshot gốc đã được xác minh trùng từng dòng với paper.

## 3. Baseline phân loại

Input của mỗi ví dụ là conversation history và tutor response; output là dự đoán riêng cho tám dimension. Ô ghi **Macro-F1 / Accuracy / Weighted-F1**. Macro-F1 là metric chính do lệch lớp.

| Dimension | Naive Bayes + TF-IDF | RF + Word2Vec | BERT nhỏ |
| --- | --- | --- | --- |
| Nhận diện lỗi | 0.312 / 0.768 / 0.690 | 0.376 / 0.777 / 0.724 | 0.293 / 0.783 / 0.688 |
| Xác định vị trí lỗi | 0.474 / 0.704 / 0.648 | 0.448 / 0.621 / 0.604 | 0.411 / 0.694 / 0.636 |
| Có tiết lộ đáp án | 0.306 / 0.847 / 0.777 | 0.408 / 0.866 / 0.826 | 0.336 / 0.854 / 0.792 |
| Cung cấp hướng dẫn | 0.348 / 0.567 / 0.472 | 0.401 / 0.538 / 0.490 | 0.328 / 0.545 / 0.451 |
| Tính hành động | 0.413 / 0.656 / 0.584 | 0.420 / 0.631 / 0.584 | 0.368 / 0.586 / 0.526 |
| Mạch lạc | 0.297 / 0.806 / 0.719 | 0.313 / 0.793 / 0.720 | 0.297 / 0.806 / 0.719 |
| Giọng điệu gia sư | 0.295 / 0.640 / 0.567 | 0.366 / 0.682 / 0.641 | 0.317 / 0.704 / 0.606 |
| Tính tự nhiên như người | 0.308 / 0.857 / 0.791 | 0.308 / 0.857 / 0.791 | 0.308 / 0.857 / 0.791 |

Trung bình không trọng số qua tám chiều:

| Mô hình | Macro-F1 mean | Accuracy mean | Weighted-F1 mean |
| --- | --- | --- | --- |
| Naive Bayes | 0.3441 | 0.7305 | 0.6560 |
| RF + Word2Vec | 0.3799 | 0.7205 | 0.6724 |
| BERT nhỏ | 0.3322 | 0.7285 | 0.6512 |

RF + Word2Vec dẫn đầu mean macro-F1 trong split này (0.3799), NB tốt nhất ở Mistake Location; BERT chưa vượt baseline cổ điển. Đây mới là một split cố định; một số lớp hiếm (Offensive chỉ có một nhãn trong audit), nên không khái quát hóa mạnh. Paper không báo cáo các metric phân loại này; không đối chiếu F1/accuracy trực tiếp với DAMR hoặc AC.

Thiết lập: NB dùng TF-IDF unigram/bigram và MultinomialNB (`alpha=0.3`); RF dùng Word2Vec Skip-Gram with Negative Sampling 100 chiều học trên train, mean pooling và Random Forest 300 cây; BERT dùng `google/bert_uncased_L-2_H-128_A-2`, tám classification heads, fine-tune 3 epoch. Tham số đầy đủ, predictions và split manifest nằm trong `reports/baseline/`.

## 4. DAMR nhãn người: release so với paper Table 3

Paper định nghĩa DAMR là tỷ lệ response đạt nhãn mong muốn: Yes cho nhận diện/vị trí lỗi, hướng dẫn, actionability, coherence và human-likeness; No cho tiết lộ đáp án; Encouraging cho tone. Bảng này tính lại DAMR từ nhãn người sẵn có trong release. MAE là sai số tuyệt đối trung bình tám chiều.

| Tutor | Paper mean 8 chiều | Release human mean | Delta signed (pp) | MAE (pp) |
| --- | --- | --- | --- | --- |
| Novice | 41.04% | 39.86% | -1.18 | 7.64 |
| Expert | 78.97% | 74.93% | -4.04 | 14.71 |
| Llama-3.1-8B | 61.39% | 62.83% | +1.43 | 1.43 |
| Phi3 | 36.91% | 37.76% | +0.85 | 1.24 |
| Gemini | 49.67% | 72.59% | +22.92 | 22.92 |
| Sonnet | 76.24% | 78.52% | +2.28 | 2.28 |
| Mistral | 73.05% | 74.54% | +1.50 | 1.50 |
| GPT-4 | 71.43% | 72.66% | +1.23 | 1.36 |
| Llama-3.1-405B | 76.17% | 77.54% | +1.37 | 1.37 |

Trên toàn bộ 72 ô, release trung bình cao hơn paper **+2.93 pp**, MAE **6.05 pp**. Gemini lệch trung bình +22.92 pp; riêng Expert ở Tutor Tone lệch -75.00 pp, còn Novice có MAE 7.64 pp và chỉ có 53 mẫu hiện tại thay vì 60. Đây là khác biệt đủ lớn để ưu tiên kiểm tra snapshot, join tutor/response, mã hóa tone và nhãn trước khi diễn giải như khác biệt thực chất về tutor.

## 5. DAMR do Qwen judge chấm

### 5.1 Response gốc đã có nhãn người

Qwen chấm 1589 response gốc và so với gold label hiện có. Trung bình unweighted 72 ô Qwen-judged DAMR là **65.91%**, so với **62.76%** ở Table 3; signed difference **+3.15 pp**, MAE **12.84 pp**. So với nhãn người cùng release, signed difference là **+0.22 pp**, nhưng MAE vẫn **10.74 pp**. Mean gần 0 che lỗi theo chiều: Tutor Tone MAE 28.72 pp, Mistake Identification 17.57 pp; Revealing Answer MAE 1.39 pp.

### 5.2 960 response mới

Các điểm là tỷ lệ đạt nhãn mong muốn do Qwen3.8-27B phán đoán; chưa có human gold. Ba model đầu trùng họ với model paper và dùng prompt Figure 2. Qwen3.5 là phần mở rộng. Gộp Bridge/MathDial có trọng số theo số mẫu của từng nguồn (53 Bridge, 139 MathDial).

| Dimension | Llama 3.1 8B | Phi-3 Mini | Mistral 7B | Qwen3.5 Zero-shot | Qwen3.5 CoT |
| --- | --- | --- | --- | --- | --- |
| Nhận diện lỗi | 48.44% | 32.29% | 25.00% | 68.75% | 65.10% |
| Xác định vị trí lỗi | 48.96% | 31.25% | 23.96% | 68.75% | 64.58% |
| Có tiết lộ đáp án | 83.85% | 42.71% | 37.50% | 44.79% | 59.90% |
| Cung cấp hướng dẫn | 45.31% | 24.48% | 19.27% | 72.92% | 70.83% |
| Tính hành động | 43.23% | 26.04% | 18.23% | 71.35% | 71.35% |
| Mạch lạc | 71.88% | 45.31% | 27.60% | 76.04% | 76.56% |
| Giọng điệu gia sư | 58.33% | 77.08% | 48.96% | 80.73% | 83.33% |
| Tính tự nhiên như người | 82.81% | 74.48% | 35.94% | 84.38% | 86.98% |

So sánh trung bình tám chiều với paper human DAMR Table 3 (chỉ ba model tương ứng):

| Model / condition | Qwen-judged mean | Paper human mean | Delta mean (pp) | MAE 8 dims (pp) |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B — Paper Figure 2 prompt | 60.35% | 61.39% | -1.04 | 13.28 |
| Phi-3 Mini 4K — Paper Figure 2 prompt | 44.21% | 36.91% | +7.29 | 15.10 |
| Mistral 7B Instruct v0.2 — Paper Figure 2 prompt | 29.56% | 73.05% | -43.49 | 51.95 |

So sánh chi tiết từng dimension của ba model có trong Table 3. Model sinh response là cùng họ/checkpoint gần tương ứng; cột generated được Qwen judge chấm còn paper là human DAMR:

<details><summary>Mở bảng 24 ô generated DAMR so với paper</summary>

| Model | Condition | Dimension | N | Qwen-judged generated % | Paper human % | Delta (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B | Paper Figure 2 prompt | Nhận diện lỗi | 192 | 48.44% | 80.21% | -31.77 |
| Llama 3.1 8B | Paper Figure 2 prompt | Xác định vị trí lỗi | 192 | 48.96% | 54.69% | -5.73 |
| Llama 3.1 8B | Paper Figure 2 prompt | Có tiết lộ đáp án | 192 | 83.85% | 73.96% | +9.89 |
| Llama 3.1 8B | Paper Figure 2 prompt | Cung cấp hướng dẫn | 192 | 45.31% | 45.31% | +0.00 |
| Llama 3.1 8B | Paper Figure 2 prompt | Tính hành động | 192 | 43.23% | 42.71% | +0.52 |
| Llama 3.1 8B | Paper Figure 2 prompt | Mạch lạc | 192 | 71.88% | 80.73% | -8.86 |
| Llama 3.1 8B | Paper Figure 2 prompt | Giọng điệu gia sư | 192 | 58.33% | 19.79% | +38.54 |
| Llama 3.1 8B | Paper Figure 2 prompt | Tính tự nhiên như người | 192 | 82.81% | 93.75% | -10.94 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Nhận diện lỗi | 192 | 25.00% | 93.23% | -68.23 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Xác định vị trí lỗi | 192 | 23.96% | 73.44% | -49.48 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Có tiết lộ đáp án | 192 | 37.50% | 86.46% | -48.96 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Cung cấp hướng dẫn | 192 | 19.27% | 63.54% | -44.27 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Tính hành động | 192 | 18.23% | 70.31% | -52.08 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Mạch lạc | 192 | 27.60% | 86.98% | -59.38 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Giọng điệu gia sư | 192 | 48.96% | 15.10% | +33.86 |
| Mistral 7B Instruct v0.2 | Paper Figure 2 prompt | Tính tự nhiên như người | 192 | 35.94% | 95.31% | -59.37 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Nhận diện lỗi | 192 | 32.29% | 28.65% | +3.64 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Xác định vị trí lỗi | 192 | 31.25% | 26.04% | +5.21 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Có tiết lộ đáp án | 192 | 42.71% | 73.96% | -31.25 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Cung cấp hướng dẫn | 192 | 24.48% | 17.71% | +6.77 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Tính hành động | 192 | 26.04% | 11.98% | +14.06 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Mạch lạc | 192 | 45.31% | 39.58% | +5.73 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Giọng điệu gia sư | 192 | 77.08% | 45.31% | +31.77 |
| Phi-3 Mini 4K | Paper Figure 2 prompt | Tính tự nhiên như người | 192 | 74.48% | 52.08% | +22.40 |

</details>

Llama 3.1 8B gần paper nhất về macro mean (-1.04 pp), nhưng MAE theo chiều 13.28 pp. Phi-3 Mini cao hơn paper +7.29 pp (MAE 15.10 pp); Mistral thấp hơn -43.49 pp (MAE 51.95 pp). Qwen judge khác annotator gold và checkpoint/decoding paper không đầy đủ, nên không kết luận trực tiếp tốt hơn/xấu hơn.

Qwen3.5 Zero-shot đạt mean 70.96%, CoT 72.33% (+1.37 pp). CoT tăng tỷ lệ đạt “không tiết lộ đáp án” 15.10 pp nhưng giảm Mistake Identification 3.65 pp và Mistake Location 4.17 pp. Đây là mô tả trên cùng 192 dialogue; chưa kiểm định ý nghĩa thống kê hoặc xác nhận nhãn người.

Mean DAMR theo nguồn và tám dimensions:

| Model / condition | Source | N | Mean DAMR |
| --- | --- | --- | --- |
| Llama 3.1 8B | Bridge | 53 | 70.99% |
| Llama 3.1 8B | MathDial | 139 | 56.29% |
| Phi-3 Mini | Bridge | 53 | 58.96% |
| Phi-3 Mini | MathDial | 139 | 38.58% |
| Mistral 7B | Bridge | 53 | 43.87% |
| Mistral 7B | MathDial | 139 | 24.10% |
| Qwen3.5 Zero-shot | Bridge | 53 | 77.36% |
| Qwen3.5 Zero-shot | MathDial | 139 | 68.53% |
| Qwen3.5 CoT | Bridge | 53 | 71.70% |
| Qwen3.5 CoT | MathDial | 139 | 72.57% |

Zero-shot tốt hơn trên Bridge (77.36%) so với MathDial (68.53%); CoT đạt 71.70% trên Bridge và 72.57% trên MathDial. Luôn báo cáo theo nguồn vì độ dài/bối cảnh và độ khó khác nhau.

## 6. Annotation Correlation (AC)

Paper dùng Pearson giữa critic annotations và human annotations, báo riêng từng tutor×dimension cho Prometheus2 và Llama-3.1-8B critic ở Tables 5–6. Notebook tính Pearson tương tự cho Qwen3.8 so với human labels hiện tại. Paper không nêu cách mã hóa nhãn phân loại thành số; notebook dùng ordinal 1–3 theo thứ tự nhãn trong rubric, nên so sánh là gần đúng. Pearson trên ordinal không đo agreement tuyệt đối và không thay thế Cohen/Fleiss kappa.

Trên 72 ô, Qwen AC mean=0.408, median=0.360; 70/72 ô dương, min=-0.061, max=1.000. Mean số học các ô paper trong CSV là Prometheus2 -0.153, Llama critic -0.190. Đây không phải chỉ số gộp do paper công bố; số mẫu, Qwen judge và mã hóa khác làm hạn chế so sánh.

Mean theo dimension:

| Dimension | Qwen3.8 r | Prometheus2 paper r | Llama critic paper r |
| --- | --- | --- | --- |
| Nhận diện lỗi | 0.396 | -0.158 | -0.161 |
| Xác định vị trí lỗi | 0.463 | -0.147 | -0.176 |
| Có tiết lộ đáp án | 0.840 | -0.221 | -0.277 |
| Cung cấp hướng dẫn | 0.389 | -0.238 | -0.310 |
| Tính hành động | 0.350 | -0.103 | -0.156 |
| Mạch lạc | 0.319 | -0.118 | -0.161 |
| Giọng điệu gia sư | 0.405 | -0.321 | -0.357 |
| Tính tự nhiên như người | 0.100 | 0.080 | 0.076 |

Mean theo tutor (N là số cặp hiện tại, Novice=53; paper dùng 60):

| Tutor | N | Qwen3.8 r | Prometheus2 paper r | Llama critic paper r |
| --- | --- | --- | --- | --- |
| Novice | 53 | 0.499 | -0.261 | -0.312 |
| Expert | 192 | 0.398 | -0.145 | -0.185 |
| Llama-3.1-8B | 192 | 0.377 | -0.129 | -0.154 |
| Phi3 | 192 | 0.695 | -0.456 | -0.560 |
| Gemini | 192 | 0.368 | -0.059 | -0.095 |
| Sonnet | 192 | 0.375 | -0.122 | -0.166 |
| Mistral | 192 | 0.295 | -0.122 | -0.184 |
| GPT-4 | 192 | 0.349 | -0.062 | -0.048 |
| Llama-3.1-405B | 192 | 0.314 | -0.021 | -0.008 |

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

| Tutor | Dimension | N hiện tại | Paper human % | Release human % | Qwen judge % | Release−paper pp | Qwen−paper pp | Qwen−human pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novice | Nhận diện lỗi | 53 | 43.33% | 49.06% | 32.08% | +5.73 | -11.25 | -16.98 |
| Novice | Xác định vị trí lỗi | 53 | 16.67% | 16.98% | 11.32% | +0.31 | -5.35 | -5.66 |
| Novice | Có tiết lộ đáp án | 53 | 80.00% | 88.68% | 88.68% | +8.68 | +8.68 | +0.00 |
| Novice | Cung cấp hướng dẫn | 53 | 11.67% | 13.21% | 5.66% | +1.54 | -6.01 | -7.55 |
| Novice | Tính hành động | 53 | 1.67% | 1.89% | 7.55% | +0.22 | +5.88 | +5.66 |
| Novice | Mạch lạc | 53 | 50.00% | 56.60% | 60.38% | +6.60 | +10.38 | +3.77 |
| Novice | Giọng điệu gia sư | 53 | 90.00% | 54.72% | 67.92% | -35.28 | -22.08 | +13.21 |
| Novice | Tính tự nhiên như người | 53 | 35.00% | 37.74% | 43.40% | +2.74 | +8.40 | +5.66 |
| Expert | Nhận diện lỗi | 192 | 76.04% | 81.25% | 65.62% | +5.21 | -10.42 | -15.62 |
| Expert | Xác định vị trí lỗi | 192 | 63.02% | 68.75% | 61.98% | +5.73 | -1.04 | -6.77 |
| Expert | Có tiết lộ đáp án | 192 | 90.62% | 97.92% | 97.92% | +7.30 | +7.30 | +0.00 |
| Expert | Cung cấp hướng dẫn | 192 | 67.19% | 72.92% | 78.12% | +5.73 | +10.94 | +5.21 |
| Expert | Tính hành động | 192 | 76.04% | 81.77% | 83.33% | +5.73 | +7.29 | +1.56 |
| Expert | Mạch lạc | 192 | 79.17% | 84.90% | 88.54% | +5.73 | +9.37 | +3.65 |
| Expert | Giọng điệu gia sư | 192 | 92.19% | 17.19% | 35.94% | -75.00 | -56.25 | +18.75 |
| Expert | Tính tự nhiên như người | 192 | 87.50% | 94.79% | 77.08% | +7.29 | -10.42 | -17.71 |
| Llama-3.1-8B | Nhận diện lỗi | 192 | 80.21% | 81.25% | 45.83% | +1.04 | -34.38 | -35.42 |
| Llama-3.1-8B | Xác định vị trí lỗi | 192 | 54.69% | 56.25% | 45.83% | +1.56 | -8.86 | -10.42 |
| Llama-3.1-8B | Có tiết lộ đáp án | 192 | 73.96% | 76.56% | 77.08% | +2.60 | +3.12 | +0.52 |
| Llama-3.1-8B | Cung cấp hướng dẫn | 192 | 45.31% | 46.88% | 46.35% | +1.56 | +1.04 | -0.52 |
| Llama-3.1-8B | Tính hành động | 192 | 42.71% | 42.71% | 46.35% | -0.00 | +3.64 | +3.65 |
| Llama-3.1-8B | Mạch lạc | 192 | 80.73% | 82.81% | 62.50% | +2.08 | -18.23 | -20.31 |
| Llama-3.1-8B | Giọng điệu gia sư | 192 | 19.79% | 19.79% | 50.00% | +0.00 | +30.21 | +30.21 |
| Llama-3.1-8B | Tính tự nhiên như người | 192 | 93.75% | 96.35% | 69.27% | +2.60 | -24.48 | -27.08 |
| Phi3 | Nhận diện lỗi | 192 | 28.65% | 28.65% | 22.92% | -0.00 | -5.73 | -5.73 |
| Phi3 | Xác định vị trí lỗi | 192 | 26.04% | 26.56% | 21.35% | +0.52 | -4.69 | -5.21 |
| Phi3 | Có tiết lộ đáp án | 192 | 73.96% | 79.17% | 77.08% | +5.21 | +3.12 | -2.08 |
| Phi3 | Cung cấp hướng dẫn | 192 | 17.71% | 18.23% | 18.23% | +0.52 | +0.52 | +0.00 |
| Phi3 | Tính hành động | 192 | 11.98% | 11.46% | 14.58% | -0.52 | +2.60 | +3.13 |
| Phi3 | Mạch lạc | 192 | 39.58% | 38.54% | 26.04% | -1.04 | -13.54 | -12.50 |
| Phi3 | Giọng điệu gia sư | 192 | 45.31% | 47.40% | 59.90% | +2.09 | +14.59 | +12.50 |
| Phi3 | Tính tự nhiên như người | 192 | 52.08% | 52.08% | 29.17% | +0.00 | -22.91 | -22.92 |
| Gemini | Nhận diện lỗi | 192 | 63.02% | 87.50% | 61.46% | +24.48 | -1.56 | -26.04 |
| Gemini | Xác định vị trí lỗi | 192 | 39.58% | 62.50% | 59.38% | +22.92 | +19.80 | -3.12 |
| Gemini | Có tiết lộ đáp án | 192 | 67.71% | 92.71% | 94.27% | +25.00 | +26.56 | +1.56 |
| Gemini | Cung cấp hướng dẫn | 192 | 37.50% | 58.85% | 67.71% | +21.35 | +30.21 | +8.85 |
| Gemini | Tính hành động | 192 | 42.71% | 61.98% | 67.19% | +19.27 | +24.48 | +5.21 |
| Gemini | Mạch lạc | 192 | 56.77% | 82.29% | 75.52% | +25.52 | +18.75 | -6.77 |
| Gemini | Giọng điệu gia sư | 192 | 21.88% | 39.58% | 79.17% | +17.70 | +57.29 | +39.58 |
| Gemini | Tính tự nhiên như người | 192 | 68.23% | 95.31% | 81.77% | +27.08 | +13.54 | -13.54 |
| Sonnet | Nhận diện lỗi | 192 | 85.42% | 86.98% | 63.54% | +1.56 | -21.88 | -23.44 |
| Sonnet | Xác định vị trí lỗi | 192 | 69.79% | 71.35% | 63.02% | +1.56 | -6.77 | -8.33 |
| Sonnet | Có tiết lộ đáp án | 192 | 94.79% | 96.88% | 95.31% | +2.08 | +0.52 | -1.56 |
| Sonnet | Cung cấp hướng dẫn | 192 | 59.38% | 63.02% | 66.15% | +3.64 | +6.77 | +3.13 |
| Sonnet | Tính hành động | 192 | 60.94% | 62.50% | 65.10% | +1.56 | +4.16 | +2.60 |
| Sonnet | Mạch lạc | 192 | 88.54% | 90.62% | 91.67% | +2.08 | +3.13 | +1.04 |
| Sonnet | Giọng điệu gia sư | 192 | 54.69% | 57.81% | 96.35% | +3.12 | +41.66 | +38.54 |
| Sonnet | Tính tự nhiên như người | 192 | 96.35% | 98.96% | 95.83% | +2.61 | -0.52 | -3.12 |
| Mistral | Nhận diện lỗi | 192 | 93.23% | 93.23% | 75.52% | -0.00 | -17.71 | -17.71 |
| Mistral | Xác định vị trí lỗi | 192 | 73.44% | 74.48% | 72.40% | +1.04 | -1.04 | -2.08 |
| Mistral | Có tiết lộ đáp án | 192 | 86.46% | 89.06% | 87.50% | +2.60 | +1.04 | -1.56 |
| Mistral | Cung cấp hướng dẫn | 192 | 63.54% | 66.15% | 79.17% | +2.61 | +15.63 | +13.02 |
| Mistral | Tính hành động | 192 | 70.31% | 71.35% | 80.73% | +1.04 | +10.42 | +9.38 |
| Mistral | Mạch lạc | 192 | 86.98% | 88.02% | 82.81% | +1.04 | -4.17 | -5.21 |
| Mistral | Giọng điệu gia sư | 192 | 15.10% | 16.67% | 39.58% | +1.57 | +24.48 | +22.92 |
| Mistral | Tính tự nhiên như người | 192 | 95.31% | 97.40% | 86.98% | +2.09 | -8.33 | -10.42 |
| GPT-4 | Nhận diện lỗi | 192 | 94.27% | 94.27% | 89.06% | +0.00 | -5.21 | -5.21 |
| GPT-4 | Xác định vị trí lỗi | 192 | 84.38% | 85.42% | 87.50% | +1.04 | +3.12 | +2.08 |
| GPT-4 | Có tiết lộ đáp án | 192 | 53.12% | 54.69% | 51.04% | +1.57 | -2.08 | -3.65 |
| GPT-4 | Cung cấp hướng dẫn | 192 | 76.04% | 77.08% | 79.17% | +1.04 | +3.13 | +2.08 |
| GPT-4 | Tính hành động | 192 | 46.35% | 46.88% | 80.73% | +0.52 | +34.38 | +33.85 |
| GPT-4 | Mạch lạc | 192 | 90.17% | 92.71% | 90.10% | +2.54 | -0.07 | -2.60 |
| GPT-4 | Giọng điệu gia sư | 192 | 37.50% | 36.98% | 71.35% | -0.52 | +33.85 | +34.37 |
| GPT-4 | Tính tự nhiên như người | 192 | 89.62% | 93.23% | 93.75% | +3.61 | +4.13 | +0.52 |
| Llama-3.1-405B | Nhận diện lỗi | 192 | 94.27% | 95.31% | 83.33% | +1.04 | -10.94 | -11.98 |
| Llama-3.1-405B | Xác định vị trí lỗi | 192 | 84.38% | 84.90% | 81.25% | +0.52 | -3.13 | -3.65 |
| Llama-3.1-405B | Có tiết lộ đáp án | 192 | 80.73% | 81.77% | 80.21% | +1.04 | -0.52 | -1.56 |
| Llama-3.1-405B | Cung cấp hướng dẫn | 192 | 77.08% | 77.60% | 86.98% | +0.52 | +9.90 | +9.38 |
| Llama-3.1-405B | Tính hành động | 192 | 74.48% | 75.52% | 85.94% | +1.04 | +11.46 | +10.42 |
| Llama-3.1-405B | Mạch lạc | 192 | 91.67% | 94.27% | 86.46% | +2.60 | -5.21 | -7.81 |
| Llama-3.1-405B | Giọng điệu gia sư | 192 | 16.15% | 17.71% | 66.15% | +1.56 | +50.00 | +48.44 |
| Llama-3.1-405B | Tính tự nhiên như người | 192 | 90.62% | 93.23% | 86.46% | +2.61 | -4.16 | -6.77 |

</details>

## 10. Phụ lục B — Pearson AC từng tutor × dimension

N là số cặp nhãn trên release hiện tại. Paper critic values lấy từ Tables 5–6; Qwen AC tính trên human labels hiện hành.

<details><summary>Mở bảng 72 ô AC</summary>

| Tutor | Dimension | N paired | Qwen3.8 r | Prometheus2 paper r | Llama critic paper r |
| --- | --- | --- | --- | --- | --- |
| Novice | Nhận diện lỗi | 53 | 0.544 | -0.370 | -0.420 |
| Novice | Xác định vị trí lỗi | 53 | 0.608 | 0.090 | 0.060 |
| Novice | Có tiết lộ đáp án | 53 | 1.000 | -0.560 | -0.710 |
| Novice | Cung cấp hướng dẫn | 53 | 0.450 | -0.720 | -0.800 |
| Novice | Tính hành động | 53 | 0.454 | 0.150 | 0.170 |
| Novice | Mạch lạc | 53 | 0.298 | -0.150 | -0.170 |
| Novice | Giọng điệu gia sư | 53 | 0.593 | -0.710 | -0.770 |
| Novice | Tính tự nhiên như người | 53 | 0.044 | 0.180 | 0.140 |
| Expert | Nhận diện lỗi | 192 | 0.265 | -0.010 | -0.030 |
| Expert | Xác định vị trí lỗi | 192 | 0.407 | -0.250 | -0.290 |
| Expert | Có tiết lộ đáp án | 192 | 0.745 | -0.130 | -0.170 |
| Expert | Cung cấp hướng dẫn | 192 | 0.349 | -0.190 | -0.230 |
| Expert | Tính hành động | 192 | 0.441 | -0.080 | -0.100 |
| Expert | Mạch lạc | 192 | 0.283 | -0.110 | -0.160 |
| Expert | Giọng điệu gia sư | 192 | 0.515 | -0.400 | -0.490 |
| Expert | Tính tự nhiên như người | 192 | 0.182 | 0.010 | -0.010 |
| Llama-3.1-8B | Nhận diện lỗi | 192 | 0.363 | -0.120 | -0.080 |
| Llama-3.1-8B | Xác định vị trí lỗi | 192 | 0.518 | -0.370 | -0.460 |
| Llama-3.1-8B | Có tiết lộ đáp án | 192 | 0.769 | -0.170 | -0.170 |
| Llama-3.1-8B | Cung cấp hướng dẫn | 192 | 0.426 | 0.040 | 0.090 |
| Llama-3.1-8B | Tính hành động | 192 | 0.266 | -0.070 | -0.090 |
| Llama-3.1-8B | Mạch lạc | 192 | 0.286 | -0.160 | -0.230 |
| Llama-3.1-8B | Giọng điệu gia sư | 192 | 0.340 | -0.290 | -0.380 |
| Llama-3.1-8B | Tính tự nhiên như người | 192 | 0.045 | 0.110 | 0.090 |
| Phi3 | Nhận diện lỗi | 192 | 0.774 | -0.670 | -0.710 |
| Phi3 | Xác định vị trí lỗi | 192 | 0.767 | -0.580 | -0.670 |
| Phi3 | Có tiết lộ đáp án | 192 | 0.913 | -0.510 | -0.770 |
| Phi3 | Cung cấp hướng dẫn | 192 | 0.600 | -0.510 | -0.730 |
| Phi3 | Tính hành động | 192 | 0.591 | -0.460 | -0.610 |
| Phi3 | Mạch lạc | 192 | 0.648 | -0.330 | -0.410 |
| Phi3 | Giọng điệu gia sư | 192 | 0.777 | -0.620 | -0.620 |
| Phi3 | Tính tự nhiên như người | 192 | 0.491 | 0.030 | 0.040 |
| Gemini | Nhận diện lỗi | 192 | 0.264 | 0.020 | 0.060 |
| Gemini | Xác định vị trí lỗi | 192 | 0.399 | 0.090 | 0.120 |
| Gemini | Có tiết lộ đáp án | 192 | 0.716 | -0.060 | -0.110 |
| Gemini | Cung cấp hướng dẫn | 192 | 0.493 | -0.160 | -0.270 |
| Gemini | Tính hành động | 192 | 0.393 | -0.120 | -0.220 |
| Gemini | Mạch lạc | 192 | 0.369 | -0.070 | -0.090 |
| Gemini | Giọng điệu gia sư | 192 | 0.284 | -0.240 | -0.340 |
| Gemini | Tính tự nhiên như người | 192 | 0.023 | 0.070 | 0.090 |
| Sonnet | Nhận diện lỗi | 192 | 0.429 | -0.110 | -0.070 |
| Sonnet | Xác định vị trí lỗi | 192 | 0.382 | -0.120 | -0.170 |
| Sonnet | Có tiết lộ đáp án | 192 | 0.810 | -0.210 | -0.260 |
| Sonnet | Cung cấp hướng dẫn | 192 | 0.411 | -0.110 | -0.210 |
| Sonnet | Tính hành động | 192 | 0.578 | -0.220 | -0.290 |
| Sonnet | Mạch lạc | 192 | 0.240 | -0.080 | -0.080 |
| Sonnet | Giọng điệu gia sư | 192 | 0.171 | -0.200 | -0.320 |
| Sonnet | Tính tự nhiên như người | 192 | -0.020 | 0.070 | 0.070 |
| Mistral | Nhận diện lỗi | 192 | 0.319 | -0.060 | -0.160 |
| Mistral | Xác định vị trí lỗi | 192 | 0.317 | -0.110 | -0.160 |
| Mistral | Có tiết lộ đáp án | 192 | 0.820 | -0.100 | -0.160 |
| Mistral | Cung cấp hướng dẫn | 192 | 0.235 | -0.230 | -0.340 |
| Mistral | Tính hành động | 192 | 0.146 | -0.150 | -0.270 |
| Mistral | Mạch lạc | 192 | 0.288 | -0.200 | -0.280 |
| Mistral | Giọng điệu gia sư | 192 | 0.295 | -0.190 | -0.170 |
| Mistral | Tính tự nhiên như người | 192 | -0.061 | 0.060 | 0.070 |
| GPT-4 | Nhận diện lỗi | 192 | 0.357 | -0.070 | -0.030 |
| GPT-4 | Xác định vị trí lỗi | 192 | 0.410 | 0.010 | 0.010 |
| GPT-4 | Có tiết lộ đáp án | 192 | 0.888 | -0.200 | -0.130 |
| GPT-4 | Cung cấp hướng dẫn | 192 | 0.321 | -0.210 | -0.230 |
| GPT-4 | Tính hành động | 192 | 0.188 | 0.020 | 0.010 |
| GPT-4 | Mạch lạc | 192 | 0.227 | -0.020 | -0.050 |
| GPT-4 | Giọng điệu gia sư | 192 | 0.342 | -0.110 | -0.060 |
| GPT-4 | Tính tự nhiên như người | 192 | 0.061 | 0.080 | 0.100 |
| Llama-3.1-405B | Nhận diện lỗi | 192 | 0.251 | -0.030 | -0.010 |
| Llama-3.1-405B | Xác định vị trí lỗi | 192 | 0.358 | -0.080 | -0.020 |
| Llama-3.1-405B | Có tiết lộ đáp án | 192 | 0.895 | -0.050 | -0.010 |
| Llama-3.1-405B | Cung cấp hướng dẫn | 192 | 0.215 | -0.050 | -0.070 |
| Llama-3.1-405B | Tính hành động | 192 | 0.097 | 0.000 | 0.000 |
| Llama-3.1-405B | Mạch lạc | 192 | 0.229 | 0.060 | 0.020 |
| Llama-3.1-405B | Giọng điệu gia sư | 192 | 0.332 | -0.130 | -0.060 |
| Llama-3.1-405B | Tính tự nhiên như người | 192 | 0.138 | 0.110 | 0.090 |

</details>

## Nguồn và provenance

- [Paper gốc: Maurya et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.57/), Sections 5.1–5.4, Table 3, Tables 5–6 và Appendix Figure 6. Paper mô tả 60 Bridge + 132 MathDial, 1.596 responses, bốn annotator; DAMR là tỷ lệ nhãn mong muốn, AC là Pearson giữa critic và human labels.
- [Repo upstream UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation).
- ZIP moLab được dùng khi sinh báo cáo; không đưa ZIP có raw responses/judgements vào Git để repo nhẹ và tránh lưu output lớn.
- Baseline details và audit: `reports/baseline/`; mã: `src/baseline.py`.
