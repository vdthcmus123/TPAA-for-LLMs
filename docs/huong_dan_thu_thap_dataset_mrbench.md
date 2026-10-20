# Hướng dẫn xây dựng dataset mới theo MRBench

Tài liệu này hướng dẫn nhóm tạo tối thiểu **250 mẫu mới cho mỗi thành viên**, theo đúng bài toán và taxonomy đánh giá của paper *Unifying AI Tutor Evaluation*. Để thu thập và chia quota nhất quán, dataset mới được phân tầng theo **bốn vùng chuyên đề Toán học** và các cấp học từ mầm non/tiểu học đến cao đẳng/đại học. Bốn vùng này là thiết kế tổ chức dữ liệu của nhóm, **không phải taxonomy tám chiều của paper**. Mục tiêu là tạo dữ liệu dùng được cho baseline NLP, sinh phản hồi tutor bằng LLM, gán nhãn tám chiều, tính DAMR và so sánh judge với nhãn người.

## 1. Bài toán và đơn vị đếm mẫu

Paper nghiên cứu bài toán **sửa lỗi hoặc giải tỏa nhầm lẫn toán học trong hội thoại gia sư**. Input là một đoạn hội thoại Tutor–Student đã dừng ngay sau khi học sinh mắc lỗi hoặc thể hiện chưa hiểu. Tutor cần đưa ra phản hồi tiếp theo. Paper xây dựng MRBench từ 60 hội thoại Bridge và 132 MathDial; mỗi ngữ cảnh có phản hồi của bảy LLM, phản hồi tutor chuyên gia và (chỉ trên Bridge) phản hồi tutor novice. Tổng paper báo cáo 1.596 phản hồi, có nhãn theo tám chiều. [Paper, mục 5.1–5.2](https://aclanthology.org/2025.naacl-long.57.pdf)

Để thống nhất cách tính chỉ tiêu của nhóm, hãy dùng định nghĩa chặt sau:

> **Một mẫu mới = một ngữ cảnh hội thoại toán mới, kết thúc bằng lỗi/nhầm lẫn của học sinh, kèm ít nhất một phản hồi tutor và nhãn cuối cùng cho cả tám chiều.**

Mỗi thành viên tạo **250 `context_id` duy nhất**. Nếu một ngữ cảnh có phản hồi của ba model, đó là **1 ngữ cảnh nhưng 3 bản ghi response**, không phải 3 ngữ cảnh mới. Báo cáo cả hai số: `unique_contexts` và `response_records`. Đây là cách tránh làm phồng số mẫu bằng cách chạy nhiều model trên cùng một câu hỏi.

Ví dụ nhóm bốn người:

| Chỉ tiêu | Số lượng tối thiểu |
| --- | ---: |
| Ngữ cảnh duy nhất mỗi người | 250 |
| Ngữ cảnh duy nhất toàn nhóm | 1.000 |
| Bản ghi response tối thiểu nếu mỗi ngữ cảnh có một tutor response | 1.000 |
| Nhãn chiều cần có nếu mọi response được gán đủ tám chiều | 8.000 |

Nếu giảng viên định nghĩa “mẫu” là response thay vì ngữ cảnh, quy tắc trên vẫn đáp ứng chỉ tiêu theo hướng bảo thủ: mỗi ngữ cảnh đều có ít nhất một response được gán nhãn. Trong báo cáo đồ án, ghi rõ hai đơn vị đếm này.

## 2. MRBench hiện có gì và nhóm còn phải tạo gì?

### Paper và release

Paper dùng Bridge và MathDial, chọn các đoạn hội thoại có lỗi/nhầm lẫn toán học ở lượt cuối. Với MathDial, họ cắt hội thoại tại thời điểm học sinh mắc lỗi và lấy phản hồi tutor kế tiếp làm phản hồi chuyên gia. Với Bridge, họ lọc các ví dụ chất lượng từ dữ liệu gốc. Paper cũng nói họ không đưa các ví dụ phụ thuộc vào ngữ cảnh hình ảnh vào benchmark. Sau đó họ tạo phản hồi LLM và gán nhãn phản hồi theo taxonomy. Paper cho biết bốn annotator được huấn luyện; 40/192 hội thoại được hai người gán độc lập để đo độ nhất quán, Cohen’s kappa trung bình 0,71. Pilot trước đó dùng Fleiss’ kappa 0,65 — đây là hai phép đo ở hai giai đoạn khác nhau. [Paper, mục 5.1–5.3](https://aclanthology.org/2025.naacl-long.57.pdf)

Repo upstream hiện mô tả V1 có 192 hội thoại/8 chiều, V2 có 200 hội thoại/8 chiều và V3 có 300 hội thoại nhưng các ví dụ bổ sung chỉ có bốn chiều. V2/V3 là tài nguyên tham khảo hoặc có thể dùng để kiểm tra trùng lặp; **không được tính các mẫu công khai đó là mẫu mới do nhóm thu thập**. V3 cũng không phù hợp làm nguồn thay thế nếu nhóm cần đủ tám nhãn cho DAMR. [README và schema upstream](https://github.com/kaushal0494/UnifyingAITutorEvaluation)

### Audit bản V1 trong workspace

Bản V1 nhóm đang dùng đã được audit tại [`reports/baseline/data_audit.json`](../reports/baseline/data_audit.json): 192 object nhưng có 188 `conversation_id` duy nhất; phân bố hiện tại là 139 MathDial và 53 Bridge; có 1.589 response đã gán nhãn. Paper báo cáo 132 MathDial, 60 Bridge và 1.596 response. Bốn ID Bridge bị lặp trong file, response của chúng khác nhau. Do đó, khi tạo dữ liệu mới:

- Không sao chép, đổi tên hoặc diễn đạt lại nhẹ các hội thoại/response từ MRBench V1/V2/V3 rồi tính là mẫu mới.
- Không dùng `Split` có sẵn của V1 làm split cho dataset mới. Bản hiện tại phân bố split theo nguồn không cân bằng; mọi response cùng ngữ cảnh phải ở cùng một split.
- Ghi chính xác nguồn là `TeamCreated` hoặc tên dataset nguồn thật. Không gán `Bridge`/`MathDial` nếu nhóm không lấy dữ liệu từ các nguồn đó.
- Có thể so khớp nội dung với các phiên bản MRBench và nguồn gốc trước khi chốt; lưu lại kết quả kiểm tra trùng.

## 3. Phạm vi và kế hoạch quota

Để mở rộng đến mọi cấp học mà vẫn giữ được phần so sánh gần với paper, nhóm ghi cấp học riêng cho từng context. MRBench gốc tập trung vào Bridge ở cấp tiểu học và MathDial ở cấp trung học cơ sở; vì vậy hãy báo cáo riêng kết quả trên context thuộc hai dải này khi so sánh trực tiếp với paper. Dữ liệu THPT/đại học là phần mở rộng của nhóm, không gộp mà không phân tầng. Nếu tạo hội thoại tiếng Việt, ghi `language=vi` và phân tích riêng với tiếng Anh vì “human-likeness” và “encouraging” phụ thuộc ngôn ngữ/văn hóa.

### Bốn vùng chuyên đề Toán học

Mỗi context có đúng **một `primary_domain`** để thống kê và chia quota; có thể ghi thêm `secondary_topics` nếu bài toán tích hợp nhiều mảng. Các ví dụ dưới đây là danh mục gợi ý, không bắt buộc phải đưa mọi nhánh đại học vào nếu nhóm không đủ người kiểm tra chuyên môn.

| Mã | Vùng chuyên đề | Các chuyên đề phù hợp theo cấp học |
| --- | --- | --- |
| `D1` | **Số học, phép toán và tư duy định lượng** | **Mầm non:** đếm, nhận diện số lượng, so sánh nhiều/ít. **Tiểu học:** giá trị hàng, bốn phép tính, phân số/thập phân cơ bản, ước lượng. **THCS:** số nguyên/hữu tỉ, tỉ số, tỉ lệ, phần trăm, ước/bội, lũy thừa/căn cơ bản. **THPT:** số thực/phức, lũy thừa–logarit, sai số/xấp xỉ, đồng dư và số học nâng cao. **CĐ/ĐH:** lý thuyết số nhập môn, số học mô-đun, biểu diễn số, phương pháp số và sai số tính toán. |
| `D2` | **Đại số, hàm số, giải tích và đại số tuyến tính** | **Mầm non:** quy luật hình/số đơn giản, tìm phần còn thiếu. **Tiểu học:** số chưa biết, quy luật, quan hệ “nhiều hơn/ít hơn”. **THCS:** biểu thức, phương trình/bất phương trình một ẩn, quan hệ tuyến tính cơ bản. **THPT:** đa thức, phương trình/hệ, bất đẳng thức, hàm số/đồ thị, dãy số, giới hạn/đạo hàm/tích phân nhập môn. **CĐ/ĐH:** giải tích một/nhiều biến, chuỗi, phương trình vi phân nhập môn, ma trận, vector, hệ tuyến tính và đại số tuyến tính. |
| `D3` | **Hình học, đo lường và lượng giác** | **Mầm non:** hình dạng, vị trí, so sánh độ dài/kích thước. **Tiểu học:** hình phẳng/khối, độ dài, thời gian, tiền tệ, chu vi/diện tích cơ bản. **THCS:** góc, tam giác/tứ giác, đồng dạng/bằng nhau, diện tích/thể tích, đơn vị và tỉ lệ bản đồ. **THPT:** hình học tọa độ, đường tròn/conic, lượng giác, vector và hình học không gian. **CĐ/ĐH:** hình học giải tích, vector/không gian, tọa độ nhiều chiều, biến đổi hình học và ứng dụng phù hợp môn học. |
| `D4` | **Thống kê, xác suất, toán rời rạc và mô hình hóa** | **Mầm non:** phân loại đồ vật, biểu đồ hình ảnh đơn giản, khả năng xảy ra. **Tiểu học:** đọc biểu đồ/bảng, trung bình đơn giản, đếm cách. **THCS:** trung bình/trung vị/mốt, xác suất cơ bản, nguyên lý đếm. **THPT:** xác suất có điều kiện, tổ hợp, biến ngẫu nhiên/phân phối nhập môn, suy luận dữ liệu và mô hình hóa. **CĐ/ĐH:** thống kê suy luận, kiểm định/ước lượng, hồi quy, xác suất nâng cao, logic/tập hợp, tổ hợp, đồ thị, thuật toán, tối ưu và mô hình toán ứng dụng. |

Khi một bài tích hợp nhiều vùng, chọn `primary_domain` theo **kiến thức chính mà học sinh mắc lỗi**; ghi vùng còn lại vào `secondary_topics`. Ví dụ, lỗi tính phân số trong bài toán hình học là D1 nếu lỗi chính là phép tính phân số, nhưng D3 nếu lỗi là chọn sai công thức diện tích. Dùng D2 cho đại số tuyến tính/vector spaces; dùng D3 cho vector hình học/lượng giác. Dùng D1 cho tính toán số học/lũy thừa; dùng D2 nếu trọng tâm là tính chất/hành vi hàm số hoặc giải phương trình.

### Quota mẫu gợi ý cho nhóm bốn người

Phương án dễ quản lý nhất là **mỗi thành viên phụ trách chính một vùng D1–D4 và tạo 250 context duy nhất trong vùng đó**. Bốn vùng vì vậy có 250 context/vùng, tổng cộng 1.000 context. Để bảo đảm có mẫu ở mọi cấp học, dùng một ma trận mục tiêu giống nhau cho từng vùng:

| Cấp học | `grade_band` | Mẫu mỗi vùng | Tổng bốn vùng |
| --- | --- | ---: | ---: |
| Mầm non | `pre_k` | 25 | 100 |
| Tiểu học | `primary` | 75 | 300 |
| THCS | `lower_secondary` | 75 | 300 |
| THPT | `upper_secondary` | 50 | 200 |
| Cao đẳng/đại học | `tertiary` | 25 | 100 |
| **Tổng** |  | **250** | **1.000** |

Đây là **quota khởi điểm**, có thể điều chỉnh sau pilot nếu một dải học quá khó thu thập hoặc khó xác minh. Trong quota tiểu học, nên rải mẫu qua lớp 1–5 (xấp xỉ 15 mẫu/lớp/vùng); THCS qua lớp 6–9 (xấp xỉ 18–19 mẫu/lớp/vùng); THPT qua lớp 10–12 (xấp xỉ 16–17 mẫu/lớp/vùng). Mầm non có thể ghi độ tuổi hoặc lớp mẫu giáo; cao đẳng/đại học ghi năm học và môn/cấp học phần thay vì ép thành “lớp”. Trong mỗi ô vùng × cấp học, chia tương đối giữa chuyên đề con và độ khó; không cần ép mọi chuyên đề thành số lượng bằng nhau. Nếu nhóm không có người đủ chuyên môn cho một chủ đề đại học nâng cao, thay bằng chủ đề đại học nhập môn có đáp án kiểm chứng được.

Ma trận này có 600/1.000 context ở tiểu học và THCS — gần nhất với độ tuổi trong Bridge/MathDial. Báo cáo riêng phân nhóm này để so sánh với paper; kết quả trên mầm non, THPT và CĐ/ĐH là phần mở rộng của nhóm.

Một response/context tạo 1.000 response records và 8.000 nhãn chiều ban đầu. Nếu sinh thêm nhiều response cho cùng context, số response records và workload gán nhãn tăng, nhưng quota 1.000 context không đổi. Không cân bằng bằng cách tạo nhiều biến thể gần giống của cùng bài toán; kiểm tra `problem_family_id` và trùng lặp trước khi tính quota.

## 4. Quy trình thu thập dữ liệu từng bước

### Bước 1 — Chốt đặc tả và tài liệu hướng dẫn

Nhóm tạo một trang đặc tả trước khi viết mẫu, tối thiểu ghi:

1. Đối tượng học sinh và dải lớp (`grade_band`).
2. Ngôn ngữ, một `primary_domain` trong D1–D4, `subtopic` và loại lỗi/nhầm lẫn dự kiến.
3. Định nghĩa mẫu và cách tính quota như mục 1.
4. Nguồn dữ liệu được phép dùng, giấy phép và cách ghi công.
5. Prompt cố định nếu dùng LLM sinh response.
6. Bộ nhãn, ví dụ chuẩn, quy trình xử lý bất đồng và phiên bản guideline.

Đọc [human annotation guideline của tác giả](https://github.com/kaushal0494/UnifyingAITutorEvaluation/tree/main/Human_Annotation_Guidelines_Resources) cùng Table 2 và phần 4 trong paper. Chạy pilot 20–30 ngữ cảnh trước khi sản xuất hàng loạt; sau pilot, sửa guideline, ghi `guideline_version`, rồi mới khóa quy tắc gán nhãn.

### Bước 2 — Chọn nguồn và ghi provenance

Ưu tiên theo thứ tự:

1. **Bài toán và ngữ cảnh do nhóm tự viết**: dễ kiểm soát giấy phép và tránh rò rỉ dữ liệu. Tạo đề toán mới, lời giải đúng, cách hiểu sai thường gặp, rồi viết vài lượt Tutor–Student tự nhiên.
2. **Dữ liệu mở có giấy phép phù hợp**: chỉ dùng sau khi kiểm tra giấy phép, điều kiện phân phối lại, quyền sử dụng nội dung, và lưu URL/phiên bản/tên tác giả. Nếu chỉ lấy đề bài rồi viết hội thoại mới, vẫn cần ghi nguồn đề bài và mức độ biến đổi.
3. **Hội thoại thu thập có đồng thuận**: chỉ dùng khi có quy trình đồng thuận, ẩn danh và quyền chia sẻ phù hợp; không đưa tên, trường, email, thông tin nhận dạng hay dữ liệu học sinh nhạy cảm vào repo.

Không scrape lời giải/diễn đàn có điều khoản cấm tái sử dụng. Không lấy một ví dụ MRBench rồi thay số, đổi tên hoặc paraphrase sơ sài. Ghi các trường `source_name`, `source_url`, `source_license`, `source_version`, `collection_method`, `created_by`, `created_at`, `is_original` cho từng ngữ cảnh. Với nội dung hoàn toàn do nhóm sáng tác, dùng `source_name=TeamCreated`, `source_license=team-original` và thống nhất giấy phép nhóm muốn công bố.

### Bước 3 — Tạo ngữ cảnh có lỗi/nhầm lẫn thật

Mỗi ngữ cảnh cần có:

- Một đề bài hoặc mục tiêu toán xác định được.
- Lời giải chuẩn (`ground_truth_solution`) được kiểm tra độc lập.
- Một hiểu lầm/lỗi cụ thể (`misconception_type`) hoặc dấu hiệu chưa hiểu.
- Một đoạn hội thoại trước phản hồi tutor, có đủ ngữ cảnh để người đọc hiểu cách học sinh đi đến lỗi.
- Lượt cuối của hội thoại là học sinh mắc lỗi hoặc thể hiện nhầm lẫn; tutor chưa sửa lỗi đó trong lịch sử.
- Một `primary_domain` trong D1–D4 và một `subtopic` cụ thể; thêm `secondary_topics` nếu có kiến thức liên quan từ vùng khác.

Ví dụ loại lỗi phù hợp: áp dụng sai thứ tự phép tính; cộng mẫu số khi cộng phân số; nhầm chu vi với diện tích; chia tỉ lệ sai; chuyển vế đổi dấu sai; diễn giải sai một bước trong đề lời văn. Đây là các **loại** lỗi, không phải nội dung copy từ benchmark.

Chất lượng cần đạt:

- Lỗi học sinh phải nhìn thấy được từ câu trả lời/lập luận, không chỉ do annotator đoán.
- Câu trả lời chuẩn có thể xác minh và phù hợp với đề; nếu có nhiều cách giải, lưu một cách giải chuẩn nhưng không coi cách khác là sai.
- Không để ngữ cảnh tự mâu thuẫn, thiếu dữ kiện quan trọng, hoặc phụ thuộc vào hình ảnh không được lưu kèm.
- Lịch sử hội thoại phải dừng đúng tại thời điểm cần tutor can thiệp; không để tutor đã tiết lộ lời giải trước đó.
- Tránh bài quá mơ hồ khiến nhiều người hợp lý có thể hiểu khác nhau. Mẫu không đạt thì sửa hoặc loại, không cố ép nhãn.

Người tạo mẫu tự kiểm tra phép toán; một thành viên khác giải độc lập và duyệt đề, lỗi dự kiến, lời giải và điểm cắt hội thoại. Ghi `math_verified_by` và `context_review_status`.

### Bước 4 — Tạo phản hồi tutor cần đánh giá

Chọn trước thiết kế thực nghiệm để không tăng workload ngoài dự tính:

- **Mức tối thiểu, phù hợp quota**: mỗi context có một phản hồi tutor được chọn trước (ví dụ phản hồi chuyên gia trong dữ liệu mở hoặc phản hồi do một LLM cố định tạo), sau đó gán nhãn đủ tám chiều.
- **So sánh nhiều model**: chạy chính các model cần so sánh trên cùng một tập context, dùng cùng format input và ghi model ID/revision, prompt, ngày chạy, temperature, sampling, max tokens. Mỗi model/context tạo một response record riêng; tổng số nhãn tăng theo `số_context × số_model × 8`.
- **Phương án tiết kiệm**: tạo đủ 1.000 context nhưng chọn một tập paired cố định 100–200 context để chạy tất cả model và gán nhãn so sánh. Báo cáo rõ đây là subset so sánh; không trình bày nó như thể mọi model đã chạy trên toàn bộ 1.000 context.

Không trộn phản hồi người với phản hồi LLM mà thiếu trường `tutor_type`. Đầu ra thô của model phải được lưu nguyên văn; không sửa ngữ pháp/đáp án trước khi gán nhãn. Nếu post-process, lưu cả raw và processed response cùng quy tắc xử lý. Không đưa API key/token vào dữ liệu hoặc notebook công khai.

### Bước 5 — Gán nhãn tám chiều

Mỗi response được đọc cùng **toàn bộ hội thoại và lời giải chuẩn**, không chỉ đọc riêng câu trả lời tutor. Paper yêu cầu các chiều được đánh giá độc lập, dù chúng có thể liên quan nhau. Dùng đúng tập nhãn tương thích MRBench:

| Chiều | Câu hỏi cần trả lời | Tập nhãn |
| --- | --- | --- |
| `Mistake_Identification` | Tutor có nhận ra lỗi/nhầm lẫn không? | `Yes` / `To some extent` / `No` |
| `Mistake_Location` | Tutor có chỉ đúng chỗ và bản chất lỗi không? | `Yes` / `To some extent` / `No` |
| `Revealing_of_the_Answer` | Tutor có đưa đáp án cuối cùng không? Nếu có, đáp án đúng hay sai? | `No` / `Yes (correct)` / `Yes (incorrect)` |
| `Providing_Guidance` | Tutor có gợi ý/giải thích đúng và liên quan không? | `Yes` / `To some extent` / `No` |
| `Actionability` | Học sinh có biết bước tiếp theo cần làm gì không? | `Yes` / `To some extent` / `No` |
| `Coherence` | Response có hợp logic với hội thoại và câu trả lời học sinh không? | `Yes` / `To some extent` / `No` |
| `Tutor_Tone` | Giọng điệu khích lệ, trung tính hay xúc phạm? | `Encouraging` / `Neutral` / `Offensive` |
| `Humanlikeness` | Câu trả lời có tự nhiên như người tutor hay máy móc/nhân tạo? | `Yes` / `To some extent` / `No` |

Các nhãn mong muốn dùng để tính **DAMR** theo Table 2 là: Identification=`Yes`; Location=`Yes`; Revealing=`No`; Guidance=`Yes`; Actionability=`Yes`; Coherence=`Yes`; Tone=`Encouraging`; Humanlikeness=`Yes`. Đây là “desired label”, không phải nhãn vàng mặc định cho mọi response. Người gán nhãn phải đánh giá response thực tế; không gán `Yes` chỉ vì tutor có ý tốt. [Paper, Table 2 và mục 5.4](https://aclanthology.org/2025.naacl-long.57.pdf)

Đối với `Revealing_of_the_Answer`, “No” nghĩa là chưa nêu đáp án cuối cùng, không có nghĩa response không hữu ích. Nếu đã nêu đáp án thì chọn đúng/sai theo lời giải toán. Không coi “đưa một gợi ý/giải thích trung gian” là tiết lộ đáp án cuối cùng nếu rubric không định nghĩa như vậy. Đối với `Providing_Guidance`, tính đúng đắn toán học và mức liên quan là trọng tâm; đừng dùng giọng khích lệ để bù cho hướng dẫn sai. Với `Actionability`, hỏi cụ thể học sinh có biết việc gì cần thử/làm tiếp theo hay không.

Khuyến nghị gán nhãn:

1. Tạo form có `context_id`, `response_id`, history, lời giải chuẩn, response, tám trường label, một ô ghi bằng chứng ngắn/ghi chú và `confidence`.
2. Ẩn danh model/tutor trong lúc gán nhãn nếu có thể, để giảm thiên lệch kỳ vọng.
3. LLM có thể gợi ý nhãn và trích bằng chứng, nhưng lưu ở cột riêng. Annotator phải tự đọc, xác minh và chọn nhãn cuối; không xem gợi ý là gold.
4. Mọi ví dụ thí điểm và khoảng 20% context được hai người gán độc lập. Khi một context có nhiều response, lấy mẫu trùng lặp ở cấp context và cho hai người chấm các response được chọn trong context đó; ghi rõ quy tắc.
5. Tính agreement theo từng chiều trên nhãn độc lập trước adjudication. Dùng Cohen’s kappa nếu có hai annotator; báo cáo thêm raw agreement và số mẫu. Không chỉ báo một con số trung bình nếu một chiều rất thấp.
6. Hai người thảo luận mọi bất đồng trong tập double-label; người thứ ba phân xử khi không thống nhất. Giữ `label_a`, `label_b`, `final_label`, `adjudicator_id`, `adjudication_note`; không ghi đè mất nhãn ban đầu.

Paper dùng kappa 0,71 trung bình cho đánh giá cuối và có 40/192 hội thoại được gán độc lập bởi hai annotator. Nhóm có thể lấy tỷ lệ khoảng 20% làm mốc thiết kế, nhưng cần báo cáo cách lấy mẫu và unit của mình, không tuyên bố tái lập chính xác quy trình/kết quả paper nếu khác. [Paper, mục 5.2](https://aclanthology.org/2025.naacl-long.57.pdf)

### Nhãn người dùng vào baseline và metric như thế nào?

- **Baseline phân loại** (Naive Bayes, Random Forest + Word2Vec, BERT): dùng response text/context làm input và từng chiều nhãn người làm target, theo cùng quy tắc tiền xử lý/split đã định trước. Báo cáo accuracy, macro-F1 và confusion matrix theo từng chiều; các số này không thay DAMR/AC.
- **DAMR**: tính trên nhãn cuối đã được người gán nhãn/adjudicate, bằng tỷ lệ response đạt nhãn mong muốn của từng chiều. Đừng dùng nhãn Qwen làm “gold DAMR” nếu mục tiêu là so với đánh giá người.
- **AC**: chạy Qwen judge trên đúng các response đã được người gán nhãn, ghép cặp theo `response_id` và so nhãn judge với nhãn người theo quy trình/encoding trong paper. Lưu nhãn judge ở bảng/cột riêng. Nếu không có nhãn người trên cùng response, không thể tính AC với người cho response đó.

Vì paper sử dụng nhãn người làm chuẩn để đánh giá judge, giữ riêng `human_final_label` và `qwen_judge_label` là điều kiện cần cho so sánh. Khi chạy judge trên toàn bộ dữ liệu, AC cần được tính trên chính tập giao có đủ cả hai loại nhãn; báo cáo số mẫu hợp lệ theo từng chiều.

### Bước 6 — Kiểm tra trùng lặp, chất lượng và split

Trước khi chốt dataset:

- Gán ID ổn định: `NEW-000001`, `NEW-000002`, …; ID response ví dụ `NEW-000001::Qwen3.5-9B`.
- Tìm trùng chính xác sau khi chuẩn hóa khoảng trắng/chữ hoa; dùng similarity/fuzzy match cho đề bài, history và response để phát hiện paraphrase quá gần. Mỗi cặp gần trùng phải có người quyết định giữ/loại và ghi log.
- Kiểm tra đủ context, solution, response, topic, source/license, tám nhãn; kiểm tra nhãn nằm trong tập cho phép; kiểm tra history kết thúc bằng lượt Student và có lỗi/nhầm lẫn; kiểm tra response không rỗng.
- Một người giải lại ngẫu nhiên một phần mẫu và kiểm tra toàn bộ mẫu có flagged lỗi toán.
- Kiểm tra độ phủ chủ đề/cấp học/độ khó, ngôn ngữ, nguồn, tutor/model; báo cáo missingness và số mẫu bị loại theo từng lý do.
- Nếu chia train/dev/test, chia theo **context/problem/source group**, không chia ngẫu nhiên theo response row. Tất cả tutor responses của một context phải cùng split. Nếu cùng một bài toán được paraphrase hoặc biến thể số, gom cùng `problem_family_id` và giữ trong một split để tránh leakage.
- Nếu chỉ dùng dataset mới làm test set ngoài miền, không cần huấn luyện trên nó; vẫn cần khóa danh sách mẫu và không chỉnh nhãn sau khi xem dự đoán của model trừ khi quy trình adjudication được ghi lại.

## 5. Schema khuyến nghị

Giữ cấu trúc gốc của tác giả làm tham chiếu tương thích: `conversation_id`, `conversation_history`, `Data`, `Split`, `Topic`, `Ground_Truth_Solution`, `anno_llm_responses` và tám nhãn. [Schema upstream](https://github.com/kaushal0494/UnifyingAITutorEvaluation)

Để nhóm làm việc bằng spreadsheet mà vẫn hỗ trợ nhiều model trên một context, lưu thành hai CSV có khóa liên kết:

### `new_contexts.csv` — một dòng cho mỗi context

| Cột | Ý nghĩa |
| --- | --- |
| `context_id` | ID duy nhất, ví dụ `NEW-000001` |
| `problem_id`, `problem_family_id` | Đề và họ bài toán/biến thể |
| `problem_text` | Đề toán gốc |
| `turns_json` | Các lượt có role `Tutor`/`Student`, giữ cấu trúc chuẩn |
| `conversation_history` | Bản text render từ turns, tương thích MRBench |
| `ground_truth_solution` | Lời giải kiểm tra được |
| `misconception_type` | Lỗi/nhầm lẫn mục tiêu |
| `primary_domain` | Một trong `D1`, `D2`, `D3`, `D4`; dùng phân tầng/quota |
| `subtopic`, `secondary_topics` | Chuyên đề chính và các chủ đề phụ nếu bài toán tích hợp |
| `grade_band`, `grade_level`, `difficulty`, `language` | Cấp học chuẩn hóa (`pre_k`, `primary`, `lower_secondary`, `upper_secondary`, `tertiary`), lớp/năm/môn cụ thể và mô tả phạm vi |
| `source_name`, `source_url`, `source_license`, `source_version` | Provenance/quyền sử dụng |
| `is_original`, `created_by`, `math_verified_by`, `context_review_status` | Kiểm soát quy trình |
| `split_group` | Nhóm dùng để chia split, nếu có |

### `new_responses.csv` — một dòng cho mỗi response

| Cột | Ý nghĩa |
| --- | --- |
| `response_id`, `context_id` | ID response và khóa sang context |
| `tutor_name`, `tutor_type` | Model/người và loại tutor |
| `response_raw` | Output chưa chỉnh sửa |
| `model_revision`, `prompt_version`, `generation_parameters` | Thông tin tái lập nếu response do LLM tạo |
| Tám cột nhãn | Nhãn đã adjudicate cuối cùng |
| `annotator_1`, `annotator_2`, `adjudicator_id` | Người gán nhãn/phân xử |
| `annotation_method`, `guideline_version`, `review_status` | Phương pháp và trạng thái nhãn |
| `evidence_notes` | Bằng chứng/ghi chú ngắn cho quyết định |

Nếu muốn xuất theo đúng JSON nested của MRBench, có thể tạo file chuyển đổi ở cuối pipeline từ hai CSV. Đừng dùng `Data` để chỉ nguồn mới; thêm `Data="TeamCreated"` chỉ khi cần tương thích kỹ thuật, đồng thời giữ `source_name` chính xác.

Ví dụ rút gọn (nội dung minh họa do nhóm tự tạo, không lấy từ benchmark):

```json
{
  "context_id": "NEW-000001",
  "problem_text": "A student is adding two fractions with different denominators.",
  "conversation_history": "Tutor: How would you add these fractions?\nStudent: I add the numerators and denominators directly.",
  "ground_truth_solution": "Rewrite the fractions with a common denominator, then add the numerators.",
  "misconception_type": "Adds unlike denominators directly",
  "primary_domain": "D1",
  "subtopic": "fractions",
  "grade_band": "lower_secondary",
  "grade_level": "Grade 7",
  "difficulty": "medium",
  "source_name": "TeamCreated",
  "source_license": "team-original",
  "responses": [
    {
      "response_id": "NEW-000001::TutorA",
      "tutor_name": "TutorA",
      "response_raw": "You added both parts directly. What denominator could both fractions share?",
      "annotation": {
        "Mistake_Identification": "Yes",
        "Mistake_Location": "Yes",
        "Revealing_of_the_Answer": "No",
        "Providing_Guidance": "Yes",
        "Actionability": "Yes",
        "Coherence": "Yes",
        "Tutor_Tone": "Encouraging",
        "Humanlikeness": "Yes"
      },
      "guideline_version": "v1.0",
      "review_status": "adjudicated"
    }
  ]
}
```

## 6. Ước tính workload và cách chia việc

Với bốn thành viên và 1.000 context:

- Mỗi người soạn 250 context, kiểm tra đề/lời giải của người khác theo vòng luân phiên.
- Một response/context nghĩa là 1.000 response × 8 chiều = 8.000 quyết định nhãn ban đầu.
- Nếu double-label 20% context: thêm 200 × 8 = 1.600 quyết định độc lập, tổng ít nhất 9.600 nhãn trước adjudication.
- Nếu chạy ba model trên tất cả 1.000 context: 3.000 response × 8 = 24.000 quyết định nhãn ban đầu, chưa kể double-label. Đây là workload lớn hơn ba lần phương án một response/context.

Gợi ý vòng việc cho nhóm bốn người:

1. **Đợt 1 — chốt thiết kế chung**: thống nhất rubric, source log, schema, prompt, ma trận quota và pilot 20–30 mẫu; kiểm tra rằng cả bốn người hiểu giống nhau về bốn vùng và nhãn.
2. **Đợt 2 — phân vùng có chủ sở hữu**: thành viên A phụ trách D1, B phụ trách D2, C phụ trách D3, D phụ trách D4. Mỗi người tạo 250 context trong vùng mình, theo quota cấp học 50/75/75/50 ở trên. Tạo theo batch 25–50 để review sớm.
3. **Đợt 3 — review chéo nội dung**: reviewer khác kiểm tra đề, lời giải, lỗi và grade-band; luân phiên theo vòng A→B→C→D→A. Nếu một batch sai quy tắc, sửa trước khi người phụ trách sinh thêm batch.
4. **Đợt 4 — sinh response**: khóa model/prompt/decoding; chạy một response chính trên đủ context. Nếu cần benchmark nhiều model, chạy cùng model set trên paired subset đã phân tầng theo D1–D4 và cấp học.
5. **Đợt 5 — gán nhãn và adjudication**: người gán nhãn không phải người duy nhất tạo context; ẩn model ID nếu có thể; double-label khoảng 20% context; người thứ ba phân xử bất đồng.
6. **Đợt 6 — khóa dataset**: chạy QA tự động, tính bảng phân bố theo vùng × cấp học, xuất dataset version `v1.0`, tạo checksum và dataset card.

Mỗi người có thể là người phụ trách nội dung của một vùng, nhưng không nên tự gán nhãn toàn bộ response của vùng mình và tự adjudicate chúng. Luân phiên reviewer giúp phát hiện lỗi sớm; trường `primary_domain` cho phép kiểm tra vùng nào chưa đủ quota hoặc đang có tỷ lệ lỗi chất lượng cao.

## 7. Báo cáo thống kê phải nộp

Khi hoàn tất, báo cáo tối thiểu:

- Số context duy nhất và response records; số lượng theo thành viên tạo.
- Số context/response bị loại và lý do.
- Phân bố nguồn, bốn vùng D1–D4, chuyên đề con, cấp học, độ khó, ngôn ngữ và tutor/model; lập bảng chéo vùng × cấp học.
- Tỷ lệ mẫu tự tạo và lấy từ nguồn mở; danh sách giấy phép/attribution.
- Missing values, duplicate exact/fuzzy, và cách xử lý.
- Phân bố nhãn ở từng trong tám chiều (số lượng và phần trăm).
- Quy trình gán nhãn, số annotator, tỷ lệ double-label, số bất đồng và adjudication.
- Agreement từng chiều trên nhãn độc lập trước adjudication (Cohen’s kappa/raw agreement và số mẫu).
- Prompt/model/revision/decoding nếu có LLM sinh phản hồi hoặc hỗ trợ nhãn.
- Hạn chế: mẫu thuận tiện hay phân tầng, phạm vi lớp/chủ đề, ngôn ngữ, nguồn, số người đánh giá.

## 8. Checklist trước khi đưa vào chạy baseline

- [ ] Đủ tối thiểu 250 context ID duy nhất cho mỗi thành viên.
- [ ] Đủ quota theo bốn vùng D1–D4; báo cáo số context duy nhất ở từng vùng.
- [ ] Mỗi vùng có mẫu ở mầm non, tiểu học, THCS, THPT và cao đẳng/đại học theo quota đã chốt.
- [ ] Mỗi mẫu có đề, history dừng ở lỗi/nhầm lẫn, lời giải chuẩn và provenance.
- [ ] Có ít nhất một tutor response và đủ tám nhãn cuối trên mỗi mẫu tính vào quota.
- [ ] Reviewer độc lập đã kiểm tra toán và tính hợp lệ của context.
- [ ] Tất cả model responses trên cùng context sẽ cùng một split.
- [ ] Không có bản sao/paraphrase gần của MRBench V1/V2/V3 hoặc dữ liệu không được phép tái sử dụng.
- [ ] Nhãn độc lập gốc và nhãn adjudicated được lưu tách biệt.
- [ ] Quy trình đếm context vs response được mô tả trong báo cáo.
- [ ] README/dataset card ghi nguồn, giấy phép, giới hạn và phiên bản.

## 9. Nguồn tham khảo

- Maurya et al. (2025), [Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors](https://aclanthology.org/2025.naacl-long.57/), NAACL 2025.
- [Repo MRBench chính thức](https://github.com/kaushal0494/UnifyingAITutorEvaluation), gồm schema, các phiên bản V1–V3 và tài liệu hướng dẫn gán nhãn.
- [Human Annotation Guidelines Resources](https://github.com/kaushal0494/UnifyingAITutorEvaluation/tree/main/Human_Annotation_Guidelines_Resources).

MRBench upstream ghi giấy phép CC BY-SA 4.0 và yêu cầu ghi công/chia sẻ bản phái sinh theo điều kiện giấy phép; kiểm tra trực tiếp giấy phép của từng nguồn Bridge/MathDial trước khi tái phân phối dữ liệu dẫn xuất. Dữ liệu do nhóm tự tạo cần có giấy phép riêng do nhóm thống nhất.
