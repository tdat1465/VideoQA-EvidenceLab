# Protocol nghiên cứu

## Câu hỏi cần kiểm chứng

Với backbone đóng băng và pool 64 frame, lựa chọn bằng chứng phân biệt hai đáp án có cải thiện accuracy/chi phí so với uniform, CLIP top-k và AKS+CLIP trên NExT-QA/STAR không? Đây là giả thuyết; chưa có số đo để khẳng định hiệu quả hoặc tính mới đủ cho publication.

## Recipe cố định

- Backbone mặc định Molmo2-4B, SDPA, một GPU, không fine-tune. `dtype=auto` chọn native BF16 nếu có, ngược lại FP16; báo precision thực tế và giữ nguyên giữa các baseline/paired runs. Không audio, subtitles hay scene graph. Đây là zero-shot **ở repo này**; không tuyên bố checkpoint chưa từng thấy benchmark trong dữ liệu huấn luyện.
- Pool lấy mẫu theo thời gian trong các bin, tối đa 64 frame khác nhau. Video quá ngắn có thể ít frame hơn, mọi run phải ghi số thực tế.
- Prompt yêu cầu chữ cái. Chấm bằng logits A–E ở bước sinh đầu, softmax trên các lựa chọn hợp lệ. Đây là **xác suất tương đối giữa lựa chọn**, không phải xác suất câu trả lời đúng đã calibration. Không chain-of-thought hoặc text parsing.
- Molmo2 nhận video đã decode và timestamp thật qua metadata; không để processor tự lấy mẫu lại. Processor Molmo2 biểu diễn thời gian đến 0.1 giây; trace lưu timestamp gốc. Qwen tùy chọn nhận chuỗi ảnh có timestamp, cần được gọi là `multi-image` trong báo cáo. Các processor đều đặt `use_fast=False` tường minh.
- Evidence giữ 8 frame ban đầu. Khi margin < 0.15, lấy hai lựa chọn đầu; score frame = `abs(CLIP(q+a)-CLIP(q+b)) + 0.25*CLIP(q)`. Luân phiên anchor thiên về từng lựa chọn, thêm frame trước/sau trong cửa sổ 1 giây, tối đa 16 frame. Đây không phải phản chứng nhân quả và có thể bị confirmation bias.
- CLIP dùng text truncation tối đa độ dài tokenizer (77 tokens); câu dài có thể mất phần đuôi lựa chọn. Ghi hạn chế này khi phân tích lỗi, nghiên cứu query decomposition là bước mở rộng hợp lý.
- `uniform_refine` có cùng gate, số lần gọi và ngân sách 8→16, nhưng thêm khung đều. Nó không cần CLIP nên vẫn có chênh lệch chi phí selector; phải báo thời gian thực.
- AKS gọi hàm phân đoạn gốc ở commit đã khóa, CLIP cosine là nguồn relevance của repo. Config 16 frame dùng depth=4 để tránh quota `int(16/2**5)=0`. Không gọi đây là tái lập số trong paper AKS. Các fallback clip ngắn/score hằng/quota bằng 0 được ghi `aks_status`.

## Thứ tự thực nghiệm

1. Chạy 2 câu thật bằng uniform16 để kiểm tra processor/model, peak VRAM và token. Cấu hình 32GB là ngân sách mục tiêu, không phải số VRAM đã đo. Dùng `doctor` để xác nhận GPU thực được cấp.
2. Pilot 200 câu **validation**, cùng seed18: uniform8/16/32, CLIP16, AKS16, uniform_refine16, evidence16. Resume từng run riêng. Không chạy các model song song trên một GPU.
3. Phân tích accuracy theo C/T/D của NExT-QA và 4 loại STAR; tỷ lệ kích hoạt refinement, số khung, token, thời gian, peak VRAM. Kiểm tra các câu chuyển đúng→sai và sai→đúng, thiếu bằng chứng thời gian, cảnh nhanh, score CLIP không phân biệt đáp án.
4. Chọn và khóa hyperparameter trên phần dev validation **tách theo video**. Chạy held-out validation để ước lượng hiệu quả; repo không tự tạo split dev/heldout nên cần lưu hai manifest độc lập trước khi tune. Đừng tune trên toàn bộ val rồi gọi kết quả val là held-out.
5. Chạy split test sau khi khóa phương án, `limit=0` trong config mới. Với STAR test không có label, dùng cơ chế đánh giá chính thức; adapter submission là bước tiếp theo.

## Ablation có giá trị

- Ngưỡng margin 0/0.05/0.15/0.30/1 (0 gần tương đương không refinement, 1 gần luôn refinement).
- Uniform second pass so với evidence second pass; báo cả chi phí hai lần encode chứ không chỉ 16 frame cuối.
- Đặt `relevance_weight=0` để bỏ thành phần relevance; biến thể bỏ trước/sau cần thay selector và run mới.
- Pool 32/64/128 và ngân sách 8/16/32; giữ cấu hình pool giống nhau trong từng paired comparison.
- Backbone Qwen chỉ khi pilot cho thấy vừa VRAM; chuyển backbone phải so sánh lại uniform với chính backbone đó.

## Đánh giá và thời gian

`accuracy_completed` là accuracy trên các câu đã hoàn thành, không được ghi như accuracy toàn split nếu `complete=false`. Bootstrap lấy mẫu **video** có hoàn lại, giữ tất cả câu của mỗi video; báo delta (right−left), CI95%, số video/câu. CI bootstrap không thay thế nhiều seed, kiểm định hypothesis độc lập hoặc phân tích bias.

`elapsed_seconds` gồm kiểm tra/đọc video, CLIP và mọi lần gọi answerer, CUDA synchronize; không gồm tải weights, tạo môi trường hoặc khởi tạo model. Decoder cache một clip, CLIP cache embedding tối đa 32 pool, mỗi run có cache riêng. Cả chi phí setup và tốc độ phụ thuộc node cần lấy thêm từ log Slurm. Không hứa hoàn thành toàn split trong 48h khi chưa đo throughput trên GPU thật. Pilot 200 câu cho phép ước lượng: thời gian câu trung bình × số câu còn lại, cộng setup và dự phòng.

## Chưa triển khai

Grounding metric NextGQA, planner/GMM của A.I.R., token pruning LENS/FOCUS, fine-tuning, quantization, tự động nộp leaderboard và tự động tìm hyperparameter. Không gán tên các phương pháp đó cho selector đơn giản hiện tại.
