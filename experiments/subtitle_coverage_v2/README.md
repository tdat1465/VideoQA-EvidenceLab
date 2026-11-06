# Thí nghiệm thay frame phụ đề có bảo vệ độ phủ

Mở `notebooks/RIDGE_Subtitle_Coverage_Molab.py` trong Molab sau khi chép phiên bản
repo có thư mục này lên phiên làm việc. Sửa ô đường dẫn checkout nếu repo nằm ở
vị trí khác `/marimo/ridge-canonical/repo-fresh`.

## Chuẩn bị môi trường

Dùng một GPU CUDA và Python 3.10–3.12 với Torch/torchvision CUDA tương thích.
Trong terminal của môi trường Molab đang dùng, cài các dependency của repo.

```bash
cd /marimo/ridge-canonical/repo-fresh
python -m pip install -r requirements-longvideo-molmo.txt
python -m pip install -e .
```

Đặt `HF_TOKEN` bằng Secrets hoặc environment của Molab. Tài khoản cần được cấp
quyền đọc LongVideoBench. Worker tải video theo từng video từ revision dataset
đã ghim. Video và PNG nằm trong tmpfs `/dev/shm`, model dùng cache Hugging Face
của môi trường. Cần đủ RAM cho các kiểm tra dự trữ 6 GiB và cache pixel tối đa
12 GiB. Precision được ghi vào contract và phải giữ nguyên khi resume.

Cung cấp các file từ lần chạy trước.

- `lvb_val.json`, SHA-256
  `d32bf2e0cbbdb4fa2e99666a5a49b5a19bca2ceec69222c9a7d3aad567d1fa23`.
- `subtitle-full-v1/anchors.json` đầy đủ 1.337 câu. Có thể để trống ô này và cung
  cấp `subtitles.tar` để dựng lại các mốc bằng đúng hàm v1.
- Các predictions đã xuất trong `results/longvideobench-20261105` đi cùng repo.

Notebook có nút preflight, chạy GPU và đọc bảng kết quả. Pilot xử lý tối đa
20 câu chưa hoàn thành, ưu tiên câu có mốc phụ đề theo quy tắc xác định từ đầu.
Đây là mẫu kiểm tra kỹ thuật. Full tiếp tục cùng journal và đủ 1.337 câu.
Chọn full sau khi kiểm tra log, frame bị thay và chi phí của pilot.

## Quy tắc cố định của v2

Giữ nguyên truy vấn phụ đề, thứ tự và tối đa 16 frame đề xuất của multi-anchor
v1 để tách tác động của quy tắc thay frame. `question_type` vẫn được dùng như v1.

1. Lấy 16 vị trí trải đều trên candidate grid cũ, có cả hai đầu. Bảo vệ frame nền
   gần nhất với từng vị trí, cùng frame nền đầu/cuối. Nếu event coverage có đủ
   các mốc này thì chúng được bảo vệ nguyên vẹn.
2. Bảo vệ các frame nền đã trùng đề xuất phụ đề và mọi frame phụ đề vừa chèn.
3. Chia video thành 16 vùng thời gian bằng nhau, hoặc ít hơn khi ngân sách ngắn.
   Với mỗi frame mới, ưu tiên thay frame nền gần nhất trong cùng vùng và có thể
   bỏ mà vẫn giữ các frame bắt buộc. Nếu hòa khoảng cách, ưu tiên điểm BLIP thấp.
4. Nếu không có frame như vậy, chọn frame nền có điểm BLIP thấp nhất ở vùng
   vẫn còn frame khác sau khi thay. Nếu hòa điểm, ưu tiên frame có hàng xóm gần.
5. Khi hết lựa chọn an toàn, bỏ qua đề xuất đó. Ngân sách, các frame bắt buộc và
   mọi vùng vốn có frame đều được giữ. Video rất ngắn có thể không chèn frame.

Điểm quan trọng là BLIP-ITM question-only mới, đo trên tối đa 64 frame nền.
Đây là chi phí bổ sung của v2. Từng câu ghi `blip_seconds`, `blip_scored_frames`,
`selector_seconds`, `swaps`, `skipped_targets` và `mandatory`.

## Phép so sánh

Bốn nhánh dùng cùng Molmo2-4B, cấu hình prompt, precision và pixel trong phiên.
Frame của ba control lấy từ kết quả đã công bố. Worker kiểm tra chọn lại
multi-anchor v1 khớp file cũ trước khi suy luận. Tất cả đáp án được tính mới,
chỉ tái sử dụng các tập frame giống hệt trong cùng journal mới.

| Nhánh | Tập frame |
|---|---|
| `ridge_sigma2` | RIDGE tác giả trong run canonical |
| `eventcover_sigma2` | Event coverage trước bước phụ đề |
| `multi_anchor_v1` | Quy tắc phụ đề cũ, control chính |
| `protected_v2` | Cùng đề xuất phụ đề với quy tắc thay frame mới |

Đánh giá chính là `protected_v2` so với `multi_anchor_v1`. Đọc cả số câu sửa đúng,
làm sai và khoảng bootstrap theo video. Bootstrap 10.000 lượt, seed 20261102,
được tính khi đủ toàn bộ 1.337 câu. Validation này đã được xem trong phát triển.

Các video được decode tuần tự hai lần để kiểm pixel, cache PNG lossless và
lưu manifest SHA-256. Pilot, full và resume dùng cùng tập pixel theo video.
Contract ràng buộc annotation, nguồn frame, mốc phụ đề, code, model, runtime và
precision. SQLite commit từng nhánh và khóa run ngăn ghi đồng thời. Đổi code
hoặc môi trường cần thư mục kết quả mới khi contract không khớp.

Kết quả gồm `results.sqlite3`, `contract.json`, `summary.json`,
`predictions.jsonl`, `pixels/`, `scores/` và log notebook. Giữ cả thư mục khi
chuyển phiên Molab. Thời gian phiên là giới hạn mềm giữa các công đoạn. Mã 75
là dừng có tiến độ. SQLite là nguồn chính nếu phiên bị ngắt trước khi xuất file.

Có thể chạy cùng worker từ terminal.

```bash
python experiments/subtitle_coverage_v2/run.py \
  --annotation /marimo/ridge-canonical/subtitle-pilot/lvb_val.json \
  --anchors /marimo/ridge-canonical/subtitle-full-v1/anchors.json \
  --output /marimo/ridge-canonical/subtitle-protected-v2 \
  --max-new-questions 20 --session-minutes 120
```

Thêm `--preflight` để kiểm đầu vào trước khi tải model. Dùng
`--max-new-questions 0` để tiếp tục toàn bộ tập. Nếu dùng archive phụ đề, thay
`--anchors ...` bằng `--subtitles /path/to/subtitles.tar`.

## Kiểm tra local

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_subtitle_coverage_v2.py -v
marimo check notebooks/RIDGE_Subtitle_Coverage_Molab.py
```

Các test CPU kiểm tra độ phủ, frame đầu/cuối, proposal v1, chọn frame gần/điểm
thấp, video ngắn, output model không hợp lệ và commit/resume. Accuracy GPU của
v2 cần đo bằng run mới.
