# Kết quả kiểm tra ngày 03/10/2026

Chi tiết máy đọc được, input/code SHA-256 và regression matrix ở
[VALIDATION_RESULTS.json](VALIDATION_RESULTS.json). Source/paper provenance ở
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

- **39 unit/integration tests pass** trên NumPy 2.3.5.
- **48.444 query-budget cases**, 24 cấu hình = hai benchmark × ba scorer
  (BLIP/CLIP/SeViLA) × K=8/16/32/64, baseline frame indices khớp fixture source
  upstream: **0 mismatch**. Chạy default all_depth=5/t1=0.8/t2=-100 cho regression,
  kể cả các trường hợp K nhỏ bị quota làm tròn về 0.
- `compileall` pass cho source đã sửa, test và generated task utils.
- `bash -n scripts/evaluate_pair.sh` pass; `git diff --check` pass.
- Test adapter thực thi method `load_video_index` thật qua AST với decoder giả
  để xác nhận cờ strict đọc đúng IDs, timestamp và reject invalid IDs. Không có
  PyTorch, lmms-eval, video benchmark hoặc checkpoint trong môi trường này.

## Chọn frame trên toàn bộ cache BLIP, K=64

Cả hai selector dùng cùng scores/frame IDs/annotations, ratio=1, t1=0.8,
t2=-100, all_depth=5. Watershed dùng sigma=1, min_prominence=0.05,
min_distance=3 candidate ticks và prominence_weight=0.25.
Paired runner xác nhận **leaf plan và số frame thực tế từng query giống nhau**.

| Metric | VideoMME gốc | VideoMME + Watershed | LVB gốc | LVB + Watershed |
|---|---:|---:|---:|---:|
| Query | 2.700 | 2.700 | 1.337 | 1.337 |
| Query có ít hơn 64 frames | 258 | 258 | 361 | 361 |
| Cặp candidate liền kề | 61.840 | 34.079 | 23.925 | 8.168 |
| Cặp gap < 3 candidate ticks | 81.450 | 49.897 | 29.285 | 9.183 |
| Relevance chuẩn hóa trung bình của frames đã chọn | 0,5749 | 0,5210 | 0,4585 | 0,3649 |
| Temporal span fraction trung bình mỗi query | 0,9724 | 0,9916 | 0,9576 | 0,9961 |
| Lần refill phải nới khoảng cách | — | 24.569 | — | 744 |

VideoMME: 2.410 query thay tập frame; LVB: 976 query thay tập frame. Relevance là
mean theo **frame**, span là mean theo **query**. Span chỉ đo khoảng từ frame đầu
đến frame cuối, không đo đầy đủ các sự kiện bên trong.

Kết quả phù hợp mục tiêu giảm chọn lân cận, đồng thời cho thấy trade-off:
relevance giảm và nhiều frame được thêm bằng refill. Không thể suy ra tăng QA
accuracy từ các thống kê này. Thời gian selector có ghi trong JSON, nhưng các job
CPU chạy đồng thời nên chưa dùng làm benchmark hiệu năng có kiểm soát.

Artifacts tại workspace:

```text
experiments/validation/upstream-regression.json
experiments/videomme-blip-k64-final/{manifest,summary}.json
experiments/videomme-blip-k64-final/{original,watershed}/{selected_frames,diagnostics,annotations}.json
experiments/longvideobench-blip-k64-final/{manifest,summary}.json
experiments/longvideobench-blip-k64-final/{original,watershed}/{selected_frames,diagnostics,annotations}.json
```

Các tasks chuẩn bị trong phiên này trỏ đến đường dẫn Windows; chưa có video tại
video_root. Chạy lại `compare_selectors.py` với video_root thật trong môi trường
GPU/Linux trước khi dùng `scripts/evaluate_pair.sh`. Hướng dẫn và những điểm cần
khóa để so sánh công bằng ở [AKS_WATERSHED.md](AKS_WATERSHED.md).
