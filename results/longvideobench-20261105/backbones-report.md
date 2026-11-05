# RIDGE σ2 so với nhiều mốc phụ đề — 64 frame

LongVideoBench validation, 1.337 câu / 753 video; cùng câu hỏi, model và ngân sách frame theo từng cặp. Hai lượt kết thúc 2.674 bản ghi, năm worker exit 0 theo supervisor. Kiểm ZIP CRC, 1.337 ID × 2 nhánh, shard theo video, frame khớp manifest, nhãn và không trùng.

| Model | RIDGE σ2 | Nhiều mốc | Sửa đúng | Làm sai | Δ pp | Bootstrap 95% theo video |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 815/1337 (60,96%) | 826/1337 (61,78%) | 32 | 21 | +0,82 | [-0,22; +1,87] |
| LLaVA-OneVision-7B | 820/1337 (61,33%) | 848/1337 (63,43%) | 44 | 16 | +2,09 | [+0,90; +3,28] |

10.000 bootstrap theo video, seed 20261102. Khác frame ở 481 câu. Validation đã xem trong phát triển; không phải test độc lập. Các lượt đã resume hai lần; hai pilot resume so 10 bản ghi nhưng không có lockfile đầy đủ runtime cũ. Chỉ tái kiểm hash pixel canonical của frame RIDGE, chưa replay toàn bộ selector và không đối chiếu độc lập pixel các frame mới. Qwen dùng max/min 224×224 pixel, không phải cấu hình mặc định 32 frame của paper. Không có thời gian end-to-end xác minh đầy đủ.
