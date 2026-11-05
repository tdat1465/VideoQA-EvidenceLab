# InternVL3-8B: RIDGE σ2 và nhiều mốc phụ đề

LongVideoBench validation, 1.337 câu / 753 video; tối đa 64 frame, hai nhánh cùng số frame cho mỗi câu.

| Nhánh | Đúng / 1.337 | Accuracy |
|---|---:|---:|
| RIDGE σ2 | 851 | 63,65% |
| Event coverage σ2 + nhiều mốc phụ đề | 879 | 65,74% |

So với RIDGE σ2, phương pháp nhiều mốc sửa đúng 46 câu và làm sai 18 câu, tăng ròng 28 câu (+2,09 điểm phần trăm). Hai nhánh đổi frame ở 481 câu và đổi dự đoán ở 100 câu. Bootstrap ghép cặp theo video, 10.000 lần, seed 20261102: khoảng tin cậy 95% của chênh lệch là [+0,91; +3,31] điểm phần trăm.

Phần 0: 666 câu / 379 video / 1.332 bản ghi. Phần 1: 671 câu / 374 video / 1.342 bản ghi. Hai phần không trùng video. Cả hai supervisor báo FULL_COMPLETE và 5 worker exit 0. Đã kiểm CRC ZIP, 2.674 cặp ID/nhánh duy nhất, phân shard theo hash video, frame khớp manifest, dự đoán hợp lệ và nhãn đúng/sai khớp manifest. Chi tiết và SHA256 ở `AUDIT.json`; các JSONL gộp ở `predictions.jsonl`.

**Giới hạn:** Đây là tập validation đã được quan sát trong phát triển, không phải test độc lập. Chênh lệch so với RIDGE σ2 là hiệu ứng gộp của event coverage và mốc phụ đề, không tách riêng đóng góp phụ đề. Hai checkpoint đã được resume trên runtime mới; pilot đối chiếu khớp 4 và 3 bản ghi trùng, nhưng không có lockfile đầy đủ để chứng minh hai runtime đồng nhất. Chỉ frame RIDGE có hash pixel tham chiếu độc lập; frame mới được giải mã tuần tự hai lượt, chưa so hash với manifest độc lập từ run trước. InternVL sử dụng SDPA và chia MLP theo 1.024 token để giảm VRAM; không tuyên bố tương đương số học tuyệt đối với eager attention. Không có thời gian end-to-end đầy đủ đã xác minh.
