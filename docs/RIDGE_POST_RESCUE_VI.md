# Chạy thử rescue sau chọn frame trên server trường

Thực nghiệm mới: **sigma2 + post_rescue1 + post_rescue2**, mỗi biến thể vẫn dùng 64 frame và Molmo2-4B BF16. Hai biến thể mới chỉ đổi tối đa 1 hoặc 2 frame sau khi RIDGE chọn xong; giữ số frame trong 16 khoảng thời gian theo trục candidate, bảo vệ frame đầu/cuối và các lõi rescue vốn đã được bao phủ. Không tìm được frame dư thừa phù hợp thì giữ nguyên.

Đây là giả thuyết cần kiểm chứng, chưa phải cải tiến đã chứng minh. Dùng lại score BLIP và trọng số Qwen; không tính lại. Kết quả mới nằm trong run riêng, không ghi đè 400 kết quả component.

## Bước 1 — cập nhật và kiểm tra trên server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
python3 scripts/slurm/submit_ridge_post_rescue.py pilot20 --node any --test-only
```

Lệnh test-only chỉ kiểm tra cache và yêu cầu Slurm xác nhận tài nguyên hợp lệ, chưa xin allocation. Nếu code đang có sửa đổi hoặc cache khác hợp đồng, xử lý lỗi trước khi chạy; không reset hoặc xoá kết quả cũ.

## Bước 2 — gửi một job pilot20

```bash
bash scripts/slurm/run_ridge_post_rescue.sh pilot20 --node any
squeue -u lnthanh03 -n ridge-post
```

Chỉ một job, một GPU được chọn trong gpu01/02/03, 8 CPU và 90 GiB RAM; ba biến thể chạy nối tiếp. Token HF được hỏi ẩn nếu chưa có trong môi trường terminal server. Không cần thêm job smoke hai câu.

Pilot gồm 20 câu thuộc 9 video, chọn bằng seed cố định và metadata, không chọn theo đáp án hoặc thành công/thất bại. Mẫu này có 2 câu đổi một frame; hai cấu hình giới hạn 1/2 cho cùng tập frame trong pilot. Vì vậy pilot kiểm tra chạy đúng và tái lập, chưa đánh giá lợi ích hay trường hợp đổi hai frame trên GPU. CPU preview toàn dev200 có 38 câu được đổi frame, trong đó 15 câu đổi hai frame ở post_rescue2.

## Bước 3 — mốc hoàn tất cần thấy

```text
completed_variant_answers: 60
variants.sigma2.n: 20
variants.post_rescue1.n: 20
variants.post_rescue2.n: 20
complete: false
pilot20.completed_questions: 20
pilot20.complete: true
```

`complete=false` là của mục tiêu toàn bộ 600 kết quả, không phải pilot bị lỗi. Pilot đủ 60 kết quả thoát mã 0. Nếu dừng giữa chừng, chạy lại đúng lệnh pilot20 để resume, không xoá run.

Control sigma2 phải khớp frame và đáp án baseline. Summary có so sánh sigma1/sigma4 trên đúng các câu đã đủ ba biến thể, lấy từ kết quả cũ tương thích. Không so 20 câu với accuracy 200 câu trong trường `read_only_reference_full_dev200`.

## Bước 4 — tải kết quả trên PowerShell máy cá nhân

```powershell
& 'E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_post_rescue.ps1'
```

Bạn sẽ thấy `Variant answers: 60/600; complete=False` và `Pilot20 questions: 20/20; complete=True`. Gửi đường dẫn thư mục vừa tải để kiểm tra. Export mới có nội dung câu hỏi/lựa chọn, frame swaps và score; không chứa video hoặc model. Giữ dữ liệu nghiên cứu ở chế độ riêng tư.

Sau khi kiểm tra pilot, nếu tiếp tục dev200:

```bash
bash scripts/slurm/run_ridge_post_rescue.sh dev200 --node any
```

Run giữ 60 kết quả pilot và làm tiếp 540 kết quả. Đích cuối 600/600, mỗi biến thể 200 câu. Chưa gửi job thay bạn. Đã kiểm tra CPU, chưa có accuracy GPU của phương pháp mới.
