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

Chỉ một job, một GPU được chọn trong gpu01/02/03, 8 CPU và 90 GiB RAM; ba biến thể chạy nối tiếp. Thời lượng tối đa mặc định **48 giờ tính cả setup**, dành 5 phút trước giới hạn để xuất checkpoint và dọn tmpfs. Job thoát ngay khi hoàn thành, không giữ GPU hết 48 giờ. Quy định GPU idle 60 phút độc lập với tổng thời lượng job. Token HF được hỏi ẩn nếu chưa có trong môi trường terminal server. Không cần thêm job smoke hai câu.

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

Lệnh giữ mọi kết quả đã commit và làm phần còn thiếu; nếu mới xong pilot thì còn 540 kết quả. Đích cuối 600/600, mỗi biến thể 200 câu. Trạng thái thực nghiệm mới nhất và cách tiếp tục được ghi ở phần dưới.

## Tiếp tục sau khi job 79497 bị hủy

Bản xuất ngày 22/10 lúc 21:25 có **238/600** kết quả: 79 câu đủ ba biến thể và một câu mới có sigma2. Trên cùng 79 câu, cả ba biến thể đều đúng 59 câu. Không so trực tiếp 59/80 của sigma2 với 59/79 của post-rescue. Có 238/238 replay CPU khớp, các ràng buộc đạt và 60 bản ghi pilot giữ nguyên. Còn **362** kết quả, gồm hai biến thể còn thiếu ở câu dang dở. Ba ca đổi hai frame đã được chạy trên GPU.

Log ghi CANCELLED; theo thông báo của trường, nguyên nhân là GPU không hoạt động trong 60 phút. Gói cũ chỉ có snapshot GPU trước khi nạp model, không có lịch sử utilization nên chưa xác nhận được khoảng idle hoặc GPU UUID trường đã theo dõi. Snapshot 0% không đại diện cho toàn phiên đã tạo 178 kết quả mới.

Sau khi cập nhật code, chạy:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
python3 scripts/slurm/submit_ridge_post_rescue.py dev200 --node any --test-only
bash scripts/slurm/run_ridge_post_rescue.sh dev200 --node any
```

Chỉ launcher, tổng thời lượng và log giám sát được đổi; worker chọn frame, contract, precision và cache giữ nguyên nên tiếp tục được journal cũ. Mỗi lệnh submit một job, mặc định tối đa 48 giờ, thoát khi hoàn thành. Mốc 55 phút tạm đặt ở commit ed250be đã được bỏ vì nhầm lẫn tổng wall time với thời gian GPU idle. Setup vẫn tải model trong tmpfs, khoảng 30 phút ở các phiên đã quan sát; giữ job đủ dài để không phải tải lại chỉ vì mốc 55 phút. Nếu bị dừng khi chưa đủ 600, sau khi job trước đã thoát thì chạy lại lệnh dev200. Không xóa run hoặc sửa SHA trong contract.

Có thể chủ động chọn tổng wall time bằng `--session-minutes 480` để xin tối đa 8 giờ, hoặc bỏ tùy chọn để dùng mặc định 48 giờ. Dải cho phép là 10–2880 phút; 60 phút là giá trị wall time hợp lệ và không được dùng làm trần theo quy định idle. Giới hạn mới chỉ áp dụng khi submit job mới. Job đã được xin 55 phút không tự đổi thời lượng khi git pull.

Trong run có thêm `gpu-util-JOBID.csv` lấy mẫu các GPU hiện với nvidia-smi mỗi 10 giây, `gpu-context-JOBID.json` ghi UUID CUDA và thông tin GPU do Slurm cấp, cùng `phases-JOBID.log`. Phải ghép UUID trước khi diễn giải index. Đây là log đọc trạng thái, không có tác vụ tạo tải GPU để giữ allocation. Downloader tải các tệp này nếu có; phiên cũ không có vẫn tải được.

Tài liệu tham chiếu cho tùy chọn triển khai: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html), [NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html). Quy định idle 60 phút là thông tin do người dùng cung cấp từ trường, không suy ra từ tài liệu Slurm.
