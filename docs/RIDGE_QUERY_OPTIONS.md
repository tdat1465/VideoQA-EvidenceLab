# Thử nghiệm RIDGE với thông tin lựa chọn đáp án

Thử nghiệm mới sau khi post-rescue hoàn tất 600/600 nhưng không sửa được lỗi nào trên dev200. Mục tiêu là kiểm tra tác động của **thông tin truy vấn dùng để chọn frame**, giữ nguyên Molmo2-4B BF16, trọng số Qwen question-only, candidate grid và ngân sách 64 frame.

Đây là đối chứng phát triển, chưa có kết quả GPU mới và chưa phải đóng góp phương pháp đã được xác nhận. Bước này chưa tách từng sự kiện hoặc tối ưu độ phủ nhiều sự kiện.

## Hai biến thể trong một job

| Biến thể | Cách chọn frame |
|---|---|
| `sigma2` | Score câu hỏi cũ, trọng số cũ, RIDGE σ2. Chạy lại Molmo2 làm control cùng phiên. |
| `qopts_mean_sigma2` | Trộn đều score câu hỏi cũ với score từ toàn bộ lựa chọn; giữ trọng số cũ và RIDGE σ2. |

BLIP cũ chỉ nhận tối đa 35 token và có tiền xử lý cắt 50 từ. Nối nguyên câu hỏi và lựa chọn sẽ có nguy cơ bỏ mất phần lựa chọn. Worker mới chuẩn hóa từng lựa chọn rồi chia toàn bộ token nội dung thành đoạn tối đa 33 token, cộng token đặc biệt thành tối đa 35. Không cắt bỏ phần cuối lựa chọn. Không ghép thêm câu hỏi vào từng đoạn; score question-only được giữ nguyên từ cache để tránh thay đổi đồng thời cách đọc phần câu hỏi.

Mỗi lựa chọn có một score bằng trung bình các đoạn của chính nó. Trung bình tiếp giữa các lựa chọn để lựa chọn dài không được tăng trọng số chỉ vì nhiều đoạn. Score mới = 0,5 × score câu hỏi cũ + 0,5 × trung bình score các lựa chọn. Hệ số 0,5 cố định trước chạy. Các lựa chọn được sắp theo nội dung chuẩn hóa nên hoán vị thứ tự A/B/C không đổi truy vấn hay score. Không dùng đáp án đúng hoặc loại câu để chọn frame.

Đây là cách gộp đơn giản để kiểm tra việc thêm thông tin lựa chọn; một kết quả âm không phủ định mọi phương pháp sử dụng lựa chọn. Các đoạn có thể cắt qua mô tả sự kiện, và trung bình có thể làm loãng một sự kiện hiếm. Hướng tách, định vị và phủ từng sự kiện cần ablation riêng sau đó.

## Chạy pilot trên terminal server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_query_options.sh pilot20 --node any
```

Launcher tự kiểm tra nguồn cache và chạy `sbatch --test-only` trước khi nộp **một job tên `ridge-query`, một GPU**. Hai biến thể chạy tuần tự trong job đó. Nhập Hugging Face read token trực tiếp ở terminal nếu được hỏi; không đưa token vào chat hoặc file hướng dẫn.

Slurm chọn một GPU trên gpu01/02/03, loại gpu04; xin 8 CPU và 90 GiB RAM. Tổng wall time mặc định 48 giờ gồm setup, có khoảng đệm 5 phút để xuất kết quả. Quy định GPU idle 60 phút là điều kiện riêng. Giữ CUDA_VISIBLE_DEVICES của Slurm và ghi telemetry/UUID thật; không có tác vụ tạo tải để giữ job.

Pilot dùng 20 câu theo toàn bộ nhóm video của manifest post-rescue trước, chọn bằng ID/video/candidate count, không chọn theo kết quả đúng sai. Có 5 câu SSS trong pilot. Đây là các câu phát triển đã được quan sát, không phải mẫu độc lập.

Theo dõi:

```bash
squeue -u lnthanh03 -n ridge-query
tail -n 20 logs/ridge-query-JOBID.out
tail -n 20 logs/ridge-query-JOBID.err
```

Thay JOBID bằng ID launcher in ra. Có thể kiểm tra scheduler mà không nộp job bằng:

```bash
python3 scripts/slurm/submit_ridge_query_options.py pilot20 --node any --test-only
```

Lệnh test-only là tùy chọn; không cần chạy nó rồi nộp thêm một smoke job. Launcher chặn khi có job RIDGE liên quan đang chạy hoặc chờ.

## Kết quả pilot cần đạt

Run mới: `/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-query-options-v1-dev200`.

```bash
cat /media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-query-options-v1-dev200/summary.json
```

- `completed_variant_answers=40`; mỗi biến thể có `n=20`.
- `pilot20.complete=true`, `pilot20.completed_questions=20`.
- `complete=false` là đúng ở pilot vì mục tiêu toàn dev200 là 400 bản ghi.
- `baseline_reproduction` có n, same_frames và same_prediction đều bằng 20. Worker dừng nếu control khác baseline; không bỏ qua kiểm tra này để tiếp tục.
- `new_qwen_calls=0`; `one_time_scored_questions=20`. BLIP chấm lựa chọn mới, không dùng lại nhầm score question-only làm score có lựa chọn.
- Score cache có query plan, toàn bộ đoạn token và score của từng lựa chọn; đủ 64 frame có thứ tự ở mỗi bản ghi.
- Không có traceback, CUDA OOM, lỗi checksum hoặc control mismatch. Một cảnh báo H264 không tự chứng minh mọi frame đã decode đúng; cần kiểm tra ảnh nếu diễn giải lỗi theo nội dung.

Offline preview trên tokenizer thật cho thấy pilot có 92.207 cặp frame và đoạn văn bản mới, khoảng 6,90 lần số cặp của câu hỏi cũ trên đúng pilot. Dev200 cần 1.127.440 cặp mới, khoảng 5,69 lần baseline question-only trên cùng tập. Đây là số phép chấm, **không phải tỷ lệ latency đã đo**. Chấm lựa chọn có thể là phần tốn thời gian nhất; pilot cần kiểm tra chi phí trước khi quyết định mở rộng.

## Tải kết quả về máy cá nhân

Sau khi worker thoát, chạy trong PowerShell local:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_query_options.ps1"
```

Downloader lấy logs, metadata, telemetry, `curves/` và score có lựa chọn dưới `scores/`; không tải video/model. Script kiểm tra SHA256 với server và chỉ tải file thiếu hoặc khác. Nếu tải bị ngắt, chạy lại cùng thư mục đã tạo:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_query_options.ps1" -Destination "E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-query-options-results-YYYYMMDD-HHMMSS"
```

Downloader dùng một kết nối SSH để đọc inventory và một phiên SFTP để lấy toàn bộ file thiếu. Với xác thực mật khẩu thông thường, có hai lượt hỏi mật khẩu nếu còn file cần tải; không hỏi lại theo từng file. Nhập sai mật khẩu hoặc mất kết nối có thể dẫn tới hỏi lại. Nếu bị ngắt, chạy lại đúng `-Destination` cũ để giữ file đã đủ. Gửi đường dẫn `Da tai vao` để đối chiếu pilot trước khi chạy tiếp. Điểm tăng trên 20 câu chỉ là tín hiệu phát triển, không phải bằng chứng cải tiến đã được xác nhận.

## Tiếp tục dev200 và resume

Sau khi pilot hợp lệ và đã xem chi phí cùng gains/losses:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
bash scripts/slurm/run_ridge_query_options.sh dev200 --node any
```

Giữ cùng run: 40 bản ghi pilot được giữ, còn 360 bản ghi. Đích là 400/400, mỗi biến thể 200 câu. Nếu một phiên bị dừng, đợi job cũ kết thúc rồi chạy lại cùng mode; các đáp án đã commit và score hoàn tất được dùng lại. Nếu đang chấm dở một câu chưa ghi score thì câu đó cần chấm lại. Không đổi code, trọng số, hệ số gộp hoặc xóa journal để resume. Worker xuất kết quả khi thoát; checkpoint SQLite là nguồn resume.

Không chạy lại các full baseline hoặc post-rescue. Nguồn read-only vẫn là baseline 1.337 câu và smoothing dev200 đã hoàn tất. Score mới và run mới hoàn toàn riêng. Chỉ khi prototype có dấu hiệu hữu ích mới cân nhắc run σ1 cùng truy vấn mới; so với σ1 cũ 127/200 để tránh chỉ vượt một đối chứng yếu. Mọi kết luận cần kiểm chứng trên video chưa dùng để chọn cấu hình.

## Phạm vi kiểm tra trước chạy

Kiểm tra CPU/mock bao gồm: giữ mọi token lựa chọn, giới hạn 35 token cho mỗi forward, thứ tự lựa chọn không đổi truy vấn, gộp score đúng, cache sai hoặc query đổi bị từ chối, dừng giữa biến thể và resume không trả lời lại control, một job một GPU và chặn job trùng. Control trên 200 cache thật tái lập đúng tập frame gốc. Đây chưa phải chạy BLIP/Molmo2 trên GPU; các kiểm tra không xác nhận accuracy của prototype.

Toàn bộ mã nguồn `src/`, requirements và pyproject của baseline giữ nguyên fingerprint. Script mới có checksum riêng trong contract. Model, thư viện và video nặng vẫn chỉ ở tmpfs của allocation; việc tách staging CPU thành job riêng chưa được triển khai trong run này.
