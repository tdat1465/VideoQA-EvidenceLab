# RIDGE smoothing ablation v1 — một job, một GPU

Lượt này kiểm tra smoothing/phân vùng trên **200 câu phát triển**, trước khi quyết định chạy benchmark lớn. Molmo2-4B, BLIP, prompt và candidate grid giữ nguyên. Dùng lại trọng số Qwen đã sinh cho RIDGE full, sau khi kiểm tra contract và checksum; không tải hoặc chạy lại Qwen.

Đã kiểm tra CPU và luồng tích hợp bằng mock; **chưa chạy GPU thật cho ablation**. Kết quả baseline 66,94% không phải kết quả của prototype.

## 1. Cập nhật code và chạy pilot

Trên terminal server, tại checkout RIDGE hiện có:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_smoothing.sh pilot
```

Nhập Hugging Face read token tại terminal nếu launcher hỏi. Launcher chạy `sbatch --test-only` trước, rồi submit **một job tên `ridge-sm`**, 1 GPU, 8 CPU, 90 GiB RAM, giới hạn 48 giờ, chỉ gpu01/02/03. Không tải model/video trên login node. Các thư viện, weights và video nằm trong tmpfs của allocation; chỉ code, cache điểm/curve và kết quả nhỏ được lưu lâu dài.

Pilot xử lý **2 câu × 5 biến thể = 10 kết quả**, cùng một job, chạy tuần tự:

| Tên | Ý nghĩa |
|---|---|
| `sigma2` | RIDGE tác giả, sigma=2, làm control |
| `sigma1` | RIDGE tác giả, sigma=1 |
| `sigma4` | RIDGE tác giả, sigma=4 |
| `adaptive` | `adaptive_sigma` có sẵn của tác giả |
| `multiscale_v1` | Prototype độc lập mô tả ở mục 5 |

BLIP chấm **một lần cho mỗi câu**; năm biến thể dùng chung score cache. Pilot và dev200 dùng cùng manifest 200 câu, cùng database và contract, nên có thể tiếp tục mà không chạy lại câu đã commit.

Baseline được đọc từ:

```text
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1
```

Run mới hoàn toàn riêng:

```text
/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-smoothing-v1-dev200
```

Baseline source fingerprint vẫn là `e92cfc762bde69a511beb494a4b5305bd43e89ccdb72d599de938366da1be057`. Không thay module source của baseline; script ablation có hash riêng trong contract. Nếu code/config/runtime/model/annotation hoặc baseline cache không khớp, worker dừng, không tự đổi protocol.

## 2. Theo dõi và mốc pilot đạt

```bash
squeue -u lnthanh03 -n ridge-sm
```

Lấy job ID launcher in ra, rồi xem:

```bash
tail -n 25 logs/ridge-sm-JOBID.out
tail -n 25 logs/ridge-sm-JOBID.err
```

Thay `JOBID` bằng ID thực. Sau khi pilot dừng:

```bash
cat /media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-smoothing-v1-dev200/summary.json
```

Mốc cần kiểm tra:

- `completed_variant_answers=10`; mỗi variant có `n=2`.
- `one_time_scored_questions=2`, thể hiện một cache BLIP cho mỗi câu.
- `complete=false`: đúng với pilot, vì manifest mục tiêu vẫn là 200 câu/1.000 kết quả.
- `baseline_reproduction.n=2`; kiểm tra `same_frames` và `same_prediction`. Nếu control khác baseline, đọc trace/runtime trước khi diễn giải cải tiến; không tự bỏ qua sai khác.
- Không có traceback/OOM hoặc lỗi checksum.

Worker trả exit 75 khi đạt cap pilot, nên Slurm có thể ghi FAILED. Dòng `Smoothing session stopped cleanly` cùng các count trên mới là căn cứ xác nhận dừng đúng thiết kế. Chỉ có exit 75 không đủ để kết luận thành công.

## 3. Tải pilot về local

Trong PowerShell local:

```powershell
cd 'E:\Codex\Documents\ChatGPT\KLTN'
powershell -ExecutionPolicy Bypass -File '.\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_smoothing.ps1'
```

Script tải logs, summary/predictions/contract/manifest và thư mục `scores`, `curves` vào `outputs\ridge-smoothing-results-<timestamp>`. Nhập mật khẩu SSH trực tiếp tại terminal nếu được hỏi. Tải sau khi pilot dừng; summary chỉ cập nhật khi worker thoát phiên. Không sao chép SQLite đang chạy để tránh snapshot không nhất quán.

Mỗi file curve có ID câu hỏi, nhãn vùng, đường đã làm mượt/đạo hàm, peak, ngân sách vùng và vị trí frame cho từng biến thể. Raw score nằm trong file cache tương ứng; không lưu frame/video lâu dài.

## 4. Tiếp tục 200 câu

Sau khi kiểm tra pilot, trên server chạy:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
bash scripts/slurm/run_ridge_smoothing.sh dev200
```

Lệnh submit **một job mới** tiếp tục cùng run, bỏ qua các variant đã commit. Không submit khi job `ridge-sm` cũ còn chạy/đợi. Mốc hoàn tất là `complete=true`, `completed_variant_answers=1000`, mỗi variant có `n=200`.

Nếu bị dừng trước đó, chạy lại mode `dev200` sau khi job đã kết thúc; SQLite và cache đã lưu là nguồn resume. Không đổi code hoặc xóa run giữa chừng. Không chạy lại hai full baseline.

Manifest được chọn cố định bằng seed=18, theo toàn bộ nhóm câu của từng video có ít nhất 64 candidate trong baseline. Không dùng kết quả đúng/sai để chọn câu. Đây là **tập phát triển đã thuộc LVB được xem kết quả**, không gọi là tập kiểm tra chưa từng dùng. Các video trong manifest cần được loại hoàn toàn khỏi một split video-disjoint sau này.

## 5. Prototype đa thang là gì

`multiscale_v1` là **ý tưởng triển khai độc lập**, chưa khẳng định novelty hoặc hiệu quả:

1. Chuẩn hóa score như RIDGE, phân tích bằng hàm tác giả ở sigma 1, 2, 4.
2. Mỗi candidate nhận nhãn có ít nhất hai thang đồng ý; nếu không có đa số, dùng nhãn sigma=2.
3. Với peak chỉ được thang nhỏ phát hiện và không có peak trong khoảng 2 candidate ở hai thang còn lại, xét độ tương phản raw score so với flank cục bộ. Chỉ giữ core 3 candidate khi contrast ít nhất 0,15 và ít nhất hai candidate có normalized score >=0,5. Đây là ngưỡng prototype được cố định trước lượt chạy.
4. Giữ các hàm phân ngân sách/chọn trong vùng của tác giả. Ranking dùng sigma=2, thay bằng ranking sigma=1 trong các core được rescue. Không tăng số frame hoặc thêm candidate chưa chấm.

Gate này có thể bỏ sự kiện thật chỉ xuất hiện ở một candidate; prototype không bảo đảm thu hồi mọi chi tiết ngắn. Ở grid khoảng 1 FPS, dữ liệu đã không có trong candidate pool không thể được phục hồi bằng smoothing. Không coi việc chạy đúng trên tín hiệu tổng hợp là bằng chứng tăng QA accuracy.

## 6. Cách đọc kết quả

So sánh đa thang với **sigma2, adaptive và sigma đơn tốt nhất** trên cùng 200 câu. Tập 200 chỉ để sàng lọc ý tưởng, không đủ mặc định chứng minh gain nhỏ. Xem đồng thời câu cải thiện, câu giảm, frame/peak thực sự thay đổi và chi phí selection. WFS-SB là đối chứng gần cần thêm khi muốn lập luận tính mới của đa thang.

Thời gian `cached_question_seconds` gồm selection và decode frame trả lời/Molmo2; loại chi phí shared BLIP, prior Qwen, model setup/indexing/transfer và I/O diagnostics/results. `one_time_scoring_seconds` báo BLIP/candidate decode một lần riêng. Không dùng timing sau cache như end-to-end speedup. Summary còn ghi thời gian setup/transfer/worker đã quan sát từ các session.

Đã kiểm tra: các control so khớp trực tiếp code tác giả; ngân sách frame; manifest không phụ thuộc nhãn đúng/sai; cache checksum; dừng giữa variant và resume; mock pipeline hai allocation chỉ chấm mỗi câu một lần, baseline giữ nguyên bytes. Thử nghiệm GPU thật vẫn bắt đầu bằng pilot ở mục 1.
