# Chạy trên GPU nhà trường, mỗi phiên 48h

`scripts/slurm/job.sh` lấy thông số vận hành từ repo DyGEnc đã tham khảo: partition `batch`, node `gpu03`, 1 GPU, 8 CPU, 90 GB RAM, 48h. Account và đường dẫn là cấu hình của người chạy, không nhúng tài khoản hoặc token cá nhân vào repo public. Nếu scheduler đổi tên/node, override bằng tùy chọn `sbatch`.

Môi trường Python, dependencies, model HF, cache, video giải nén và frame nằm trong tmpfs executable (mặc định `/dev/shm`); không fallback xuống disk. Mã, archive/annotation đầu vào do người dùng cấp và kết quả nhỏ có thể nằm ở persistent storage. Xem lưu ý lưu archive ở [DATA.md](DATA.md). 90 GB là **RAM hệ thống**, hoàn toàn khác VRAM của GPU.

## Chuẩn bị một lần

```bash
git clone https://github.com/tdat1465/VideoQA-EvidenceLab.git
cd VideoQA-EvidenceLab
mkdir -p logs
cp scripts/slurm/env.example.sh /path/to/private/videoqa-env.sh
# Sửa các đường dẫn trong videoqa-env.sh cho đúng máy.
```

Nếu dùng NExT-QA, tải annotation nhỏ bằng `python3 scripts/fetch_nextqa_annotations.py --split val --output /path/to/annotations`. Tải video theo trang dataset và đặt `VIDEO_ARCHIVE` tới archive ZIP/TAR chứa video gốc. STAR dùng `STAR_val.json` + archive Charades video gốc. Repo không thể tự suy ra nơi bạn đã lưu các video này từ checkpoint AGQA của DyGEnc.

Máy cần Python >=3.10 với `venv`, internet ra PyPI/PyTorch/Hugging Face trong giai đoạn setup, và driver CUDA phù hợp. Recipe khóa torch2.8.0/torchvision0.23.0 CUDA12.6. Nếu cluster hỗ trợ một wheel khác, đổi `TORCH_INDEX` có chủ đích và kiểm tra lại; không trộn torch/torchvision khác cặp. Job không cài FlashAttention.

## Phiên pilot

```bash
source /path/to/private/videoqa-env.sh
export MAX_NEW_SAMPLES=2
export RESUME=0
sbatch --account=YOUR_SCHOOL_ACCOUNT --export=ALL scripts/slurm/job.sh
```

Chờ job kết thúc, kiểm tra `RUN_DIR/summary.json` và log. Mã 75 / trạng thái Slurm FAILED với exit75 chỉ có nghĩa dừng chủ động chưa hết split; các lỗi khác phải được đọc và xử lý. Chưa có kết quả GPU đo trên máy trường trong bản phát hành này.

## Tiếp tục cùng run

```bash
export MAX_NEW_SAMPLES=0
export RESUME=1
sbatch --account=YOUR_SCHOOL_ACCOUNT --export=ALL scripts/slurm/job.sh
```

Giữ nguyên `RUN_DIR`, config, annotation, bytes video, source và các phiên bản thư viện. Mỗi phiên dựng lại môi trường/weights trên RAM; journal persistent giúp bỏ qua câu đã commit. Khi gần hết giờ, Slurm gửi USR1 trước 180 giây; runner cố hoàn thành câu hiện tại rồi dừng. Nếu câu quá lâu hoặc bị SIGKILL, transaction đã commit vẫn tồn tại và câu dở chạy lại. Không tự requeue để tránh dùng thêm allocation ngoài ý muốn.

Nếu lỗi trong giai đoạn cài dependencies/staging trước khi có `results.sqlite3`, sửa lỗi rồi dùng `RESUME=0`. Nếu journal đã được tạo, dùng `RESUME=1`. Nếu đổi source/config hoặc dữ liệu để sửa lỗi, dùng **RUN_DIR mới**; đừng xóa hoặc sửa contract để ép resume.

## Đổi thí nghiệm

```bash
export CONFIG_FILE=configs/molmo2_evidence16.json
export RUN_DIR=/absolute/persistent/path/videoqa-runs/nextqa-val-evidence16
export RESUME=0
export MAX_NEW_SAMPLES=2
sbatch --account=YOUR_SCHOOL_ACCOUNT --export=ALL scripts/slurm/job.sh
```

Sau pilot, resume không giới hạn câu mới. Chạy đủ baseline và đối chứng trên cùng manifest trước khi kết luận cải thiện. Để chạy cả split, sao chép config, đặt `limit: 0`, dùng run mới ngay từ đầu; không đổi limit của run đã chạy.

Các file `environment.latest.txt` và `gpu.latest.txt` ghi môi trường/GPU phiên gần nhất. `contract.json` là đặc tả run; SQLite giữ event session và mọi câu đã commit. Xem peak VRAM sau pilot; BF164B phù hợp để bắt đầu nhưng context, processor và độ phân giải cũng tiêu tốn VRAM. Qwen7B/32frame cần pilot riêng.
