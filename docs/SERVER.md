# Chạy trên server trường: dữ liệu/model trên RAM, 1 GPU, 192 GiB RAM, 48h

Root hiện dùng: `/media/lnthanh03/DatHa`. Thực hiện hướng dẫn này trong terminal SSH trên **máy login**. Máy login chỉ clone code, tạo cấu hình nhỏ và submit; mọi cài đặt nặng và tải dữ liệu/model diễn ra trong job được cấp trên `gpu03`.

## Dữ liệu nằm ở đâu?

| Nội dung | Vị trí |
|---|---|
| Code, cấu hình nhỏ, log, SQLite/kết quả | `/media/lnthanh03/DatHa` |
| Python venv, thư viện, pip temporary files, model, HF/cache CUDA | Thư mục riêng trong tmpfs của node GPU |
| NExTVideo.zip, annotation, video giải nén, manifest | Cùng tmpfs của job GPU |
| ZIP sau khi giải nén split | Xóa ngay để trả lại RAM |
| RAM tạm khi job kết thúc bình thường | Tự dọn; phiên sau tải lại |

Mặc định `DATA_MODE=nextqa_auto`: không cần `VIDEO_ARCHIVE`, không cần tự tải annotation, không tải dữ liệu về máy login. Script tải archive NExTVideo 24.25 GB từ mirror đã khóa, kiểm tra SHA256, chỉ giải nén video của split được chọn. Validation gồm 570 video (~2.49 GB giải nén), 4.996 câu. Runner mặc định chọn 200 câu từ toàn bộ manifest; pilot xử lý hai câu trong tập 200 này. Mỗi phiên vẫn cần tải archive đầy đủ vì chưa triển khai tải chọn từng file từ ZIP từ xa.

Mức **192 GiB RAM hệ thống** là yêu cầu Slurm, không phải VRAM và không phải peak đã đo. Phải được node/partition/account cho phép. Xin nhiều RAM không tự tăng dung lượng mount `/dev/shm`; job kiểm tra tmpfs executable, dung lượng trống và RAM khả dụng trước khi tải. Không chuyển sang disk khi RAM thiếu.

## 1. Clone hoặc cập nhật code

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab"
mkdir -p "$PERSIST_ROOT/code"
```

Nếu chưa có checkout:

```bash
git clone https://github.com/tdat1465/VideoQA-EvidenceLab.git "$REPO_ROOT"
cd "$REPO_ROOT"
mkdir -p logs
```

Nếu đã clone bản cũ, và **không có job đang chạy hoặc run cần resume bằng code cũ**:

```bash
cd "$REPO_ROOT"
git status --short
git pull --ff-only
mkdir -p logs
```

Nếu checkout có thay đổi, giữ chúng và kiểm tra trước khi pull. Bản RAM này thay đổi source contract: dùng RUN_DIR mới có hậu tố `-ram`, không ép resume journal từ bản cũ.

## 2. Kiểm tra tài nguyên Slurm

```bash
sinfo -N -n gpu03 -o "%N %P %t %m %G"
scontrol show node gpu03
scontrol show partition batch
/usr/bin/python3 --version
```

`RealMemory`, `AllocMem`, `FreeMem` là thông tin node; Slurm còn xét giới hạn account/QOS. `sinfo` cột memory dùng MiB, 192 GiB tương đương 196608 MiB. Trạng thái pending vì tài nguyên đang bận khác với bị từ chối do cấu hình không hợp lệ. Không suy ra GPU trống từ tổng số GPU của node.

Không cài torch, tải model hoặc video trên login. Môi trường yêu cầu Python >=3.10 có `venv`; node GPU cần internet tới PyPI, PyTorch, GitHub, Hugging Face và endpoint tải file mà Hugging Face chuyển hướng tới. Recipe mặc định torch2.8.0/torchvision0.23.0 CUDA12.6; job sẽ kiểm tra đúng 1 CUDA GPU có native BF16.

## 3. Tạo file cấu hình riêng

Chỉ thực hiện lệnh copy nếu chưa có file này; nếu đã có, chỉnh file hoặc dùng tên mới để giữ cấu hình cũ.

```bash
cp -n scripts/slurm/env.example.sh "$PERSIST_ROOT/videoqa-env.sh"
nano "$PERSIST_ROOT/videoqa-env.sh"
```

Nội dung cần có:

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab"
export RUN_DIR="$PERSIST_ROOT/runs/videoqa/nextqa-val-uniform16-ram"
export CONFIG_FILE=configs/molmo2_uniform16.json
export DATA_MODE=nextqa_auto
export DATASET=nextqa
export SPLIT=val
export PYTHON_BIN=/usr/bin/python3
export RAM_BASE=/dev/shm
export MIN_RAM_FREE_GIB=64
export RESUME=0
export MAX_NEW_SAMPLES=2
```

Trong nano: Ctrl+O, Enter, Ctrl+X. `MIN_RAM_FREE_GIB=64` là kiểm tra tối thiểu trước setup, **không phải RAM yêu cầu Slurm**; `--mem=192G` mới là yêu cầu tài nguyên. Không tự đặt `CUDA_VISIBLE_DEVICES`.

## 4. Kiểm tra yêu cầu 192 GiB, rồi submit pilot

```bash
source "$PERSIST_ROOT/videoqa-env.sh"
cd "$REPO_ROOT"
mkdir -p logs

sbatch --test-only --mem=192G --export=ALL scripts/slurm/job.sh
```

Nếu được chấp nhận:

```bash
JOB_ID=$(sbatch --parsable --mem=192G --export=ALL scripts/slurm/job.sh)
echo "$JOB_ID"
squeue -j "$JOB_ID"
```

Script đã mặc định 192G; ghi lại trên dòng lệnh giúp dễ kiểm tra. Nếu Slurm đòi account, thêm `--account=ACCOUNT_DUOC_CAP` vào cả hai lệnh, dùng đúng account được cấp (không mặc định lấy tên đăng nhập).

Nếu yêu cầu 192G bị từ chối, kiểm tra output bước 2 hoặc hỏi quản trị giới hạn. Có thể override một mức thấp hơn được cho phép, nhưng phải đủ RAM khả dụng/tmpfs; không giảm tùy ý để bỏ qua lỗi. `--test-only` chỉ kiểm tra yêu cầu scheduler, chưa xác nhận internet, dữ liệu hay peak RAM thực tế.

## 5. Theo dõi từ login

Khi job đã bắt đầu:

```bash
tail -f "logs/videoqa-${JOB_ID}.out"
```

Ctrl+C chỉ dừng xem log. Job tiếp tục chạy nếu đóng SSH. Tiến trình mong đợi: kiểm tra RAM → tạo môi trường trên RAM → kiểm tra GPU → tải annotation/ZIP vào RAM → xác minh hash → giải nén validation → xóa ZIP → tải model vào RAM → commit kết quả từng câu.

```bash
tail -n 80 "logs/videoqa-${JOB_ID}.err"
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

Không đánh giá trạng thái chỉ từ file `.err`: pip/model có thể in tiến độ hoặc cảnh báo vào đó. `MaxRSS` không nhất thiết phản ánh đầy đủ mọi trang tmpfs được tính vào memory cgroup; vẫn cần quan sát OOM/Slurm log. Muốn hủy đúng job do mình submit: `scancel "$JOB_ID"`.

## 6. Kiểm tra pilot hai câu

Sau khi job kết thúc:

```bash
python3 -m json.tool "$RUN_DIR/summary.json"
cat "$RUN_DIR/gpu.latest.txt"
```

Mong đợi `completed_count=2`, `selected_count=200`, `complete=false`. Kiểm tra `peak_vram_allocated_gib`, latency và số token. Exit **75** / Slurm `FAILED` với exit75 là chủ động dừng chưa hết tập câu hỏi, không tự động có nghĩa model lỗi. Nếu chưa có summary/journal, xem log setup trước.

## 7. Resume cùng run, chạy đủ 200 câu

```bash
export RESUME=1
export MAX_NEW_SAMPLES=0
JOB_ID=$(sbatch --parsable --mem=192G --export=ALL scripts/slurm/job.sh)
echo "$JOB_ID"
```

`MAX_NEW_SAMPLES=0` không giới hạn số câu mới trong phiên; config vẫn giới hạn tổng cộng 200 câu. Job tải lại môi trường, data và model vào RAM mới rồi bỏ qua các câu đã commit. Không đổi source, model revision, config hoặc dữ liệu khi resume.

Đăng nhập SSH lại thì khôi phục biến trước:

```bash
source /media/lnthanh03/DatHa/videoqa-env.sh
cd "$REPO_ROOT"
export RESUME=1
export MAX_NEW_SAMPLES=0
sbatch --mem=192G --export=ALL scripts/slurm/job.sh
```

Không submit hai job cùng RUN_DIR. Job lock chặn tình huống đó. Nếu lỗi xảy ra trước khi có `results.sqlite3`, sửa nguyên nhân rồi dùng `RESUME=0`; nếu journal đã tồn tại thì dùng `RESUME=1`. Nếu sửa code/config để xử lý lỗi, tạo run mới.

## 8. Thử phương án evidence và đối chứng

Khi baseline hoàn thành, bắt đầu run mới:

```bash
export CONFIG_FILE=configs/molmo2_evidence16.json
export RUN_DIR="$PERSIST_ROOT/runs/videoqa/nextqa-val-evidence16-ram"
export RESUME=0
export MAX_NEW_SAMPLES=2
JOB_ID=$(sbatch --parsable --mem=192G --export=ALL scripts/slurm/job.sh)
echo "$JOB_ID"
```

Kiểm tra pilot rồi resume theo bước 7. Nếu đóng SSH, nhớ đặt lại CONFIG_FILE/RUN_DIR cho evidence sau khi source file env, hoặc lưu cấu hình evidence ra một file env riêng. Chạy thêm `molmo2_uniform_refine16.json` với một RUN_DIR khác để phân biệt hiệu quả chọn bằng chứng với lợi ích gọi VLM lần hai. Chỉ chạy một thí nghiệm tại một thời điểm nếu ngân sách là một GPU.

## 9. So sánh hai run hoàn thành, không cần cài thư viện lên login

```bash
PYTHONPATH="$REPO_ROOT/src" python3 -m evidencelab compare \
  "$PERSIST_ROOT/runs/videoqa/nextqa-val-uniform16-ram" \
  "$PERSIST_ROOT/runs/videoqa/nextqa-val-evidence16-ram" \
  --output "$PERSIST_ROOT/runs/videoqa/comparison.json"
```

Lệnh chỉ đọc SQLite/kết quả nhỏ, dùng thư viện chuẩn Python. Để chạy toàn bộ 4.996 câu validation sau pilot, sao chép config, đặt `limit: 0` rồi bắt đầu một RUN_DIR mới. Không thay limit giữa run.

## Khi gặp lỗi

| Hiện tượng | Xử lý |
|---|---|
| RAM request bị Slurm từ chối | Kiểm tra RealMemory/partition/account/QOS; gửi output `scontrol` để chọn mức hợp lệ |
| `RAM filesystem is noexec` | Cần tmpfs executable do quản trị cấp; đặt RAM_BASE tới mount đó, không dùng disk thay thế |
| Thiếu tmpfs/RAM dù xin 192G | `--mem` không tăng kích thước mount; kiểm tra tmpfs trên GPU node và RAM node còn trống |
| Tải bị chặn | GPU node cần truy cập nguồn tải; không tự chuyển tải về disk login |
| `CUDA out of memory` | Dùng config uniform8 và RUN_DIR mới; xin thêm RAM hệ thống không tăng VRAM |
| `Resume contract mismatch` | Quay về đúng source/config hoặc dùng RUN_DIR mới, không sửa contract |
| Job bị kill cứng | Các câu đã commit còn nguyên; RAM cleanup có thể không chạy. Chỉ dọn đúng thư mục job của mình sau khi xác nhận job đã dừng |

STAR vẫn cần `DATA_MODE=local_archive` và nguồn video/annotation do người dùng cung cấp; chế độ tự tải RAM hiện áp dụng cho NExT-QA. Chưa chạy scheduler hoặc model đầy đủ trên GPU trường trong lần cập nhật này.
