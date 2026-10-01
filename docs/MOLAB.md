# Chạy LongVideoBench/LENS trên MoLab marimo

Profile riêng `scripts/molab.py`; không chạy submit.py/sbatch, không giả lập SLURM_JOB_ID.
Code hỗ trợ Molmo2/LLaVA và Uniform/FOCUS/LENS với các config `lvb_*.json` hiện có.

Thông tin người dùng cung cấp: Python 3.13.11, Torch 2.11.0+cu130, RTX PRO 6000 Blackwell
Server Edition (97887 MiB VRAM), driver 595.71.05, CUDA matmul hoạt động; cgroup v1 có
giới hạn **160 GiB**. Số 8.589.934.592 GiB ở filesystem là giá trị không dùng được để suy ra
quota; 503 GiB tmpfs là tổng filesystem, không phải RAM cấp riêng.

## Thiết kế và giới hạn

- Giữ kernel notebook nguyên trạng. `uv` chuẩn bị Python **3.12.14** riêng, Torch **2.8.0+cu128**,
  torchvision 0.23.0 và requirements đã pin riêng cho backbone. Không cài đè Torch vào Python 3.13.
  [uv quản lý Python](https://docs.astral.sh/uv/guides/install-python/),
  [wheel chính thức PyTorch](https://pytorch.org/get-started/previous-versions/).
- Python/venv nằm trong scratch `/tmp/videoqa.molab.runtime.*` vì `/dev/shm` trong container
  có thể `noexec`. Cache lớn, trọng số và video vẫn nằm trong `/dev/shm/videoqa.molab.*`.
  Đây là scratch của MoLab, không phải disk login của trường; profile Slurm không đổi.
- Đọc cgroup v1/v2 của container, chặn tải khi thiếu reserve 6 GiB trong ngân sách 90 GiB;
  driver theo dõi RAM mỗi giây và kết thúc nhóm subprocess nếu chạm 90 GiB. Đây là guard
  phần mềm, **không phải hard limit mới**: cgroup vẫn 160 GiB, spike giữa các lần đo có thể vượt.
- Tải một video bằng HTTP Range 4 MiB; xử lý xong thì xóa. Setup/cache được dùng lại giữa
  pilot và resume trong cùng notebook session. Mỗi lần gọi runner vẫn nạp model lên GPU lại.
- Một workspace chỉ cho phép một setup/run tại một thời điểm; không chạy hai backbone song song.
- Driver mặc định mỗi run tối đa 3 giờ, CLI cho tối đa 10 giờ. Thời gian này không cho biết
  notebook còn bao lâu trong quota của MoLab; tính cả thời gian đã mở phiên và setup.
- Kết quả ở `/marimo/videoqa-molab/runs/` **chưa phải bản sao bền vững ngoài notebook**.
  Khi runner kết thúc, driver xuất ZIP qua SQLite backup API, gồm journal, metadata,
  predictions, summary và index nhỏ. Không chứa `.env`, token, weights hoặc video.
  Phải bấm download ZIP về máy trước khi đóng/hết phiên. Mất container đột ngột trước
  download vẫn có thể mất kết quả kể từ bản backup đã tải gần nhất.
  [Chính sách lưu trữ MoLab](https://marimo.io/pages/molab/storage).

Các block dưới là **cell Python**. Chuyển notebook sang chế độ chạy thủ công/lazy để các
cell tạo subprocess không tự chạy lại khi sửa cell khác. Chạy từng bước, không Run All.
[Tài liệu chạy cell marimo](https://docs.marimo.io/guides/reactivity/).

## 1. Token và code

Thêm `HF_TOKEN` trong Secrets, có quyền đọc LongVideoBench và đã chấp nhận điều kiện dataset.
Không ghi token vào code hoặc gửi token qua chat. Nếu env chưa nhận token, tải lại môi trường
theo UI MoLab rồi kiểm tra boolean; không in giá trị token.

```python
import os as _os
import subprocess as _sp
from pathlib import Path

assert _os.environ.get("HF_TOKEN"), "Thêm HF_TOKEN vào Secrets trước"
MOLAB_REPO = Path("/tmp/VideoQA-EvidenceLab-molab")
if not MOLAB_REPO.exists():
    _sp.run([
        "git", "clone", "--branch", "codex/molab-longvideobench", "--single-branch",
        "https://github.com/tdat1465/VideoQA-EvidenceLab.git", str(MOLAB_REPO)
    ], check=True)
print(_sp.check_output(["git", "-C", str(MOLAB_REPO), "rev-parse", "HEAD"], text=True))
```

Ghi lại commit này. Phiên sau checkout **đúng commit cũ** trước khi resume, không tự git pull.
Nếu repo đã tồn tại, cell không cập nhật code. Lệnh so sánh yêu cầu cùng source/runtime;
không ghép run MoLab cu128 với run trường cu126 trong comparator hiện tại.

## 2. Setup Molmo2

```python
import subprocess as _sp
import sys as _sys

_sp.run([
    _sys.executable, str(MOLAB_REPO / "scripts/molab.py"), "setup",
    "--config", str(MOLAB_REPO / "configs/lvb_molmo2_lens64.json")
], check=True)
```

Chờ `Setup complete` và doctor xác nhận GPU/precision. Bước này kiểm tra quyền dataset và
HTTP Range trước khi tải Torch; model weights tải ở bước run. Setup đã xong được tái sử dụng.
Lỗi setup có thể chạy lại cùng cell; không xóa kết quả cũ để sửa lỗi cài đặt.

## 3. Pilot 2 câu chạy nền

```python
import subprocess as _sp
import sys as _sys
from pathlib import Path as _Path

_work = _Path("/marimo/videoqa-molab")
_work.mkdir(exist_ok=True)
_out = _work / "runs/molmo2-lens64-200"
assert not (_out / "results.sqlite3").exists(), "Đã có journal: dùng cell resume"
with (_work / "molmo2-lens64.log").open("a") as _log:
    _process = _sp.Popen([
        _sys.executable, "-u", str(MOLAB_REPO / "scripts/molab.py"), "run",
        "--config", str(MOLAB_REPO / "configs/lvb_molmo2_lens64.json"),
        "--output", str(_out), "--max-new-samples", "2", "--max-seconds", "10800"
    ], stdout=_log, stderr=_sp.STDOUT, start_new_session=True)
print("Driver PID:", _process.pid)
```

Chỉ bấm chạy cell này một lần. Process có session riêng; nút interrupt của cell không phải
cách dừng nó. Dùng lệnh stop ở bước 7. Không có squeue/sacct trên MoLab.

## 4. Xem log và summary

Chạy lại cell này để xem tiến độ; không khởi động thêm run:

```python
from pathlib import Path as _Path

_work = _Path("/marimo/videoqa-molab")
_log = _work / "molmo2-lens64.log"
print("\n".join(_log.read_text(errors="replace").splitlines()[-60:]) if _log.exists() else "Chưa có log")
for _name in ("summary.json", "session-status.json"):
    _file = _work / "runs/molmo2-lens64-200" / _name
    if _file.exists():
        print(_name, _file.read_text())
```

Pilot đúng: `completed_count=2`, `complete=false`, `mean_vlm_calls=2` cho LENS,
driver `exit_code=75`, log có `BACKUP 2 questions`. 75 là dừng có kiểm soát, không phải lỗi model.
Lỗi model/setup phải xem traceback; không bỏ qua chỉ vì có SQLite từ lần trước.

## 5. Download backup

Sau khi kết thúc mỗi phiên runner, ZIP tự tạo. Cell sau hiển thị nút; **bấm nút để tải về máy**:

```python
import marimo as _mo
from pathlib import Path as _Path

_zip = _Path("/marimo/videoqa-molab/backups/molmo2-lens64-200.zip")
_mo.stop(not _zip.exists(), "Chưa có backup; xem log trước")
_mo.download(_zip.read_bytes(), filename=_zip.name, mimetype="application/zip", label="Tải backup kết quả")
```

Muốn snapshot cả khi run đang chạy, chạy cell này trước cell download:

```python
import subprocess as _sp
import sys as _sys

_sp.run([
    _sys.executable, str(MOLAB_REPO / "scripts/molab.py"), "backup",
    "--output", "/marimo/videoqa-molab/runs/molmo2-lens64-200",
    "--archive", "/marimo/videoqa-molab/backups/molmo2-lens64-200.zip"
], check=True)
```

Snapshot chỉ bao gồm câu đã commit tại thời điểm backup, không bao gồm câu đang suy luận.

## 6. Resume trong cùng notebook session

Chỉ chạy khi pilot đã kết thúc. Cell chạy nền giống bước 3, đổi danh sách args thành:

```python
import subprocess as _sp
import sys as _sys
from pathlib import Path as _Path

_work = _Path("/marimo/videoqa-molab")
with (_work / "molmo2-lens64.log").open("a") as _log:
    _process = _sp.Popen([
        _sys.executable, "-u", str(MOLAB_REPO / "scripts/molab.py"), "run",
        "--config", str(MOLAB_REPO / "configs/lvb_molmo2_lens64.json"),
        "--output", str(_work / "runs/molmo2-lens64-200"),
        "--resume", "--max-new-samples", "0", "--max-seconds", "10800"
    ], stdout=_log, stderr=_sp.STDOUT, start_new_session=True)
print("Resume driver PID:", _process.pid)
```

`max-new-samples=0` chạy phần còn lại trong giới hạn thời gian. Tải backup mới sau mỗi phiên.

## 7. Dừng có kiểm soát

```python
import subprocess as _sp
import sys as _sys
_sp.run([_sys.executable, str(MOLAB_REPO / "scripts/molab.py"), "stop"], check=True)
```

Đợi log BACKUP và session-status rồi download. Driver gửi SIGUSR1 cho inference, chờ tối đa
180 giây; nếu buộc kill thì các câu đã commit vẫn còn. Không đóng notebook ngay khi gửi stop.

## 8. Resume sau khi notebook cũ đã mất

Tạo/bật GPU notebook mới, thêm token, lấy đúng commit đã dùng và chạy setup.
Upload ZIP đã tải về qua Files, ví dụ đường dẫn `/marimo/molmo2-lens64-200.zip`.
Restore vào thư mục **chưa tồn tại**, rồi dùng cell resume ở bước 6:

```python
import subprocess as _sp
import sys as _sys
_sp.run([
    _sys.executable, str(MOLAB_REPO / "scripts/molab.py"), "restore",
    "--archive", "/marimo/molmo2-lens64-200.zip",
    "--output", "/marimo/videoqa-molab/runs/molmo2-lens64-200"
], check=True)
```

Không xóa thư mục đang có journal để restore đè. Nếu kết quả cũ còn, dùng luôn nó với
`--resume` hoặc chọn output mới để phục hồi ZIP. Đổi model/config/source/runtime bị từ chối.

## LLaVA và các đối chứng

Các block tương tự, thay config, tên run/log/ZIP cho đúng:

| Config | Tên run dưới `/marimo/videoqa-molab/runs/` |
|---|---|
| `configs/lvb_llava_video_lens64.json` | `llava-video-lens64-200` |
| `configs/lvb_molmo2_uniform64.json` | `molmo2-uniform64-200` |
| `configs/lvb_molmo2_focus64.json` | `molmo2-focus64-200` |
| `configs/lvb_llava_video_uniform64.json` | `llava-video-uniform64-200` |
| `configs/lvb_llava_video_focus64.json` | `llava-video-focus64-200` |

Có config 8 ảnh tương ứng. Không thay ngân sách giữa pilot và resume. Cần chạy các cặp
đối chứng mới ở cùng code MoLab này; bảo toàn run trường để báo cáo riêng.

## Kiểm chứng

55 tests CPU, gồm cgroup v1 namespace giống output người dùng, soft budget 90 trên limit160,
giữ nguyên guard Slurm/tmpfs, snapshot SQLite với transaction chưa commit, bỏ secret/video,
restore và resume đủ câu, từ chối path traversal. Chưa thực thi installer hoặc full-model
GPU trong tài khoản MoLab của người dùng; pilot là bước xác nhận thực tế.
