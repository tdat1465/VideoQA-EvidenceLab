# Chạy FOCUS với Molmo2 và LLaVA trên server trường

Workflow mới: **LongVideoBench validation, video-only, không phụ đề/audio**.
Giữ nguyên các run NExT-QA đã hoàn tất và checkout cũ để có thể tiếp tục các run cũ khi cần.
Mỗi job vẫn dùng **1 GPU, 8 CPU, 90 GiB RAM, tối đa 48 giờ**, chỉ gpu01/gpu02/gpu03.

Luồng streaming: **Hugging Face → các khối HTTP Range tối đa 4 MiB → một video trong
tmpfs → xử lý các câu hỏi của video → đóng decoder, xóa video và bộ đệm → video tiếp theo**.
Không tải trước video kế tiếp. FOCUS cần seek ngẫu nhiên nên giữ nguyên video đang dùng
đến khi xử lý hết các câu của nó; không thể xóa từng khối file trước khi FOCUS đọc xong.
Chỉ metadata/index, code và kết quả nhỏ nằm trên ổ bền vững.

Nếu đã triển khai bản trước, xem [lệnh cập nhật streaming](UPDATE_STREAMING.md).

## 1. Đưa code mới lên server

Bản bàn giao có file `VideoQA-EvidenceLab-focus-stream.bundle` chứa commit mới dựa trên
`ccbafcad3a008f31450aa78ef54b397258807ade`. Đây là Git bundle chỉ chứa code, không có dataset/model.
**Chưa cần `git pull`; bundle không có nghĩa code đã được push lên GitHub.**

Chạy trong **PowerShell trên máy Windows**, dùng cùng địa chỉ SSH bạn đang dùng để vào trường
(thay `login01` bằng hostname/IP thực tế nếu tên này chỉ phân giải trong mạng trường):

```powershell
scp "E:\Codex\Documents\ChatGPT\KLTN\VideoQA-EvidenceLab-focus-stream.bundle" lnthanh03@login01:/media/lnthanh03/DatHa/code/
```

Sau đó chạy trên **login01** để tạo checkout riêng. Không yêu cầu checkout cũ sạch;
file untracked `1` nếu còn sẽ không bị xóa hoặc chép vào checkout mới.

```bash
(
  set -euo pipefail
  export PERSIST_ROOT=/media/lnthanh03/DatHa
  OLD_REPO="$PERSIST_ROOT/code/VideoQA-EvidenceLab"
  NEW_REPO="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
  BUNDLE="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus-stream.bundle"

  git -C "$OLD_REPO" cat-file -e ccbafcad3a008f31450aa78ef54b397258807ade^{commit}
  git -C "$OLD_REPO" bundle verify "$BUNDLE"
  if [ -e "$NEW_REPO" ]; then
    echo "Checkout mới đã tồn tại: $NEW_REPO. Kiểm tra git status ở đó; không tạo chồng."
    exit 1
  fi
  git -C "$OLD_REPO" fetch "$BUNDLE" codex/focus-longvideobench
  git -C "$OLD_REPO" worktree add --detach "$NEW_REPO" FETCH_HEAD
  git -C "$NEW_REPO" log -1 --oneline
  git -C "$NEW_REPO" status --short
)
```

Tất cả lệnh tiếp theo dùng checkout **`VideoQA-EvidenceLab-focus`** này.

## 2. Quyền truy cập LongVideoBench

Nguồn chính thức hiện là gated dataset. Đăng nhập trình duyệt vào
[LongVideoBench trên Hugging Face](https://huggingface.co/datasets/longvideobench/LongVideoBench),
chấp nhận điều kiện truy cập, rồi tạo token có quyền đọc dataset đó.

Trên login01 nhập token kín, không ghi token vào file env, lệnh có giá trị literal, log hay chat:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
cd "$REPO_ROOT"
```

Không cần `hf download` trên login. Job kiểm tra annotation và hỗ trợ HTTP Range trước khi
tải Torch/model. Nếu bị 401/403, kiểm tra quyền token trên website rồi submit lại.
Token được Slurm nhận qua môi trường job; code không in hoặc lưu token vào kết quả.

## 3. Pilot FOCUS + Molmo2: 2 câu

Không nạp `videoqa-90g.env` cũ vì nó trỏ tới các run NExT-QA.
`MAX_NEW_SAMPLES=2` chỉ giới hạn phiên pilot; cấu hình vẫn chọn cùng 200 câu.

```bash
(
  set -euo pipefail
  : "${HF_TOKEN:?Nhập và export HF_TOKEN theo bước 2}"
  export PERSIST_ROOT=/media/lnthanh03/DatHa
  export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
  cd "$REPO_ROOT"
  export CONFIG_FILE=configs/lvb_molmo2_focus64.json
  export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200"
  export RESUME=0 MAX_NEW_SAMPLES=2
  if [ -e "$RUN_DIR" ]; then
    echo "RUN_DIR đã tồn tại: $RUN_DIR. Kiểm tra trước khi chạy/resume."
    exit 1
  fi
  python3 -m json.tool "$CONFIG_FILE"
  python3 scripts/slurm/submit.py --workflow longvideo --test-only
  JOB_ID=$(python3 scripts/slurm/submit.py --workflow longvideo --parsable)
  echo "FOCUS Molmo2 pilot JOB_ID=$JOB_ID"
  squeue -j "${JOB_ID%%;*}"
)
```

Theo dõi (thay `JOB_ID` bằng số thực, không giữ nguyên chữ):

```bash
squeue -u lnthanh03
tail -n 60 /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-focus/logs/lvbqa-JOB_ID.out
tail -n 60 /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-focus/logs/lvbqa-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

Pilot dừng sạch với exit **75**, nên Slurm có thể ghi `FAILED 75:0`; kiểm tra thông báo
`Session stopped cleanly` và `completed_count: 2`, không kết luận lỗi chỉ từ state/`.err`.
Lần đầu có bước `Indexed ... tar members` để lập index nhỏ; chưa phải suy luận.

```bash
python3 -m json.tool /media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-focus64-200/summary.json
```

Sau khi pilot đúng, resume để đủ 200 câu:

```bash
(
  set -euo pipefail
  export PERSIST_ROOT=/media/lnthanh03/DatHa
  export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
  cd "$REPO_ROOT"
  export CONFIG_FILE=configs/lvb_molmo2_focus64.json
  export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200"
  export RESUME=1 MAX_NEW_SAMPLES=0
  python3 scripts/slurm/submit.py --workflow longvideo --test-only
  python3 scripts/slurm/submit.py --workflow longvideo --parsable
)
```

`MAX_NEW_SAMPLES=0` nghĩa không giới hạn số câu mới trong phiên. Không đổi code/config
trước khi resume. Nếu lỗi xảy ra **trước khi có `results.sqlite3`**, dùng `RESUME=0` để thử
lại cùng RUN_DIR (không cần xóa log); nếu đã có journal, dùng `RESUME=1`.

## 4. Ba cấu hình còn lại

| Cấu hình | RUN_DIR dưới `$PERSIST_ROOT/runs/videoqa/` |
|---|---|
| `configs/lvb_molmo2_uniform64.json` | `lvb-val-molmo2-uniform64-200` |
| `configs/lvb_llava_video_focus64.json` | `lvb-val-llava-video-focus64-200` |
| `configs/lvb_llava_video_uniform64.json` | `lvb-val-llava-video-uniform64-200` |

Dùng block pilot ở bước 3, thay đúng **hai biến** `CONFIG_FILE` và `RUN_DIR` theo bảng.
Đặc biệt cần pilot LLaVA trước khi chạy đủ 200 câu. Ví dụ phần thiết lập LLaVA FOCUS:

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
cd "$REPO_ROOT"
export CONFIG_FILE=configs/lvb_llava_video_focus64.json
export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-llava-video-focus64-200"
export RESUME=0 MAX_NEW_SAMPLES=2
python3 scripts/slurm/submit.py --workflow longvideo --test-only
python3 scripts/slurm/submit.py --workflow longvideo --parsable
```

Khi pilot đạt, giữ nguyên hai đường dẫn, đổi `RESUME=1 MAX_NEW_SAMPLES=0` và submit tiếp.
Mỗi cấu hình có journal riêng. Các job mới tải lại môi trường/trọng số vào RAM;
index archive nhỏ được dùng lại từ `$PERSIST_ROOT/metadata/longvideobench/`.

## 5. So sánh khi mỗi cặp đã đủ 200 câu

Lệnh so sánh chỉ cần Python chuẩn, chạy được trên login và không tải model:

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"
python3 -m evidencelab compare \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-uniform64-200" \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200" \
  --output "$PERSIST_ROOT/runs/videoqa/lvb-molmo2-uniform-vs-focus-200.json"
python3 -m evidencelab compare \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-llava-video-uniform64-200" \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-llava-video-focus64-200" \
  --output "$PERSIST_ROOT/runs/videoqa/lvb-llava-video-uniform-vs-focus-200.json"
```

Có accuracy tổng, theo question category và duration group; paired sai→đúng/đúng→sai;
bootstrap theo video; token, frame, số frame BLIP đã chấm, latency và VRAM.
So sánh trong từng backbone. Không dùng comparator này ghép Molmo2 với LLaVA.
`max_input_tokens`, model/revision, source/runtime, precision thực tế, annotation/ID và
hash video phải tương thích. Các node khác nhau có thể làm latency khác nhau.

## 6. Chạy đủ validation sau pilot/thăm dò

Tạo config mới, không sửa config đang có run cần resume. Ví dụ Molmo2 FOCUS:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-focus
python3 - <<'PY'
import json
from pathlib import Path
for model in ('molmo2', 'llava_video'):
    for method in ('uniform', 'focus'):
        src = Path(f'configs/lvb_{model}_{method}64.json')
        dst = Path(f'configs/lvb_{model}_{method}64_full.json')
        data = json.loads(src.read_text())
        data['limit'] = 0
        with dst.open('x') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        print(dst)
PY
```

Sau đó chọn config `_full.json`, dùng RUN_DIR mới có hậu tố `-full`, `RESUME=0` và
`MAX_NEW_SAMPLES=0`. Full validation có **1.337 câu**, không phải test.
Mẫu 200 nằm trong full validation, nên full validation không phải tập holdout độc lập
nếu đã dùng 200 câu để chọn phương pháp/siêu tham số; cần ghi rõ khi báo cáo nghiên cứu.

## Giao thức và giới hạn đã biết

- FOCUS core gốc, SHA-verified; không giới hạn vào pool 64 ứng viên. BLIP-large ITM
  tìm trên toàn timeline; ngân sách 64 là số frame cuối vào VLM, không phải số frame BLIP chấm.
  Không tự bù frame nếu upstream trả ít hơn ngân sách; số thực tế có trong predictions.
- BLIP chạy FP32, batch 4 trên cùng GPU; bản Transformers + preprocess LAVIS.
  Chưa xác minh parity số học với LAVIS của paper. Seed chọn frame gắn với question ID
  để resume không phụ thuộc node/rank/thứ tự; khác seed theo rank của script upstream.
- LLaVA dùng model chính thức `lmms-lab/LLaVA-Video-7B-Qwen2`, video modality, template
  `qwen_1_5`, SigLip được pin, pooling bilinear stride 2 và SDPA cho LM. Molmo2 dùng backend
  native video hiện có. Chỉ đọc logits token chữ cái ở bước đầu; không parse đáp án tự do.
  Input tokens của LLaVA tính **sau khi chèn visual embeddings**, vượt giới hạn sẽ dừng.
- Uniform dùng các chỉ số đều trên toàn video theo EvidenceLab. Chưa khẳng định recipe
  này trùng mọi sampling/prompt/subtitle/scoring setting của lmms-eval trong paper.
- Thời gian suy luận gồm decode, chọn frame/BLIP và VLM. Tải video + hash được báo riêng;
  tải/nạp model và lập index không được tính là latency suy luận. Peak VRAM allocated là
  số PyTorch ghi, khác tổng `nvidia-smi`. Đọc RAM từ cgroup v1/v2 của job khi có
  giới hạn hữu hạn phù hợp với allocation; không lấy số liệu toàn node làm RAM của job.
- Archive chính thức gồm 31 phần (~150,47 GiB). HTTP Range chỉ đọc header để lập index,
  sau đó tải từng video cần dùng. Một video xong mới chuyển video tiếp; không tải/ghép cả tar.
  Kiểm tra Content-Range, giới hạn video 8 GiB, dự phòng tmpfs/cgroup 6 GiB, không fallback disk.
  Kiểm tra lại headroom trước mỗi khối tải, sau khi tải và trước decode/VLM.
  Bộ đệm đọc không vượt sang video kế tiếp và được xóa sau tải. Khi gặp lỗi/stop,
  phần `.partial` được dọn; video hiện tại được giải phóng trước khi xuất báo cáo.
  Log `Released RAM video: ...` và event `video_evicted` xác nhận bước dọn cuối mỗi video.
  Nếu không đọc được cgroup, chỉ đo được tmpfs; không coi đây là xác nhận peak RAM tổng.
  Không tự giảm giới hạn 90 GiB dựa trên VRAM.
- **Chưa chạy full-model trên GPU trường cho workflow mới.** Cần pilot thực tế để xác minh
  VRAM, decode video dài, quyền tải và HTTP Range qua mạng compute node. Tập dữ liệu có gate
  nên phiên phát triển chưa đọc được annotation/video thực bằng tài khoản đã được cấp quyền.

Nguồn: [FOCUS](https://github.com/NUS-HPC-AI-Lab/FOCUS),
[LLaVA-Video model](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2),
[LongVideoBench loader](https://github.com/longvideobench/LongVideoBench).
