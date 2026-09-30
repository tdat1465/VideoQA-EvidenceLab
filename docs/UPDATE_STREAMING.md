# Cập nhật bản streaming theo video

Bản mới giữ allocation **90 GiB ≈ 96,6 GB**, không tải toàn LongVideoBench.
RAM của venv/cache/trọng số vẫn tính chung trong job. Chỉ một video nguồn ở RAM;
HTTP Range đọc khối tối đa 4 MiB và không đọc sang video kế tiếp.
Đọc xong mọi câu của video thì đóng decoder và xóa file. Câu đã commit vẫn giữ để resume.

## Chuyển gói code mới từ Windows

Gói mới bao gồm cả tích hợp FOCUS trước đó và cập nhật streaming, dựa trên commit
`ccbafcad3a008f31450aa78ef54b397258807ade`. Code cũng có trên nhánh GitHub `codex/focus-longvideobench`.

```powershell
scp "E:\Codex\Documents\ChatGPT\KLTN\VideoQA-EvidenceLab-focus-stream.bundle" lnthanh03@login01:/media/lnthanh03/DatHa/code/
```

Thay `login01` bằng địa chỉ SSH thực tế nếu cần. Nếu chưa triển khai checkout FOCUS,
làm bước 1 trong [hướng dẫn chính](LONGVIDEOBENCH.md), dùng bundle mới này.

## Nếu đã có checkout FOCUS

Chỉ cập nhật khi không cần tiếp tục run cũ bằng checkout này. Không dùng bản mới để
resume journal tạo bởi bản trước vì fingerprint code đã đổi. Dùng RUN_DIR mới cho
cả Uniform và FOCUS để các cặp so sánh cùng code/protocol.

```bash
(
  set -euo pipefail
  export PERSIST_ROOT=/media/lnthanh03/DatHa
  REPO="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
  BUNDLE="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus-stream.bundle"
  # Có thể bỏ qua bundle và lấy code trực tiếp từ GitHub như lệnh fetch bên dưới.
  if ! git -C "$REPO" diff --quiet || ! git -C "$REPO" diff --cached --quiet; then
    echo 'Checkout có thay đổi chưa commit; giữ lại và kiểm tra trước khi cập nhật.'
    exit 1
  fi
  git -C "$REPO" fetch https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/focus-longvideobench
  git -C "$REPO" switch --detach FETCH_HEAD
  git -C "$REPO" log -1 --oneline
)
```

## Pilot mới: Molmo2 + FOCUS, 2 câu

Đã chấp nhận quyền truy cập dataset trên Hugging Face và export HF_TOKEN như hướng dẫn chính.
Tất cả tải lớn được thực hiện sau khi Slurm cấp compute node, không chạy trên login.

```bash
(
  set -euo pipefail
  : "${HF_TOKEN:?Nhập và export HF_TOKEN có quyền đọc LongVideoBench}"
  export PERSIST_ROOT=/media/lnthanh03/DatHa
  export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
  cd "$REPO_ROOT"
  export CONFIG_FILE=configs/lvb_molmo2_focus64.json
  export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200-stream"
  export RESUME=0 MAX_NEW_SAMPLES=2
  if [ -e "$RUN_DIR" ]; then
    echo "RUN_DIR đã tồn tại: $RUN_DIR; kiểm tra journal trước khi resume."
    exit 1
  fi
  python3 scripts/slurm/submit.py --workflow longvideo --test-only
  python3 scripts/slurm/submit.py --workflow longvideo --parsable
)
```

Sau pilot, giữ nguyên config/RUN_DIR, đặt `RESUME=1 MAX_NEW_SAMPLES=0` và submit lại
để đủ 200 câu. LLaVA dùng `configs/lvb_llava_video_focus64.json` với RUN_DIR riêng.
Nếu `MemoryError` xuất hiện, dừng xem log; không tăng RAM lên trên 90 GiB và không tải
dataset xuống disk login. Hiện chưa đo peak RAM thật trên GPU trường cho bản này.
