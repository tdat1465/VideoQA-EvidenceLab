# LENS: sửa text BLIP vượt context và chạy tiếp trên server trường

Job 78213 (`compare-v2`) lưu 170/200 câu rồi dừng: BLIP nhận 590 token trong khi
text encoder chỉ có 512 position embeddings. Đây không phải lỗi thiếu RAM.
AKS job 78211 đã đủ 200 câu; Uniform/FOCUS cũng đã hoàn tất. Giữ các run đó nguyên trạng.

## Bản sửa

Nhánh `codex/lens-blip-text-limit` dựa trên bản giảm RAM `1dcb169`:

- LENS dùng `truncation=True`, giới hạn nhỏ hơn giữa context của text encoder và
  `tokenizer.model_max_length`, gồm cả token đặc biệt. Checkpoint BLIP đã pin có
  giới hạn 512; cắt ở phía phải. Không thay đổi weights/embedding.
- Chỉ giới hạn đầu vào BLIP dùng để chấm frame. Lượt phân bổ ngân sách, CLIP mask
  và prompt trả lời vẫn nhận query đầy đủ theo giao thức hiện có.
- Prediction ghi `lens_blip_text_policy`, `lens_blip_max_length`,
  `lens_blip_query_tokens`, `lens_blip_query_truncated`, `lens_blip_dropped_tokens`.
  Query dài có thể mất lựa chọn ở cuối trong bước BLIP; đây là chính sách xử lý
  context của bản tích hợp, không khẳng định tái lập nguyên recipe/accuracy paper.
- `lens_implementation` đổi thành `upstream-lazy-1fps-ssim32-blip-limit-v2`;
  source/config fingerprint thay đổi. Không resume `compare-v2` bằng nhánh này,
  không sửa contract để nhập 170 prediction cũ vào run mới.
- Tests offline dùng HF tokenizer/processor và BLIP nhỏ có context 512: tái hiện
  lỗi 590 token, xác minh sau cắt forward chạy và token đặc biệt đúng; query ngắn
  và query đúng 512 token giữ nguyên tensor input. Không tải model lớn khi test.

64 tests CPU đã chạy qua với Transformers 4.57.1 / tokenizers 0.22.1 như server.
Bản sửa chưa được xác minh bằng full checkpoint trên GPU trường; run mới là bước xác minh đó.

Mặc định LongVideoBench xin **64 GiB (~68,7 GB), 1 GPU, 8 CPU, 48 giờ**;
cho phép gpu01/02/03 và luôn loại gpu04. Có `--mem-gib 80` / `90` nếu cần tăng
dự phòng. Mọi dataset/weights/env/cache lớn vẫn nằm trong RAM trên compute node;
staging một video rồi xóa, không giữ cả split. 64 GiB chưa chứng minh đủ cho mọi
video của full validation. LLaVA chưa có số đo riêng, nên pilot với 90 GiB.

## 1. Chuẩn bị checkout riêng trên login01

Không source env NExT-QA cũ. Nhập token read đã được cấp quyền dataset; không lưu
token vào Git hoặc log. Clone chỉ tải code nhỏ vào root bền vững.

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-lens-textfix"
export PYTHON_BIN=/usr/bin/python3
export RAM_BASE=/dev/shm
export MIN_RAM_FREE_GIB=64

if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp 'Hugging Face read token: ' HF_TOKEN
  echo
fi
export HF_TOKEN

(
  set -euo pipefail
  trap 'echo "Dừng tại dòng $LINENO; xem lỗi phía trên." >&2' ERR
  if [ ! -e "$REPO_ROOT" ]; then
    git clone --single-branch --branch codex/lens-blip-text-limit \
      https://github.com/tdat1465/VideoQA-EvidenceLab.git "$REPO_ROOT"
  fi
  cd "$REPO_ROOT"
  git log -1 --oneline
  test "$(git branch --show-current)" = codex/lens-blip-text-limit
  test -z "$(git status --porcelain)"
)
```

Giữ commit của checkout mới cố định sau khi submit, kể cả qua các phiên resume.
Checkout `VideoQA-EvidenceLab-blip-fix` và `VideoQA-EvidenceLab-64g` vẫn giữ nguyên.

## 2. Chạy LENS mới đủ 200 câu

Dùng label mới `lens-textfix-v1`; kết quả:
`/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-lens64-200-lens-textfix-v1`.
Không dùng label `compare-v2` của run 170 câu. `--max-new-samples 0` chạy tới
giới hạn 200 trong config, không phải full validation. Chỉ submit một phương pháp
mỗi lần vì các job song song trước đây có gặp giới hạn QoS của trường.

```bash
(
  set -euo pipefail
  trap 'echo "Dừng tại dòng $LINENO; xem lỗi phía trên." >&2' ERR
  cd "$REPO_ROOT"
  : "${HF_TOKEN:?Nhập và export HF_TOKEN}"
  python3 scripts/slurm/submit_longvideo_suite.py \
    --backend molmo2 --frames 64 --methods lens --label lens-textfix-v1 \
    --max-new-samples 0 --mem-gib 64 --test-only
  python3 scripts/slurm/submit_longvideo_suite.py \
    --backend molmo2 --frames 64 --methods lens --label lens-textfix-v1 \
    --max-new-samples 0 --mem-gib 64 --submit
  squeue -u lnthanh03
)
```

Nếu một phiên dừng trước 200 câu, kiểm tra lý do trước khi tiếp tục; dùng **cùng
checkout/config/label**, thêm `--resume` vào cả hai lệnh trên. Không thay đổi code
hoặc config khi resume. Job mới sẽ dựng lại môi trường và tải weights vào RAM.
Nếu thiếu RAM, tăng `--mem-gib 80` hoặc `90` ở phiên tiếp theo; không phải đổi config.

## 3. Kiểm tra sau job

```bash
RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-lens64-200-lens-textfix-v1"
python3 -m json.tool "$RUN_DIR/summary.json"
python3 - "$RUN_DIR/predictions.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
cut = [r for r in rows if r.get('lens_blip_query_truncated')]
print('Completed:', len(rows), 'BLIP-truncated queries:', len(cut))
for r in cut:
    print(r['id'], 'tokens:', r['lens_blip_query_tokens'],
          'limit:', r['lens_blip_max_length'], 'dropped:', r['lens_blip_dropped_tokens'])
PY
```

Xác nhận `complete=true`, `completed_count=200`, không có traceback BLIP hoặc OOM.
Đọc cờ truncation để biết câu nào thực sự áp dụng bản sửa. Số cgroup `current_bytes`
sau mỗi câu chỉ là quan sát tại thời điểm đó, không phải peak toàn job.

## 4. Xác nhận trên full validation

Khi LENS mới đủ 200 và ổn định, ưu tiên Uniform/AKS trên toàn bộ 1.337 câu.
Dùng cùng checkout mới, cùng label `full-textfix-v1`, config full có sẵn; không
đổi limit của một run 200 đang tồn tại. Có thể chạy thêm FOCUS/LENS trên cùng bản
code sau đó. Các phép paired chính thức cần cùng source/runtime/dataset/protocol;
không ghép LENS mới với các run `compare-v2` để vượt kiểm tra fingerprint.

Ví dụ Uniform full; sau khi hoàn tất, thay `--methods uniform` bằng `--methods aks`.
Chạy từ cùng checkout đã dùng ở bước 2, giữ nguyên revision model và precision.

```bash
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform --label full-textfix-v1 \
  --full-validation --max-new-samples 0 --mem-gib 64 --test-only
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform --label full-textfix-v1 \
  --full-validation --max-new-samples 0 --mem-gib 64 --submit
```

Nếu full run cần nhiều phiên 48 giờ, thêm `--resume` ở phiên sau, giữ nguyên label.
Run đã đủ selected_count được suite bỏ qua; không nộp trùng một run đang chạy/chờ.
