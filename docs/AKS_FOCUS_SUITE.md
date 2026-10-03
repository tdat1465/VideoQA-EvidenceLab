# AKS / FOCUS / Uniform / LENS trên LongVideoBench

## Sửa lỗi BLIP của pilot compare-v1 (77818–77821)

Uniform 77818 và LENS 77821 đã commit 2/200 câu. AKS 77819 và FOCUS 77820
dừng trước câu đầu với `BLIP tokenizer lacks [ENC]`. Checkpoint HF đã pin có
30.524 embedding rows nhưng tokenizer chỉ có 30.522 token BERT. Bản sửa trên
nhánh `codex/blip-itm-tokenizer-fix` đăng ký `[DEC]` rồi `[ENC]` theo LAVIS,
kiểm tra ID 30522/30523 khớp embedding đã huấn luyện. Không resize embedding
hoặc thay bằng `[CLS]`/`[UNK]`. Đây vẫn là HF port, chưa chứng minh parity LAVIS.

Giữ checkout và run `compare-v1` nguyên trạng. Vì source fingerprint thay đổi,
không resume run cũ bằng bản sửa và không ghép source khác nhau để paired compare.
Dùng checkout riêng và label `compare-v2`. Trước hết chỉ pilot AKS/FOCUS:

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-blip-fix"
git clone --single-branch --branch codex/blip-itm-tokenizer-fix \
  https://github.com/tdat1465/VideoQA-EvidenceLab.git "$REPO_ROOT"
cd "$REPO_ROOT"
git log -1 --oneline
git status --short

if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp 'Hugging Face read token: ' HF_TOKEN
  echo
fi
export HF_TOKEN

python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods aks focus \
  --label compare-v2 --max-new-samples 2 --test-only
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods aks focus \
  --label compare-v2 --max-new-samples 2 --submit
```

Sau khi cả hai pilot commit đủ 2 câu và dừng sạch, tiếp tục AKS/FOCUS và khởi
chạy Uniform/LENS mới trên cùng source. `--resume` hỗ trợ cả run cũ và phương pháp
chưa có journal. Lệnh sau chạy tới 200 câu mỗi phương pháp, không phải full val:

```bash
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v2 --resume --max-new-samples 0 --test-only
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v2 --resume --max-new-samples 0 --submit
```

Các phần bên dưới mô tả suite gốc `compare-v1`; khi dùng bản sửa, dùng checkout
và label mới ở trên. Không pull hoặc chỉnh code/config giữa các phiên resume.

Nhánh `codex/lvb-aks-focus-suite` dựa trên bản Slurm `1839719`. Mỗi phương pháp dùng
job và RUN_DIR riêng, 1 GPU, 8 CPU, 90 GiB RAM, tối đa 48 giờ, gpu01/02/03.
Submit nhiều job không bảo đảm Slurm cấp GPU đồng thời; không cố định cùng một node.
Dữ liệu được tải từng video vào RAM trên compute node rồi xóa; không tải dataset,
trọng số hoặc venv lên disk login. Mỗi job có cache và môi trường RAM riêng.
Header index nhỏ dùng chung có lock; job đầu tạo index, các job khác có thể chờ.

## Phương pháp và giao thức

- AKS tải `frame_select.py` khi chạy vào RAM; revision
  `b0b8a58fedf1d05d78151e2969cecde60e83721d`, SHA256
  `594557130aa702fb1c3eafae9df120514d36e9684443b415e0483dce14b4f149`.
  Không redistribute upstream vì chưa có LICENSE rõ ràng ở revision đã pin.
- Candidate AKS theo upstream: `step=int(fps)`, số frame `floor(total/step)`;
  clip dưới 1 giây / FPS dưới 1 có guard giữ ít nhất một candidate, step tối thiểu 1.
  BLIP chấm toàn candidate theo câu hỏi, batch 4; không giữ toàn bộ ảnh candidate.
  AKS chuẩn hóa score và dùng `t1=.8`, `t2=-100`, `depth=5`, phân bổ quota từ upstream.
  Trace ghi candidate count, số frame scorer xử lý, batch và fallback.
  Không padding khi upstream trả ít hơn ngân sách. Clip ngắn/score hằng dùng fallback
  Uniform đã ghi rõ. Các guard này không được gọi là tái lập toàn bộ bài báo.
- AKS và FOCUS dùng chung HF BLIP ITM FP32, revision/preprocessing cố định.
  Đây là port từ môi trường LAVIS; chưa khẳng định numerical parity với bài báo.
- Molmo2 và LLaVA dùng trọng số cố định, letter logits, video-only, không subtitle/audio.
  Default 200 câu stable hash seed18; full validation dùng `limit=0` (1.337 câu).
  So sánh riêng từng backbone; không ghép Molmo2 với LLaVA trong paired comparator.

[Code AKS chính thức](https://github.com/ncTimTang/AKS/tree/b0b8a58fedf1d05d78151e2969cecde60e83721d).

## Bảo toàn run LENS đang dang dở

Run `lvb-val-molmo2-lens64-200-lensv1` phải tiếp tục với checkout cũ
`VideoQA-EvidenceLab-slurm-fix`, commit `1839719`.
Runner khóa source trong resume contract, vì vậy không đổi checkout đó để chạy suite.
Các run suite dùng source mới; chạy Uniform/AKS/FOCUS/LENS mới trên cùng source để
paired comparator chấp nhận. Không dùng run LENS cũ thay cho LENS của suite mới.

## Chuẩn bị trên login01

```bash
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-aks-focus"
if [ ! -e "$REPO_ROOT" ]; then
  git clone --single-branch --branch codex/lvb-aks-focus-suite \
    https://github.com/tdat1465/VideoQA-EvidenceLab.git "$REPO_ROOT"
fi
cd "$REPO_ROOT"
git log -1 --oneline
git status --short
```

Ghi lại commit và giữ nguyên khi resume. Nếu thư mục đã tồn tại ở commit khác,
không tự pull giữa run; chọn checkout/label mới cho code mới.
Token cần có quyền đọc dataset gated LongVideoBench:

```bash
if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp 'Hugging Face read token: ' HF_TOKEN
  echo
fi
export HF_TOKEN
```

## Pilot Molmo2 trên 200 câu

Mỗi job chỉ xử lý 2 câu mới. Chạy lần lượt test-only, rồi submit:

```bash
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v1 --max-new-samples 2 --test-only

python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v1 --max-new-samples 2 --submit

squeue -u lnthanh03
```

Không có `--submit`/`--test-only` thì script chỉ in kế hoạch, không gửi job.
`--methods aks focus` chỉ gửi hai phương pháp; `uniform` là baseline cùng ngân sách.
Label phân biệt nhóm thí nghiệm. Ví dụ các RUN_DIR:

```
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-uniform64-200-compare-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-aks64-200-compare-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-focus64-200-compare-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-lens64-200-compare-v1
```

Log ở `$REPO_ROOT/logs/lvbqa-<JOB_ID>.out/.err`; mỗi submission có JSON nhỏ
`suite-<label>-<JOB_ID>.json`, không lưu token. Pilot dừng sạch với status75,
Slurm có thể báo FAILED75:0; xem summary và thông báo dừng, không bỏ qua traceback.

## Resume đủ 200 câu

Sau khi pilot từng phương pháp thành công, dùng:

```bash
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v1 --resume --max-new-samples 0 --test-only

python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --label compare-v1 --resume --max-new-samples 0 --submit
```

Script kiểm tra source/config và số câu trực tiếp từ SQLite. Nó bỏ qua run đã đủ câu;
run có journal dùng RESUME1, run chưa có journal dùng RESUME0. Runner kiểm tra tiếp
manifest/model/runtime/precision thực tế trên compute node. Run lỗi phải sửa lỗi trước,
không submit lại liên tục. Không submit thêm phiên cho run còn đang chạy/chờ Slurm.

## LLaVA và toàn validation

Thay `--backend molmo2` bằng `--backend llava_video`, label `llava-compare-v1`.
Mỗi backbone sẽ có RUN_DIR độc lập. Giữ cả pilot và resume cùng backbone/budget/label.

Full validation có config 64-frame riêng, tên `configs/lvb_<backend>_<method>64_full.json`.
Bắt đầu nhóm mới bằng label mới, không resume run200 sang full:

```bash
python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --full-validation --label full-v1 --max-new-samples 2 --test-only

python3 scripts/slurm/submit_longvideo_suite.py \
  --backend molmo2 --frames 64 --methods uniform aks focus lens \
  --full-validation --label full-v1 --max-new-samples 2 --submit
```

Sau pilot dùng chính lệnh trên với `--resume --max-new-samples 0`.
Khi 48 giờ kết thúc, dùng lại các args đó để tiếp tục job mới; không tự requeue.
Full dataset chưa được chia shard giữa nhiều job cùng phương pháp: suite hiện song song
các phương pháp/backbone, mỗi run giữ một journal. Không dùng chung RUN_DIR cho hai job.

## So sánh

Chạy khi hai run đã hoàn tất, cùng source/runtime, cùng scope và backbone:

```bash
PYTHONPATH=src python3 -m evidencelab compare \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-uniform64-200-compare-v1" \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-aks64-200-compare-v1" \
  --output "$PERSIST_ROOT/runs/videoqa/molmo2-uniform-vs-aks-compare-v1.json"
```

CLI compare chỉ cần Python chuẩn, không tải model/dataset. Đổi AKS thành FOCUS/LENS
để so các cặp khác. Báo cáo accuracy, wrong→right/right→wrong, paired bootstrap theo
video và chi phí scorer/latency/token/frame/VRAM. 200 câu là pilot nghiên cứu, không
so trực tiếp với SOTA full validation/test.

## Kiểm chứng

CPU suite, parity AKS với source upstream đã hash-check, smoke LongVideoBench AKS
với scorer giả lập (192 candidate, batch4, 64 frame). CI Linux Python3.10/3.12.
Chưa có kết quả GPU AKS/FOCUS LongVideoBench trong phiên này; pilot server xác nhận.
