# Reproduce Coffee-Mate trên NExT-QA

Nhánh `feat/coffee-mate` bắt đầu từ main `ccbafcad3a008f31450aa78ef54b397258807ade`,
không từ nhánh AKS+Watershed. Đây là tích hợp source tác giả, không phải chuyển
selector sang Molmo/Qwen. Framework `evidencelab run` cũ không chạy Coffee-Mate;
dùng riêng `scripts/coffee_mate.py` và môi trường riêng.

## Nguồn và phạm vi

- Paper local: `../paper/COFFEE_MATE.pdf`, SHA256 trong `COFFEE_MATE_PROVENANCE.json`.
- [Source tác giả](https://github.com/yuanrr/Coffee-Mate), pinned commit
  `beaadb3925e2c6ded40e52b95190858cc89933df`. Toàn bộ 49 file source/requirements/
  instructions/license ở `vendor/coffee_mate`, nguyên byte từ clone chính thức.
- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2):
  nguồn checkpoint UMT-L/Q-former, Vicuna-v0 7B và VideoChat2 stage3.
- [NExT-QA](https://github.com/doc-doc/NExT-QA): train/val CSV và NExTVideo.

Paper: T16 candidate frames → M8 segments, k1/segment → Ts8. Reuse shared Q-former
và matching head; teacher/student cùng trọng số, teacher targets detach. Huấn luyện
3 epochs, LR2e-5 (head1e-4), LoRA rank16/alpha32/dropout0.1, lambda ramp0→0.4 trong
epoch đầu; paper dùng 2 V10032GB. Không có pretrained CoMa weights/link trong
README/source đã kiểm tra; cần training để có matching head đã học. Không dùng
CLIP hay BLIP cache thay matching head và gọi đó là reproduction.

**Chưa có accuracy GPU.** Chỉ xác minh CPU integration/source/protocol/tests.
Training/inference cần GPU + VideoChat2/Vicuna + full NExT-QA train/val + video thật.
Không bảo đảm job train xong trong 48 giờ hoặc vừa mọi GPU.

## Hai profile và các khác biệt còn lại với paper

`source` (default) giữ recipe code tác giả: batch1/rank, accumulation2, weight decay
0.02. Với 2 GPU global batch4. `paper` đổi weight decay thành 0.2 và accumulation
thành 4/world_size (microbatch1) để giữ global batch4 trên 1 hoặc 2 GPU. Tên `paper`
chỉ đánh dấu hai hyperparameters này; không có nghĩa đã tái lập chính xác mọi chi tiết.

Những khác biệt giữ lại, phải nêu khi đánh giá:

- Feature loss trong code là MSE(mean query tokens), không MSE toàn tensor như Eq.2.
- Logit distillation là softmax trên vector10 triples, không CE từng scalar riêng.
- Teacher chạy cùng module Q-former trong training mode rồi detach targets,
  không phải mạng EMA hay teacher eval/no_grad riêng.
- PEFT mặc định LoRA trên q/v projections; config không khai báo FFN targets dù
  paper mô tả attention và feedforward. Giữ cơ chế gốc để checkpoint tương thích.
- Encoder chạy lần nữa trên selected raw images; không tái sử dụng feature subset
  như sơ đồ/pseudocode paper. Vision encoder có temporal attention.
- Source có 10 instructions; paper nói5. Converter dùng5 đầu cyclic theo CSV train,
  val dùng instruction đầu. Chính sách phân phối instruction của tác giả chưa công bố.
- Upstream grader `check_ans` dùng substring của option; output rỗng có thể bị
  tính đúng do Python empty-string containment. Runner giữ grader gốc để audit
  code reproduction, đồng thời ghi `strict_option_accuracy` (chữ A–E hợp lệ đầu
  response, invalid tính sai), `invalid_option_outputs` và response để audit.
  Không diễn giải metric này như một strict MCQ grader mới.
- Final epoch02 được dùng cố định; không tìm checkpoint tốt nhất trên val.

Các patch compatibility chỉ trên bản runtime copy, vendor không sửa:

1. Chuyển target `qsn_id` lên GPU trước CE loss.
2. Validation dùng `sample_type_test=middle`, sửa factory vốn dùng `rand`.
3. Validation trong training bọc `torch.no_grad()`.
4. Negative pool trừ `{current_triple}`, không trừ `set(string)`; sort trước random
   sampling để seed tái lập. Vẫn 9 negatives từ video khác/cùng type, TP group vào TN.
5. Không thay câu decode lỗi bằng câu ngẫu nhiên; lỗi dừng run.
6. Config chuyển `method` vào `model.method`, vì code constructor đọc field này.

## Môi trường riêng (Linux/Python3.10)

Chọn storage được trường cấp quyền/quota. Không dùng chung venv của job AKS đang chạy.
Ví dụ, sau khi clone nhánh Coffee-Mate vào checkout khác:

```bash
export REPO_ROOT=/media02/lnthanh03/VideoQA-CoffeeMate
export COMA_WORK=/media02/lnthanh03/coffee-mate-work
export COMA_VENV="$COMA_WORK/venv"
export HF_HOME="$COMA_WORK/hf-cache"
export PIP_CACHE_DIR="$COMA_WORK/pip-cache"
mkdir -p "$COMA_WORK" "$HF_HOME"
cd "$REPO_ROOT"
python3 -m venv "$COMA_VENV"
source "$COMA_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-coffee-mate.txt
python -m pip install -e .
python -m pip check
```

Driver phải hỗ trợ CUDA wheel; không cài driver bằng apt. Dependencies giữ major
API gốc Transformers4.28.1/PEFT0.3.0. Không cài `apex==0.9.10dev` từ PyPI (không
phải NVIDIA Apex); AdamW recipe không yêu cầu Apex. PyAV được import khi chọn backend
av; NExT-QA dùng decord nên không cần build PyAV/FFmpeg dev libraries. `imageio`
được cài cho import chung của video_utils. GPU runtime chưa được xác minh.

Prefetch BERT vì code gốc yêu cầu `local_files_only=True`:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("bert-base-uncased")'
```

Nguồn checkpoint initialization trong bảng Model VideoChat2:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, HfApi
repo = 'OpenGVLab/videochat'
revision = HfApi().model_info(repo).sha
print('PIN THIS REVISION:', revision)
for filename in ('umt_l16_qformer.pth', 'videochat2_7b_stage3.pth'):
    print(filename, hf_hub_download(repo, filename, revision=revision))
PY
```

Ghi lại hai đường dẫn in ra. **Vicuna-7B-v0 phải chuẩn bị riêng** theo hướng dẫn
[VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) và
[Vicuna versions](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md),
tôn trọng điều kiện truy cập/license LLaMA. Không tự thay bằng Vicuna1.5/Mistral/Qwen
rồi so số paper. Đường dẫn `--vicuna` phải chứa config/tokenizer và model shards.
Prepare kiểm tra kiến trúc7B, chưa thể chứng minh v0 chỉ từ config.json; cần provenance
nguồn weights. SHA256 toàn bộ files ghi vào contract để resume không đổi weights.

## Dữ liệu và chuẩn bị recipe

Tải NExT-QA train.csv/val.csv/NExTVideo theo nguồn chính thức; cần cả train videos,
không chỉ val subset. Repo có `scripts/fetch_nextqa_annotations.py` tải annotation;
`stage_nextqa_ram.py` cũ chỉ cho inference subset, không dùng để train CoMa full.
Video có thể trong subfolder (`1164/3238737531.mp4`), converter tìm theo stem và
reject ambiguous stem. Không tải/giải nén full dataset tự động khi submit.

```bash
python scripts/fetch_nextqa_annotations.py --split train --output "$COMA_WORK/annotations/train"
python scripts/fetch_nextqa_annotations.py --split val --output "$COMA_WORK/annotations/val"

export COMA_PREPARED="$COMA_WORK/nextqa-source-seed42"
python scripts/coffee_mate.py prepare \
  --train-csv "$COMA_WORK/annotations/train/train.csv" \
  --val-csv "$COMA_WORK/annotations/val/val.csv" \
  --video-root /path/to/NExTVideo \
  --umt /path/to/umt_l16_qformer.pth \
  --vicuna /path/to/vicuna-7b-v0 \
  --stage3 /path/to/videochat2_7b_stage3.pth \
  --profile source --method coma --seed 42 --world-size 2 \
  --output "$COMA_PREPARED"
```

Prepare không tải model/video hay train. Nó kiểm tra đủ video, đáp án, duplicate IDs,
train/val question overlap và negative pools; checksum toàn bộ video/weights có thể
tốn thời gian đọc I/O. Có `train.json`, `val.json`, `config.json`, `contract.json`,
`runtime/` tại output. Không chỉnh config/runtime sau prepare. Đổi seed/profile tạo
directory mới. Nếu RAM/VRAM chỉ đủ1GPU, `--world-size 1 --profile paper` và submit
`--gres=gpu:1`; vẫn ghi rõ khác tài nguyên paper2GPU.

## Submit training rồi evaluation

Kiểm tra các paths/môi trường đã tồn tại trước submit; job không tự bootstrap chúng.

```bash
sbatch --export=ALL scripts/slurm/coffee_mate.sbatch
squeue -u "$USER"
```

Job xin2GPU cùng node,64G RAM,48h. RAM không thay storage. Các variables REPO_ROOT,
COMA_VENV, COMA_PREPARED, HF_HOME phải export ở shell submit. School account/partition
và quota theo quyền được cấp, không thay thư mục job AKS77948 đang chờ/chạy.
Script train3epochs qua `torchrun --standalone`, rồi eval checkpoint `ckpt_02.pth`
trên1GPU trong cùng allocation. Trong lúc eval GPU thứ2 vẫn được giữ bởi allocation;
có thể submit evaluation riêng1GPU để dùng tài nguyên hiệu quả hơn:

```bash
# Trong allocation1GPU/môi trường đã activate:
python scripts/coffee_mate.py evaluate --prepared "$COMA_PREPARED" \
  --checkpoint "$COMA_PREPARED/training/ckpt_02.pth" \
  --output "$COMA_PREPARED/evaluation-final"
```

Resume sau timeout: `export COMA_RESUME=1` rồi submit lại. Training chỉ resume từ
**epoch hoàn thành**, do source checkpointing gốc; không mid-epoch exact resume.
Checkpoint gốc chưa ghi atomically, kiểm tra file hoàn chỉnh nếu bị ngắt lúc save.
Nếu bị ngắt trước epoch đầu hoàn tất chưa có checkpoint để resume: tạo run mới.
Không cam kết một session48h hoàn tất train. Checkpoint mỗi epoch giữ riêng, cần storage.
Evaluation ghi atomically từng câu; resume đúng config/code/environment/checkpoint,
không bỏ qua lỗi; thiếu trained parameters của matching head/Q-former/LoRA bị reject.

## Báo cáo và ba seed

`training/`: logs, ckpt_00/01/02.pth, optimizer/scheduler states.
`evaluation-final/`: eval_contract.json, predictions từng câu, predictions.jsonl,
summary.json. Record có response/gold/type, candidate scores, frame IDs thực sự,
selected indices, FPS, generation time và peakVRAM. Không đưa gold vào generate
hay scoring; gold chỉ dùng grade sau generation.

```bash
python scripts/coffee_mate.py report --output "$COMA_PREPARED/evaluation-final"
```

Overall accuracy là question-weighted; breakdown C/T/D và từng type. Report đánh dấu
coverage/incomplete; thời gian không gồm model load/prepare/decode. Đầu tiên kiểm tra
output/VRAM và các response không hợp lệ trước kết luận. Paper report mean±std3seed;
chạy ba preparations riêng (vd42,43,44; tác giả không công bố IDs seed), tính mean/std
của full-validation overall từng run. Không coi ba seed là ba subset khác nhau.
Sau ba run hoàn thành, tổng hợp mean và sample std (ddof1):

```bash
python scripts/coffee_mate.py aggregate --evaluations \
  "$COMA_WORK/nextqa-source-seed42/evaluation-final" \
  "$COMA_WORK/nextqa-source-seed43/evaluation-final" \
  "$COMA_WORK/nextqa-source-seed44/evaluation-final" \
  --output "$COMA_WORK/nextqa-coffee-mate-three-seeds.json"
```

Đối chứng M2L-SD dùng `--method m2l-sd` nhưng **train riêng**, không chỉ switch method
trên cùng CoMa checkpoint rồi gọi là số paper. Baseline VideoChat2† cần recipe
finetuning không distillation riêng; runner chưa triển khai baseline đó.

Không đo LongVideoBench bằng recipeNExT-QA rồi đối chiếu bảng paper. Qwen2-VL
plug-in trong paper không có implementation/config trong source hiện tại; phạm vi
nhánh này là official VideoChat2 CoMa trênNExT-QA theo yêu cầu người dùng.

## Kiểm chứng local

`PYTHONPATH=src python -m unittest discover -s tests -v`; test source hashes,
runtime patches, CSVformat/negatives/prompt không gold leak, recipeprofiles và
partial report. Tests fixtures không có model accuracy thật. Bash `-n` trên job
template đã qua. Dependencies GPU chưa được cài/chạy trên server trong phiên này.

Không push trong bước triển khai này. Push từ máy bạn:

```bash
cd /mnt/e/KL1/coffee-mate
git push -u origin feat/coffee-mate
```

Server dùng checkout riêng:

```bash
git clone --branch feat/coffee-mate https://github.com/tdat1465/VideoQA-EvidenceLab.git /media02/lnthanh03/VideoQA-CoffeeMate
```
