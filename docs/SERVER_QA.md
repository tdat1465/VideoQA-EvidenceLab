# Chạy QA gốc và Watershed trên server

Nhánh `feat/aks-watershed-selector` có hai đường evaluation: adapter LLaVA/lmms-eval
cũ trong `AKS_WATERSHED.md`, và runner độc lập Qwen2.5-VL trong hướng dẫn này.
Runner mới dùng cùng checkpoint, prompt, greedy generation, pixel cap và số frame
thực tế cho hai selector, giữ nguyên BLIP score cache và cơ chế AKS đã kiểm thử.
Đây là ablation selector với downstream model Qwen; không so trực tiếp accuracy
với bảng LLaVA trong paper. Không chạy full LENS làm baseline.

**Trạng thái kiểm chứng:** selector/validation/report/download extraction được
kiểm thử CPU; chưa chạy tải toàn bộ dataset/checkpoint hoặc inference trên GPU.
Các accuracy trong test chỉ là fixture, không phải kết quả thực nghiệm.

## 1. Push và clone

Tạo repository hoặc fork trong tài khoản GitHub của bạn. Remote hiện tại trỏ
tới tác giả AKS, vì vậy đổi remote trước khi push:

```bash
git remote rename origin upstream
git remote add origin https://github.com/tdat1465/VideoQA-EvidenceLab.git
git push -u origin feat/aks-watershed-selector
```

Trên server clone đầy đủ, không dùng sparse checkout, để có `outscores/`:

```bash
git clone --branch feat/aks-watershed-selector https://github.com/tdat1465/VideoQA-EvidenceLab.git aks-watershed
cd aks-watershed
```

## 2. Chọn storage và tạo môi trường

Thông tin cụm đã biết: Ubuntu 22.04/Python 3.10, Slurm partition `batch`, node
khai báo 4 hoặc 8 GPU; loại GPU/VRAM/account/quota chưa biết. Job template xin 1 GPU,
4 CPU, RAM 32G và 1 ngày; trường có thể yêu cầu thêm `--account` hoặc sửa RAM.
Các giá trị này là yêu cầu tài nguyên, không đảm bảo cụm cấp được.

Thay đường dẫn dưới đây bằng thư mục bạn được trường cấp quyền/quota.
Không suy ra quyền hoặc quota từ `/media02` còn 139GB dùng chung. Không chạy
inference trên login node; setup/download cũng phải tuân theo chính sách trường.

```bash
export WORK=/path/to/approved/storage/lnthanh03/aks
mkdir -p "$WORK"
export HF_HOME="$WORK/hf-cache"
export VENV="$WORK/venv"
export PIP_CACHE_DIR="$WORK/pip-cache"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
```

Cài wheel PyTorch tương thích driver GPU của cụm; không cài driver bằng apt.
Ví dụ CUDA 11.8, nếu driver cụm hỗ trợ wheel này:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-server.txt
python -m pip check
python -m unittest discover -s tests -v
```

Wheel khác: xem [PyTorch versions](https://pytorch.org/get-started/previous-versions/).
Dependencies pin theo API [Transformers 4.51.3 Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_5_vl).
Checkpoint mặc định `Qwen/Qwen2.5-VL-3B-Instruct`; có thể dùng 7B khi đủ VRAM.
Không đảm bảo 3B/K64 vừa mọi GPU; nếu OOM, tạo experiment mới với K/pixel cap phù hợp
cho cả hai selector. Không âm thầm giảm frame hay offload từng selector khác nhau.

## 3. Chuẩn bị VideoMME

Mặc định script chỉ lấy metadata và ghi kế hoạch dung lượng, chưa tải video:

```bash
export VIDEO_ROOT="$WORK/VideoMME"
python prepare_videomme.py --root "$VIDEO_ROOT" --limit 6
```

Xem `$VIDEO_ROOT/download_plan.json`. Dataset public hiện gồm nhiều ZIP chunk.
Script lấy revision SHA từ Hub, tải lần lượt và chỉ giải nén video thuộc các câu
cần chạy, dừng khi đủ. **Một subset nhỏ vẫn có thể phải tải tất cả archive** vì
chưa biết video nằm chunk nào; archive giữ lại để resume. Dung lượng đỉnh gồm
archive + video giải nén + weights/cache/env, có thể vượt phần trống hiện tại.
Kiểm tra filesystem free không thay thế kiểm tra quota. Nếu trường đã có dataset,
dùng đường dẫn đó, bỏ toàn bộ bước download.

Thực sự tải subset bằng lệnh riêng:

```bash
python prepare_videomme.py --root "$VIDEO_ROOT" --limit 6 --download
```

Script không tải subtitles, không dùng audio. Video ở `$VIDEO_ROOT/data/<videoID>.mp4`.
Annotations và BLIP scores đã có trong repo, dùng đúng thứ tự cache AKS.
Metadata pin nguồn [Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME/tree/main);
script không tự tải mô hình khi chuẩn bị dataset. Nếu job GPU không có mạng,
prefetch checkpoint ở máy được phép có mạng và dùng cùng `HF_HOME`:

```bash
python -c 'from huggingface_hub import HfApi, snapshot_download; repo="Qwen/Qwen2.5-VL-3B-Instruct"; rev=HfApi().model_info(repo).sha; print(rev); snapshot_download(repo, revision=rev)'
```

Nếu compute node không có mạng, đặt `export OFFLINE=1` và
`export MODEL_REVISION=<SHA40_đã_in_ra>` trước submit. Runner dùng cache với
`local_files_only=True`, không gọi Hub metadata. Cùng `HF_HOME` phải nhìn thấy
toàn bộ checkpoint đã tải; CLI tương ứng là `--offline --revision <SHA40>`.

## 4. Smoke test rồi chạy đầy đủ

Tạo subset 6 câu, K8 với depth3 (quota terminal có thể bằng 0 khi K quá nhỏ so với depth):

```bash
export EXPERIMENT="$WORK/experiments/videomme-k8-smoke"
python compare_selectors.py \
  --score_path outscores/videomme/blip/scores.json \
  --frame_path outscores/videomme/blip/frames.json \
  --dataset_name videomme --max_num_frames 8 --all_depth 3 --limit 6 \
  --annotation_path datasets/videomme/include_frame_idx.json \
  --video_root "$VIDEO_ROOT" --output_dir "$EXPERIMENT"
python run_qa_pair.py --experiment "$EXPERIMENT" --video-root "$VIDEO_ROOT" --preflight-only
sbatch --export=ALL scripts/server_pair.sbatch
```

Nếu trường yêu cầu account: `sbatch --account=YOUR_GRANTED_ACCOUNT --export=ALL scripts/server_pair.sbatch`.
Theo dõi `squeue -u "$USER"` và `aks-qa-<jobid>.log`. Job tự ghi GPU/driver,
Python packages, checkpoint SHA, video SHA256, code hashes và cấu hình.
Mô hình được tải từ Hub khi job bắt đầu nếu chưa có cache. Một process/model dùng
chung cho cả hai selector; không cần lmms-eval/LLaVA/flash-attn cho đường này.

Để chạy đầy đủ, tải video đủ bộ (bỏ `--limit` ở prepare), tạo experiment mới
bỏ `--limit`, dùng `--max_num_frames 64 --all_depth 5`; submit cùng job template.
Đổi model bằng `export MODEL=Qwen/Qwen2.5-VL-7B-Instruct` trước submit.
Pixel cap của job có thể đặt bằng `export MAX_PIXELS=200704`.
Để tùy chỉnh pixel cap, chạy trực tiếp trong job của bạn:

```bash
python run_qa_pair.py --experiment "$EXPERIMENT" --video-root "$VIDEO_ROOT" \
  --model Qwen/Qwen2.5-VL-3B-Instruct --max-pixels 200704
python analyze_qa_pair.py --experiment "$EXPERIMENT"
```

Resume sau timeout/interruption với cùng code, model, videos, dependencies, args:

```bash
export RESUME=1
sbatch --export=ALL scripts/server_pair.sbatch
```

Đã xong từng cặp thì không chạy lại; cặp bị ngắt chưa ghi atomically sẽ chạy lại cả hai.
OOM/decode error dừng job, ghi `qa_pair/last_error.json`; không bỏ câu lỗi hay đoán
đáp án. Run đổi config cần experiment mới. File lock chặn hai process viết cùng run.
Có thể gọi report riêng bất kỳ lúc nào để xem partial coverage.

LongVideoBench hỗ trợ inference/report bằng cùng runner khi đã có videos tại
`<root>/videos/<video_path>`; tạo selection với `--dataset_name longvideobench`,
cache `outscores/longvideobench/blip/`, annotation tương ứng. Dataset LVB gated:
xin quyền và tải theo [nguồn chính thức](https://huggingface.co/datasets/longvideobench/LongVideoBench).
Downloader tự động trong nhánh này chỉ hỗ trợ VideoMME.

## 5. Kết quả và giới hạn diễn giải

Trong `$EXPERIMENT/qa_pair/`:

- `config.json`, `environment.json`: inputs/model/video/code hashes, versions, GPU.
- `predictions/<index>.json`: hai câu trả lời, gold, frame IDs/timestamps, token grid,
  số token, thời gian decode/preprocess/generate, peak GPU memory.
- `report/comparison.json`, `comparison.md`: accuracy hai selector, delta điểm
  phần trăm, coverage, improved/regressed, breakdown, bootstrap CI và selection stats.
- `report/per_question.csv`, `improved.json`, `regressed.json`: xem từng câu thay đổi.

Accuracy chỉ có khi mô hình chạy thật. Report chấm lại từ response, letter output
không hợp lệ/giải thích nhiều câu tính sai; không random fallback như một số grader
LVB upstream. Prompt theo template video-only AKS, kể cả wording VideoMME có nhắc
subtitles; runner không cung cấp subtitles, giống nhau cho cả hai selector.

Decoder lấy đúng ID gốc, không uniform fallback khi thiếu K. Qwen nhận mảng frame
trực tiếp, resize không gian theo cùng pixel cap; grid hai selector phải bằng nhau.
Temporal embedding dùng FPS giả định 1.0 cho chuỗi frame đã chọn, không tái tạo
khoảng thời gian không đều thật; timestamp thật ghi riêng để audit. Processor có
thể lặp frame cuối khi N lẻ để đủ temporal patch, không thêm ID mới. Đây là giới hạn
protocol chung, cần nêu trong báo cáo. GPU inference chưa được xác minh trong phiên này.

Tổng timing inference không gồm download/model load; pair đầu có warmup; order
luân phiên giảm lệch nhưng chưa là microbenchmark tốc độ. Selection runtime đo riêng.
Resume sang GPU khác bị từ chối để tránh trộn môi trường. Bootstrap lấy mẫu theo
video, không coi nhiều câu cùng video độc lập. Không kết luận cải thiện từ smoke test;
khóa hyperparameters trước test và đánh giá cùng queries/K/checkpoint/seed.

`experiments/`, weights, downloads và virtualenv không đưa lên GitHub. Các path
server được đặt bằng biến môi trường, không hard-code `/media02` hay account trường.
