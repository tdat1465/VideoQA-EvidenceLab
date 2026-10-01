# LENS trên LongVideoBench: Molmo2 và LLaVA-Video

Nguồn: [paper](https://arxiv.org/abs/2607.25125),
[code chính thức](https://github.com/zhangce01/LENS/tree/a3868ab0c50afd34078c350a9daed9382ab2d251).
Revision được pin: `a3868ab0c50afd34078c350a9daed9382ab2d251`.
Đây là **bản tích hợp nghiên cứu**, chưa xác nhận parity số học hoặc tái lập accuracy paper.
Molmo2 là backbone bổ sung của EvidenceLab; không coi là kết quả backbone do tác giả LENS công bố.

## Giao thức

Giữ LongVideoBench **validation, video-only, không phụ đề/audio**, cùng 200 ID theo hash/seed 18.
Đáp án cuối vẫn chấm logits A–E. Không đưa nhãn đúng hoặc referred timestamps vào selector.
Trong từng backbone, so sánh Uniform, FOCUS và LENS ở cùng ngân sách **8 hoặc 64 ảnh đầu vào**.
64 để đối chiếu các thí nghiệm FOCUS hiện có; 8 để thăm dò ngân sách nhỏ, không mặc định là
toàn bộ recipe đánh giá của paper. Không so accuracy mẫu 200 trực tiếp với bảng full-split.

LENS gồm:

1. Một lượt **text-only** trên chính backbone đóng băng, tối đa 16 token, đọc câu hỏi để
   sinh tỷ lệ spatial `r`. Prompt nguyên bản; parse float, clamp [0,1], fallback 0.5 nếu
   không parse được hoặc NaN/Inf. Lưu nguyên output và cờ fallback.
2. Chấm BLIP ITM FP32 theo batch 4 trên timeline khoảng 1 fps. Dùng prompt câu hỏi + các
   lựa chọn theo loader LongVideoBench gốc; không có đáp án đúng. Dùng HF BLIP processor
   gốc LENS, khác preprocess LAVIS-port của FOCUS.
3. Spatial: watershed chọn anchor, CLIP ViT-L/14-336 PRS ở layer 22 tạo mask vùng ít liên quan.
4. Temporal: pool `min(T, max(128, 8*N))`, Gaussian SSIM, graph diffusion với beta=0.8,
   10 vòng, sigma=0.3; ghép 4 frame lân cận thành ảnh 2×2 thu nhỏ về kích thước một frame.
5. Sắp các view theo timestamp anchor, truyền **chính các ảnh đã mask/ghép** vào VLM.

Ảnh 2×2 chứa nhiều thời điểm: số ảnh đầu vào bằng nhau không có nghĩa số frame nguồn bằng nhau.
`mean_vlm_calls` tính cả lượt phân bổ; `input_tokens` gồm phân bổ + trả lời;
`answer_input_tokens` giữ riêng token trả lời. `lens_views` ghi anchor và nguồn từng ô.
`source_frame_presentations`, `unique_source_frames`, số lần chấm BLIP, latency và peak VRAM
được ghi để phân tích chi phí. `frame_presentations`/`final_frames` là số view đầu vào VLM.

## Thay đổi kỹ thuật so với upstream

- Dùng nguyên các hàm watershed, graph diffusion, prompt và API_CLIP có manifest SHA256.
  File vendored giữ nguyên byte của checkout đã kiểm tra; không sửa thuật toán trong vendor.
- Decode lười theo chỉ số, không materialize toàn video khoảng 1 fps vào RAM. Dùng grid
  `range(0,T,max(1,round(fps)))`, tối đa 9999 như loader LLaVA upstream. Đây là quy ước chung
  cho cả Molmo2/LLaVA ở bản này, không khẳng định giống mọi loader khác của upstream.
- SSIM giữ cùng công thức, area resize cạnh dài 224, nhưng chia batch tối đa 32 cặp trên GPU;
  local statistics ở CPU. CPU test đối chiếu với hàm upstream đạt tolerance 2e-6 tuyệt đối,
  2e-5 tương đối; không đảm bảo bitwise parity trên GPU. Graph/watershed chạy CPU, tie-order
  có thể khác CUDA. RNG gắn question ID để resume độc lập thứ tự job.
- Runner upstream nhánh `llava_video` preprocess lại `raw_video` sau khi đã tạo LENS views.
  Bản này truyền đúng transformed views. Đồng thời thống nhất tensor/image layout và resize
  hyperframe cho cả hai backbone (nhánh numpy upstream không resize).
- Video ngắn có thể trả ít hơn ngân sách; lưu số thực tế, không âm thầm thêm view. Hyperframe
  thiếu 4 frame được lặp boundary. Mask không hữu hạn làm run dừng để kiểm tra, không ghi
  prediction lỗi. Không skip câu lỗi để làm tăng accuracy trên phần dễ.
- Chấm đáp án bằng logits, không dùng phép kiểm tra substring trong câu trả lời như runner gốc.

## RAM và giấy phép

Vẫn dùng 1 GPU, 90 GiB RAM (~96,6 GB), 48 giờ, gpu01/02/03; luôn loại gpu04.
Video tải HTTP Range từng khối 4 MiB vào `/dev/shm/videoqa.<job>.*`; xong video thì xóa.
BLIP xử lý từng batch, graph chỉ giữ pool ảnh thu nhỏ. Trọng số/env/cache nằm trong RAM;
CLIP checkpoint gốc OpenAI tải theo khối vào RAM và xác minh SHA256, không dùng `~/.cache/clip`.
Guard cgroup/tmpfs và reserve 6 GiB vẫn có hiệu lực. Chưa đo peak RAM/VRAM với trọng số thật;
guard không chứng minh rằng mọi spike khi nạp/chạy mô hình đều nằm trong giới hạn.

LENS top-level có MIT license (`_vendor/lens/LICENSE`). Component `API_CLIP/clip_prs`
giữ **CC BY-NC 4.0** và attribution của Yossi Gandelsman, Alexei A. Efros, Jacob Steinhardt
trong README/LICENSE đi kèm. Không coi toàn bộ component này là MIT. File nguồn có attribution
OpenAI/OpenCLIP được giữ nguyên. Các sửa đổi tích hợp nằm ngoài cây vendor.

## 1. Lấy checkout riêng trên login01

Không cập nhật checkout đang dùng để resume FOCUS/NExT-QA. Các cặp so sánh mới phải chạy
Uniform/FOCUS/LENS từ cùng bản code/môi trường này vì fingerprint đã đổi.

```bash
(
  set -euo pipefail
  trap 'echo "Dừng tại dòng $LINENO; xem lỗi phía trên." >&2' ERR
  ROOT=/media/lnthanh03/DatHa
  OLD_REPO="$ROOT/code/VideoQA-EvidenceLab"
  NEW_REPO="$ROOT/code/VideoQA-EvidenceLab-lens-v1"
  if [ -e "$NEW_REPO" ]; then
    echo "Checkout đã tồn tại: $NEW_REPO; kiểm tra trước khi chạy."
    git -C "$NEW_REPO" log -1 --oneline
    exit 1
  fi
  git -C "$OLD_REPO" fetch https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/lens-longvideobench
  git -C "$OLD_REPO" worktree add --detach "$NEW_REPO" FETCH_HEAD
  git -C "$NEW_REPO" log -1 --oneline
)
```

Chấp nhận quyền truy cập [dataset](https://huggingface.co/datasets/longvideobench/LongVideoBench)
trên website rồi nhập token kín trên cùng terminal login01:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-lens-v1"
cd "$REPO_ROOT"
```

## 2. Pilot Molmo2 LENS-64: 2 câu

Không source env NExT-QA cũ. Chỉ submit trên login; mọi tải lớn diễn ra trên compute node.

```bash
(
  set -euo pipefail
  : "${HF_TOKEN:?Nhập và export HF_TOKEN}"
  cd "$REPO_ROOT"
  export CONFIG_FILE=configs/lvb_molmo2_lens64.json
  export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-lens64-200-lensv1"
  export RESUME=0 MAX_NEW_SAMPLES=2
  if [ -e "$RUN_DIR" ]; then
    echo "RUN_DIR đã tồn tại: $RUN_DIR; kiểm tra trước khi chạy."
    exit 1
  fi
  python3 -m json.tool "$CONFIG_FILE"
  python3 scripts/slurm/submit.py --workflow longvideo --test-only
  JOB_ID=$(python3 scripts/slurm/submit.py --workflow longvideo --parsable)
  echo "LENS pilot JOB_ID=$JOB_ID"
  squeue -j "${JOB_ID%%;*}"
)
```

Thay `12345` bằng JOB_ID thật:

```bash
JOB_ID=12345
squeue -j "$JOB_ID"
tail -n 80 "$REPO_ROOT/logs/lvbqa-$JOB_ID.out"
tail -n 80 "$REPO_ROOT/logs/lvbqa-$JOB_ID.err"
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
python3 -m json.tool "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-lens64-200-lensv1/summary.json"
```

Pilot dừng sạch ở 2 câu với exit 75, có thể hiện `FAILED 75:0`. Kiểm tra log
`Session stopped cleanly`, `completed_count: 2`, `mean_vlm_calls: 2`, allocation output,
số view và RAM/VRAM trước khi chạy tiếp. Hai câu không đủ để kết luận chất lượng chọn frame.

## 3. Resume đủ 200 câu

```bash
(
  set -euo pipefail
  : "${HF_TOKEN:?Nhập và export HF_TOKEN}"
  cd "$REPO_ROOT"
  export CONFIG_FILE=configs/lvb_molmo2_lens64.json
  export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-lens64-200-lensv1"
  export RESUME=1 MAX_NEW_SAMPLES=0
  python3 scripts/slurm/submit.py --workflow longvideo --test-only
  python3 scripts/slurm/submit.py --workflow longvideo --parsable
)
```

Không sửa code/config giữa pilot và resume. Nếu lỗi xảy ra trước khi tạo `results.sqlite3`,
retry bằng `RESUME=0`; nếu journal đã có, dùng `RESUME=1` với đúng contract.

## 4. Các cấu hình đối chiếu

Thay CONFIG_FILE và RUN_DIR ở block pilot/resume; luôn RUN_DIR riêng:

| CONFIG_FILE | Tên RUN_DIR dưới `$PERSIST_ROOT/runs/videoqa/` |
|---|---|
| `configs/lvb_molmo2_uniform64.json` | `lvb-val-molmo2-uniform64-200-lensv1` |
| `configs/lvb_molmo2_focus64.json` | `lvb-val-molmo2-focus64-200-lensv1` |
| `configs/lvb_llava_video_lens64.json` | `lvb-val-llava-video-lens64-200-lensv1` |
| `configs/lvb_llava_video_uniform64.json` | `lvb-val-llava-video-uniform64-200-lensv1` |
| `configs/lvb_llava_video_focus64.json` | `lvb-val-llava-video-focus64-200-lensv1` |

Có đủ sáu config ngân sách 8: thay `64` bằng `8` trong tên config và RUN_DIR.
Đặc biệt cần pilot LLaVA riêng. Không đổi ngân sách để resume journal của ngân sách khác.

Khi Molmo2 FOCUS và LENS cùng hoàn tất 200 câu ở bản này:

```bash
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"
python3 -m evidencelab compare \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200-lensv1" \
  "$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-lens64-200-lensv1" \
  --output "$PERSIST_ROOT/runs/videoqa/lvb-molmo2-focus-vs-lens64-lensv1.json"
```

Lệnh compare chỉ cần Python chuẩn, không tải model trên login. Tương tự cho Uniform và LLaVA.
Sau thăm dò, cấu hình mới `limit: 0` dùng full 1.337 câu và RUN_DIR mới.

## Kiểm chứng hiện có

49 tests CPU chạy qua, gồm SSIM chia batch đối chiếu upstream, CLIP PRS/mask bằng trọng số
tiny ngẫu nhiên, góc ảnh ghép, ratio/fallback, scoring lười theo batch, trace chi phí,
runner truyền đúng ảnh biến đổi và dọn video. Smoke LLaVA source thực với tiny weights
xác nhận text-only generate trả token mới và logits đáp án khớp generate một token.
Chưa chạy mô hình thật, chưa đo accuracy LENS hoặc peak RAM/VRAM trên server trường.
