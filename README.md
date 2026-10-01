# VideoQA-EvidenceLab

Thử nghiệm **cải tiến lúc suy luận** cho video multiple-choice QA trên **1 GPU, ngân sách khoảng 32 GB VRAM, mỗi phiên 48 giờ**. Backbone mặc định: **Molmo2-4B đóng băng**. Dataset đang nghiên cứu: **NExT-QA MC** và **LongVideoBench validation**. Adapter STAR cũ được giữ để đọc lại thí nghiệm, không còn là mục tiêu chạy mới. Không cần huấn luyện lại hoặc dịch vụ API trả phí.

Repo này triển khai một **giả thuyết nghiên cứu cần kiểm chứng**, không có tuyên bố SOTA. Người dùng đã chạy baseline NExT-QA Uniform-16 + Molmo2-4B trên A100, đạt 173/200 câu (86,5%); đây là kết quả mẫu 200 câu, không phải full-split. Khung chạy tham khảo kinh nghiệm Slurm/RAM/resume từ [tdat1465/DyGEnc](https://github.com/tdat1465/DyGEnc). DyGEnc đang dùng AGQA và scene graph; repo mới dùng video gốc, không dùng checkpoint/graph ground truth của DyGEnc.

## FOCUS trên LongVideoBench

Đã bổ sung Uniform-64/FOCUS-64 cho **Molmo2-4B** và **LLaVA-Video-7B-Qwen2**.
Xem [hướng dẫn server trường](docs/LONGVIDEOBENCH.md). Dùng **video-only validation**, mặc định
200 câu cố định, `limit: 0` cho đủ 1.337 câu. Chấm logits A–E; FOCUS chạy trên toàn timeline
với BLIP-large ITM. Đây là tích hợp cần pilot GPU, không phải tuyên bố tái lập số liệu paper.

Workflow mới chỉ giữ một video trong RAM; không tải toàn bộ archive khoảng 150,47 GiB.
Các cấu hình `lvb_*.json` dùng entrypoint `python -m evidencelab.longvideo` và
`submit.py --workflow longvideo`. Không đưa chúng vào runner NExT-QA cũ.

## Phương án đang triển khai

**LENS trên LongVideoBench:** đã có adapter cho Molmo2/LLaVA, ngân sách 8/64 ảnh,
cùng Uniform/FOCUS để đối chiếu. Xem [hướng dẫn LENS và khác biệt so với upstream](docs/LENS.md).
Tính cả lượt LLM phân bổ, frame nguồn trong hyperframe và chi phí BLIP/CLIP/SSIM.
Chưa có kết quả GPU LENS; bản này cần pilot riêng và RUN_DIR mới.

1. Lấy 8 khung hình đều, chạy VLM và tính xác suất tương đối của các lựa chọn A–E.
2. Nếu chênh lệch hai lựa chọn đầu nhỏ hơn ngưỡng: dùng CLIP tìm khung phân biệt hai lựa chọn, thêm ngữ cảnh trước/sau trong ngân sách tối đa 16 khung.
3. Chạy VLM thêm một lần trên tập khung đã mở rộng. Ghi cả hai lần gọi, token, thời gian và đỉnh VRAM.

Đây là prototype `evidence`, **không phải bản tái lập A.I.R.**; độ tin cậy dùng margin heuristic, chưa được calibration. Ý tưởng sai cũng phải có thể đo được bằng các đối chứng sau:

| Config | Thiết lập | Vai trò |
|---|---|---|
| `molmo2_uniform8/16/32.json` | 8/16/32 khung đều, 1 lần gọi | Đường cơ sở theo tài nguyên |
| `molmo2_clip16.json` | CLIP chọn top-16 | Giá trị của chọn khung theo câu hỏi |
| `molmo2_aks16.json` | Hàm chia đoạn AKS gốc + CLIP, tối đa 16 | Baseline chuyển sang backbone/dataset mới |
| `molmo2_uniform_refine16.json` | 8 → 16 khung đều khi margin thấp | Đối chứng cùng cơ chế gọi lần hai |
| `molmo2_evidence16.json` | 8 → 16 khung theo bằng chứng | Giả thuyết cần kiểm chứng |
| `qwen25_uniform16.json` | Qwen2.5-VL-7B, chuỗi ảnh có timestamp | Backbone bổ sung, cần pilot VRAM riêng |

Các cấu hình NExT-QA phía trên mặc định chọn cùng **200 câu** bằng seed/ID, pool tối đa **64 khung**, CLIP chạy CPU để nhường VRAM cho VLM. `limit: 0` chạy cả split; đổi cấu hình phải tạo run mới. Khung đầu vào đã được thu nhỏ cạnh dài tối đa 448 px. Protocol này khác recipe đánh giá gốc của từng paper. Workflow LongVideoBench/FOCUS có giao thức riêng được mô tả trong tài liệu liên kết ở trên.

## Chạy thử không cần GPU

Python 3.10 trở lên, chạy tại thư mục repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m evidencelab demo --output runs/demo
```

Trên Windows dùng `.venv\Scripts\activate`. Demo tự tạo 12 câu **giả lập**, dừng sau 4 câu rồi resume; dùng để kiểm tra hệ thống, không phải đo chất lượng mô hình.

## Chạy video thật

Hướng dẫn đúng môi trường trường học, lưu tạm trên RAM và Slurm: **[docs/SERVER.md](docs/SERVER.md)**.

**Máy login thiếu ổ đĩa:** dùng hướng dẫn Slurm ở trên. Profile xin **90 GiB RAM hệ thống (~96,6 GB), 1 GPU trên một node trong gpu01/02/03**, loại gpu04. Submit bằng `python3 scripts/slurm/submit.py`; job tự tải annotation và NExTVideo vào tmpfs (`DATA_MODE=nextqa_auto`), chỉ giải nén split cần chạy rồi xóa ZIP. Model và môi trường Python cũng nằm trên RAM. Không cần chuẩn bị VIDEO_ARCHIVE trên login. Root là `/media/lnthanh03/DatHa`; dùng RUN_DIR mới khi chuyển từ code cũ sang bản này. Precision mặc định `auto`: native BF16 nếu có, ngược lại FP16; không yêu cầu A100.

Các lệnh cài đặt thủ công dưới đây dành cho máy có không gian lưu trữ riêng; **không chạy chúng trên login của trường**. Trên máy Linux có GPU, đã cấp đúng 1 GPU và driver tương thích CUDA 12.6:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-server.txt
pip install -e .
python -m evidencelab doctor
```

Chuẩn bị [video và annotation](docs/DATA.md), ví dụ NExT-QA validation:

```bash
python scripts/fetch_nextqa_annotations.py --split val --output data/annotations
python -m evidencelab prepare --dataset nextqa --split val \
  --annotations data/annotations/val.csv --video-root /path/to/NExTVideo \
  --output data/nextqa-val.jsonl

python -m evidencelab run --config configs/molmo2_uniform16.json \
  --manifest data/nextqa-val.jsonl --video-root /path/to/NExTVideo \
  --output runs/nextqa-val-uniform16 --max-new-samples 2
```

Lệnh pilot trả mã **75** khi chủ động dừng chưa xong split. Kiểm tra `summary.json`, VRAM và số token. Tiếp tục bằng chính lệnh trên, **bỏ `--max-new-samples 2` và thêm `--resume`**. Nếu OOM, tạo run mới với 8 khung; không âm thầm đổi protocol giữa run.

Chạy lại với `molmo2_evidence16.json` và một output mới, sau đó:

```bash
python -m evidencelab compare runs/nextqa-val-uniform16 runs/nextqa-val-evidence16 \
  --output runs/comparison.json
```

Lệnh so sánh yêu cầu hai run hoàn thành, cùng manifest, backbone, source/runtime và tập câu hỏi. Báo cáo chênh lệch accuracy và khoảng bootstrap 95% theo **video** để xử lý nhiều câu liên quan cùng video. STAR có cả accuracy gộp và mean accuracy của bốn loại câu hỏi.

Để dùng AKS:

```bash
python -m evidencelab fetch-aks --output third_party/aks/frame_select.py
# Với config molmo2_aks16, thêm --aks-file third_party/aks/frame_select.py vào lệnh run.
```

## Kết quả và tái lập

Mỗi run có `results.sqlite3` (nguồn kết quả chính, commit từng câu), `contract.json`, `summary.json`, `predictions.jsonl`, `predictions.csv`. JSONL lưu khung/timestamp, lựa chọn cạnh tranh, xác suất, từng lần gọi và chi phí. Có thể tạo lại các file xuất bằng `python -m evidencelab report runs/TEN_RUN` khi run đã dừng.

Resume kiểm tra hash mã nguồn, config, manifest (bao gồm SHA256 từng video), model revision và phiên bản thư viện chính. Video được xác minh lại khi đọc ở mỗi phiên. Không tự bỏ qua lỗi dữ liệu, không tính câu lỗi là hoàn thành, không tự giảm số khung khi OOM. Cần giữ nguyên checkout/môi trường khi resume.

**[Protocol nghiên cứu](docs/EXPERIMENTS.md)** · **[Dữ liệu](docs/DATA.md)** · **[Máy trường](docs/SERVER.md)** · **[Nguồn và giấy phép](THIRD_PARTY.md)** · **[Kiểm thử](docs/VALIDATION.md)**

## Các nguồn chính

- [Molmo2, 2026](https://arxiv.org/abs/2601.10611), [code chính thức](https://github.com/allenai/molmo2), [Molmo2-4B weights](https://huggingface.co/allenai/Molmo2-4B).
- [AKS, CVPR 2025](https://arxiv.org/abs/2502.21271), [code chính thức](https://github.com/ncTimTang/AKS).
- [A.I.R., ICLR 2026 — nguồn cảm hứng mở rộng tìm bằng chứng](https://github.com/UCF-AIR/A.I.R.).
- [NExT-QA](https://github.com/doc-doc/NExT-QA), [STAR](https://github.com/csbobby/STAR_Benchmark).

Tên backbone mạnh không bảo đảm SOTA dưới ngân sách/protocol khác. Chỉ so sánh với paper sau khi khớp split, input, training exposure, số frame và cách chấm điểm.
