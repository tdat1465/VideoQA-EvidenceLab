# AKS gốc và AKS + Watershed

## Source và phạm vi

Đã đọc paper AKS (`paper/AKS.pdf`) và LENS (`paper/LENS.pdf`) trong workspace,
bao gồm phần phương pháp, thực nghiệm và appendix liên quan. FOCUS và
Coffee-Mate được đọc để đối chiếu bối cảnh: FOCUS thay chiến lược exploration /
exploitation; Coffee-Mate có self-distillation. Hai cơ chế này không được đưa vào
ablation chỉ thay selector của AKS.

Source chính thức được xác định từ URL trong paper:

| Source | Commit đã kiểm tra | Vị trí trong workspace |
|---|---|---|
| [AKS](https://github.com/ncTimTang/AKS) | `b0b8a58fedf1d05d78151e2969cecde60e83721d` | `E:\KL1\AKS` (bản gốc nguyên trạng) |
| [LENS](https://github.com/zhangce01/LENS) | `a3868ab0c50afd34078c350a9daed9382ab2d251` | `E:\KL1\LENS` (bản gốc nguyên trạng) |
| Nhánh triển khai | `feat/aks-watershed-selector` dựa trên commit AKS ở trên | `E:\KL1\aks-watershed` |

Nhánh triển khai giữ lịch sử AKS. Checkout dùng sparse checkout cho source,
`evaluation/` và `datasets/`; những file upstream còn lại vẫn thuộc lịch sử Git.
Cache scores hiện có ở `../AKS/outscores`. Không cần đưa bản clone LENS vào nhánh
AKS: implementation mới chỉ tham khảo ý tưởng basin, không import LENS hay thay
pipeline bằng LENS. File fixture `tests/fixtures/upstream_frame_select.py` là bản
nguyên trạng của selector AKS tại commit đã ghi, dùng để kiểm tra regression.

## Vị trí tích hợp: không phải chia đoạn cố định rồi top-K

`feature_extract.py` tính score query-frame bằng BLIP / CLIP / SeViLA và lưu
`scores.json`, `frames.json` theo thứ tự câu hỏi. Source scorer này không sửa.
AKS gốc `frame_select.py` thực hiện:

1. `nums = len(scores) // ratio`, lấy các vị trí `i * ratio` cho `i < nums`.
   Điều này bỏ phần dư, không hoàn toàn giống slice `scores[::ratio]`.
2. Chuẩn hóa min-max **trên toàn bộ chuỗi candidate**.
3. `meanstd(...)` tính mean, std và `mean(top_K) - mean` trong mỗi node.
   Code public truyền **K toàn cục** vào mọi lần đệ quy, không chia K cho phép
   tính judge trong node con.
4. Dừng node nếu `mean_diff > t1 and std > t2`; nếu chưa dừng và còn độ sâu,
   chia đôi candidate ở vị trí giữa. Đây là judge-and-split phụ thuộc score.
5. Mỗi terminal leaf ở depth `d` nhận quota `int(K / 2**d)`. `heapq.nlargest`
   cuối pipeline chọn theo quota, rồi toàn bộ frame IDs được sắp theo thời gian.

Có khác biệt giữa source public và cách diễn đạt trong paper: appendix Algorithm 1
đề cập quota tỷ lệ chiều dài vùng và cách truyền M; bản public dùng depth-based
quota và K toàn cục khi judge. Nhánh này ưu tiên **tái hiện source public**, không
tự sửa các khác biệt đó. Top-K trong judge được giữ; chỉ top-K tại terminal leaf
được thay khi bật `--selector watershed`.

LENS `utils/sampling.py::select_anchor_frames_watershed` tìm valleys, lấy argmax
giữa các valleys và dùng k-means 1D trên timestamp nếu có quá nhiều ứng viên.
Các wrapper CLIP/BLIP/BLIP2 và graph của LENS chuyển **toàn bộ** kết quả về top-K
nếu tìm được quá ít anchor. `runners/run_lens.py` còn có graph refinement, spatial
masking, hyperframes và allocator bằng MLLM. Không đưa các thành phần đó vào AKS.

## Selector mới

Tùy chọn mặc định vẫn là `original`. Với `watershed`, mỗi terminal leaf:

1. Làm mượt Gaussian với edge replication, chỉ để tìm hình dạng đường điểm.
   Không làm mượt score dùng trong judge, không đổi scoring model.
2. Gộp plateau thành một marker ở giữa. Nhận cả cực đại một phía ở endpoint.
3. Tính prominence = chiều cao đỉnh trừ mức đáy cao hơn của hai phía, trước
   cực đại cao hơn hoặc ranh giới leaf. Endpoint dùng phía khả dụng. Loại marker
   không đủ prominence. Đơn vị prominence là scale chuẩn hóa toàn video.
4. Flood `-smoothed_score` từ các markers trên đồ thị láng giềng 1D bằng priority
   queue: marker-controlled watershed thực sự, có nhãn basin cho các candidate.
   FIFO phá hòa ở plateau. Đây không phải gọi tên Watershed cho phép top-K.
5. Chọn frame có **score chưa làm mượt cao nhất trong basin**. Khi bằng điểm,
   ưu tiên gần marker rồi timestamp sớm hơn.

Ứng viên của mọi leaf được xếp theo
`utility = normalized_relevance + prominence_weight * prominence` giảm dần;
phá hòa bằng relevance, prominence rồi thời gian. Chỉ chấp nhận khi leaf còn quota
và cách mọi frame đã chọn ít nhất `min_distance`. NMS áp dụng cả qua ranh giới leaf;
quota của mỗi leaf không đổi. Smoothing / flooding không vượt ranh giới leaf AKS.

Nếu thiếu frame trong leaf, bổ sung từ các candidate chưa chọn của **leaf đó**:

- Giữ nguyên các đỉnh đã chấp nhận; không thay chúng bằng một tập top-K mới.
- Ưu tiên khoảng cách nhỏ nhất tới tập đã chọn lớn nhất (farthest-point refill),
  tiếp theo là score chưa làm mượt, sau cùng là thời gian sớm hơn.
- Nếu chưa chọn frame nào, lấy score cao nhất, phá hòa bằng thời gian.
- Ưu tiên candidate đáp ứng khoảng cách. Chỉ khi không còn candidate hợp lệ trong
  bất kỳ leaf đang thiếu quota nào mới nới khoảng cách cho lần bổ sung đó.
  `distance_relaxations` và reason `relaxed_refill` ghi rõ tình huống này.

Khoảng cách là **soft constraint khi refill**: không thể vừa bảo đảm lấy đủ quota
vừa bảo đảm gap quá lớn trên video ngắn. Greedy NMS/refill cũng không phải nghiệm
tối ưu của bài toán đóng gói khoảng cách. Không đánh đồng số cặp gần nhau cuối
cùng với số lần nới gap: một frame bổ sung có thể tạo nhiều cặp gần.

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--selector` | `original` | baseline hoặc biến thể |
| `--watershed_sigma` | `1.0` | Gaussian sigma theo candidate tick sau ratio; `0` tắt smoothing |
| `--watershed_min_prominence` | `0.05` | ngưỡng prominence trên scale chuẩn hóa toàn video |
| `--watershed_min_distance` | `3` | gap theo candidate tick, không phải decoded frame ID |
| `--watershed_prominence_weight` | `0.25` | trọng số prominence trong utility; `0` xếp theo relevance |

Các giá trị này là cấu hình khởi đầu, chưa được tối ưu bằng QA accuracy. Vì scorer
gốc lấy khoảng 1 candidate/s với stride `int(fps)`, gap 3 xấp xỉ 3 giây khi ratio=1;
không bảo đảm đúng 3 giây, nhất là với frame IDs/timestamp không đều.

## Ngân sách và các trường hợp biên

Tổng output không vượt K. Trong một cây nhị phân, tổng `2**(-depth)` của các leaf
không vượt 1, nên tổng quota làm tròn không vượt K. Output mới lấp
`min(quota_leaf, candidate_count_leaf)` cho mỗi leaf. Không chuyển slot còn dư
sang leaf khác: như vậy sẽ thay đổi allocator AKS.

Do đó có thể ít hơn K khi K lẻ, all_depth quá lớn hoặc số candidate không đủ.
Đặc biệt K=8 với all_depth=5 có thể tạo leaf quota=0, thậm chí tập rỗng, **đúng
như source gốc**. Cho so sánh ngân sách 8/16/32/64, có thể chọn độ sâu dưới đây
cho **cả hai** selector:

| K | all_depth cho một cấu hình khả dụng |
|---:|---:|
| 1 | 0 |
| 8 | 3 |
| 16 | 4 |
| 32 | 5 |
| 64 | 5 (mặc định source public) |

Đây là các cấu hình chạy đề xuất, không phải tuyên bố chúng tái hiện mọi bảng
kết quả trong paper. Luôn công bố all_depth, t1, t2 và actual frame count.

- Chuỗi phẳng: chuẩn hóa về 0 và ép judge chia như hành vi NaN của source gốc;
  selector Watershed không tạo peak giả, dùng refill.
- Empty / K=0: trả tập rỗng. Short-video path (`N<K` sau ratio) giữ nguyên AKS,
  trả toàn bộ candidate, không chạy peak selection / NMS.
- Empty child trong đệ quy bị bỏ qua để tránh warning và mở rộng cây vô ích;
  mọi nonempty node vẫn chia tới cùng stopping depth của upstream.
- Frame ID float có giá trị nguyên trong cache chính thức được nhận và xuất
  thành integer. Reject ID trùng, đảo thời gian, âm, fraction; reject score NaN/Inf
  và số query / độ dài score-frame không khớp. Không âm thầm zip bỏ dữ liệu.
- `t1/t2` được parse thành float: source cũ đặt `type=int` dù default t1=0.8,
  làm CLI không nhận được ngưỡng thập phân. Default không đổi.

## Chạy kiểm thử CPU

Từ `E:\KL1\aks-watershed` (PowerShell hoặc terminal Linux):

```text
python -m pip install -r requirements-selection.txt
python -m unittest discover -s tests -v
python tests/check_cached_scores.py --score_root ../AKS/outscores --report experiments/validation/upstream-regression.json
```

Trong phiên triển khai dùng Python bundled:
`C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
Nếu `python` không có trong PATH, thay bằng đường dẫn này; với PowerShell dùng
`& 'đường dẫn python.exe' ...`.

## Chọn frame riêng lẻ

Baseline nguyên API output:

```text
python frame_select.py --score_path ../AKS/outscores/videomme/blip/scores.json --frame_path ../AKS/outscores/videomme/blip/frames.json --dataset_name videomme --max_num_frames 64 --selector original --output_file experiments/original
```

Biến thể, cùng input và cùng tham số AKS:

```text
python frame_select.py --score_path ../AKS/outscores/videomme/blip/scores.json --frame_path ../AKS/outscores/videomme/blip/frames.json --dataset_name videomme --max_num_frames 64 --selector watershed --output_file experiments/watershed --diagnostics_file experiments/watershed/diagnostics.json
```

Mỗi output vẫn là JSON list-of-lists frame IDs tại
`OUTPUT/dataset_name/extract_feature_model/selected_frames.json`.
Chạy selector riêng có thể ghi đè output; dùng thư mục riêng như ví dụ.

## Chuẩn bị phép so sánh cùng dữ liệu, mô hình và ngân sách

Lệnh dưới đây chạy cả hai selector trên cùng cache, kiểm tra leaf_plan và actual
frame count từng query giống nhau, ghi SHA-256 inputs/code và tạo task evaluation
riêng cho mỗi selector. Dùng output_dir mới/rỗng để tránh checkpoint cũ.

```text
python compare_selectors.py --score_path ../AKS/outscores/videomme/blip/scores.json --frame_path ../AKS/outscores/videomme/blip/frames.json --dataset_name videomme --max_num_frames 64 --all_depth 5 --annotation_path datasets/videomme/include_frame_idx.json --video_root /path/to/VideoMME --output_dir experiments/videomme-blip-k64-new
```

`--video_root` là thư mục benchmark chứa `data/*.mp4` cho VideoMME, hoặc
`videos/*.mp4` cho LVB. Tại Windows dùng đường dẫn Windows thật; khi chuyển sang
Linux để inference, chạy lại prepare ở Linux để task YAML có đường dẫn Linux.
Không tái sử dụng task YAML chứa đường dẫn Windows trong Linux.

Ví dụ LVB với ngân sách 8:

```text
python compare_selectors.py --score_path ../AKS/outscores/longvideobench/blip/scores.json --frame_path ../AKS/outscores/longvideobench/blip/frames.json --dataset_name longvideobench --max_num_frames 8 --all_depth 3 --annotation_path datasets/longvideobench/include_frame_idx.json --video_root /path/to/LongVideoBench --output_dir experiments/lvb-blip-k8
```

Có thể bỏ `--annotation_path` và `--video_root` để chỉ phân tích selector; thêm
`--limit 100` để smoke test cùng 100 query đầu. Full annotation count vẫn phải khớp
full score count trước khi áp dụng limit.

Các file xuất ra:

- `original/selected_frames.json`, `watershed/selected_frames.json`.
- `original/diagnostics.json`, `watershed/diagnostics.json`: quota, basin, prominence,
  lý do chọn và số lần nới gap.
- `manifest.json`: arguments, input/code hashes, phiên bản NumPy, Git trạng thái.
- `summary.json`: count histogram, số query thiếu K, score, temporal span, gap,
  runtime. Đây là **thống kê chọn frame, chưa phải QA accuracy**.
- Khi có annotations/video_root: từng nhánh có `annotations.json` và `lmms_task/`.
  Task names riêng; prompt, generation kwargs và grading lấy từ template AKS.
  Annotation gốc không bị sửa.

Cache AKS là list theo thứ tự, không có query ID trong scores. Không thể chứng
minh alignment với một annotation mới chỉ từ length. Phải dùng đúng annotation
và thứ tự đã dùng khi extract scores; ghi lại hash. Muốn đổi model scoring,
extract một lần rồi chia sẻ **chính cache đó** cho cả hai selector.

## Evaluation QA trên GPU

Môi trường selector CPU không thay thế môi trường LLaVA-Video / lmms-eval.
Thực hiện setup theo README upstream trong môi trường Linux/CUDA tương thích,
cài các adapter mà AKS yêu cầu. Khi copy `evaluation/llava_vid.py` vào
`lmms_eval/models/llava_vid.py`, phải dùng **file trên nhánh này** để có cờ strict.
`evaluation/task.py`, `evaluation/evaluator.py` giữ source upstream. API lmms-eval
hiện tại có thể khác bản mà AKS dựa vào; chưa xác nhận runtime compatibility
trong phiên này. Pin commit/versions của environment đã chạy thành công, lưu
`pip freeze`, model revision/checkpoint hash và GPU vào báo cáo thực nghiệm.

```bash
# Chạy từ root repo, sau khi chuẩn bị task tại Linux và cài adapter.
bash scripts/evaluate_pair.sh experiments/videomme-blip-k64-new /path/to/LLaVA-NeXT-Video-7B-Qwen2 64 1 42
```

Hai lần inference có cùng checkpoint, task prompt, deterministic generation từ
template, seed=42, K, batch size và adapter. K phải khớp manifest; script kiểm tra.
Output QA tách biệt ở `original/qa/` và `watershed/qa/`; không dùng request cache
chung hoặc chạy tiếp vào output đã có.

Điểm đáng chú ý: `load_video_index` gốc thay tập chọn bằng uniform nếu thiếu K.
Cờ mới `strict_selected_frames=True` tắt fallback đó, đọc chính xác các frame đã
xuất và dùng timestamp theo FPS thật; bật cho **cả hai** selector. Default adapter
vẫn giữ hành vi cũ khi không bật cờ. Strict reject tập rỗng, duplicate/out-of-range
ID và `force_sample=True`. K nhỏ/all_depth lớn cần kiểm tra trước khi inference.

Script này hỗ trợ adapter `llava_vid` đã kiểm tra. Với Qwen2-VL/OneVision hoặc
LENS runner, phải kiểm tra riêng đường decode/mapping và fallback; không chỉ đổi
tên model rồi giả định frame ID được dùng đúng. Không dùng full LENS như baseline
cho ablation này vì spatial masking, graph và hyperframes là các yếu tố khác.

## Lưu ý khi đánh giá

So sánh QA accuracy overall và theo độ dài/type trên cùng tập query. Báo cáo số
frame thực tế, query lỗi/bị bỏ qua, decode IDs thực tế, token counts, runtime chọn
frame và runtime tổng. Với nhiều câu hỏi cùng video, dùng paired bootstrap theo
video hoặc phân tích thống kê theo nhóm video; không mặc định query độc lập.

Temporal span chỉ là max timestamp trừ min timestamp; không chứng minh bao phủ
mọi sự kiện. Gap lớn giảm trùng lân cận nhưng có thể bỏ chi tiết/action ngắn;
Gaussian có thể nhập đỉnh, prominence thấp có thể giữ nhiễu, prominence cao có
thể mất đỉnh yếu cần thiết. Refill có thể chọn frame kém liên quan để tăng phân tán.
Report cả relevance giảm và redundancy giảm, không chỉ metric có lợi.

Chạy ablation cùng input với sigma=0, prominence=0, distance=1 hoặc weight=0 để
phân biệt tác động Watershed/smoothing/prominence/NMS/refill. Chọn hyperparameter
trên validation riêng rồi khóa trước test; không chọn cấu hình từ test accuracy.
Khi N<K, hai selector giống nhau do giữ short-video path; tách nhóm này trong báo cáo.

## File triển khai và push sau

- Sửa `frame_select.py`: API/CLI selector, validation, tích hợp terminal leaves.
- Thêm `watershed_selector.py`: Gaussian, prominence, priority-flood, NMS/refill.
- Sửa `evaluation/llava_vid.py`: cờ strict opt-in; mặc định không đổi.
- Thêm `compare_selectors.py`, `scripts/evaluate_pair.sh`: paired outputs và QA commands.
- Thêm `tests/test_selection.py`, fixture upstream, `tests/check_cached_scores.py`.
- Thêm `requirements-selection.txt`, `.gitignore`, `.gitattributes`, tài liệu và provenance.
- README upstream được giữ nội dung, thêm link hướng dẫn biến thể.

Hai clone nguồn gốc AKS/LENS không bị chỉnh sửa. Không push trong phiên triển khai.
Khi bạn đã có fork AKS hoặc repository GitHub của mình:

```bash
cd /path/to/aks-watershed
git remote rename origin upstream
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin feat/aks-watershed-selector
```

Không dùng remote upstream để push nếu không có quyền. `experiments/`, models,
checkpoint và data phát sinh được ignore; báo cáo tổng hợp có thể commit riêng,
không cần đưa toàn bộ diagnostic/QA artifacts vào Git.
