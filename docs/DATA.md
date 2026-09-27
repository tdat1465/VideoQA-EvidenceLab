# Dữ liệu

Không commit dataset, annotation gốc, checkpoint, cache hoặc token. Người chạy tải video theo điều khoản của từng dataset. Chuẩn bị manifest có thể tốn thời gian vì cần hash video, nhưng mỗi video chỉ hash một lần trong lượt chuẩn bị.

## NExT-QA multiple choice

Trên server trường, dùng `DATA_MODE=nextqa_auto` theo [SERVER.md](SERVER.md): job tự tải CSV chính thức và archive video vào tmpfs trên GPU node. Không cần tải gì về login. Video lấy từ mirror [rhymes-ai/NeXTVideo](https://huggingface.co/datasets/rhymes-ai/NeXTVideo) tại revision `7e8ea8e056742292b95688d92a0773e05df00393`; archive có 24,253,160,356 bytes, SHA256 `2e3b1bc3e761122864b46fe3a1790b281301a6ed69de5ca1fceba17c504fa49c`. Kiểm tra danh mục ZIP ngày 2026-09-27 thấy đủ 5.440 video, bao phủ toàn bộ 570 video validation và 1.000 video test. Mirror không phải tài khoản tác giả dataset; nguồn gốc/điều khoản dataset vẫn theo NExT-QA.

Job tải toàn bộ ZIP vào RAM, kiểm tra SHA256, chỉ giải nén video của split rồi xóa ZIP. Validation giải nén 2,490,785,980 bytes; test 4,081,254,699 bytes. Không đổi split theo nguồn tải. File `data-source.json` nhỏ trong RUN_DIR lưu provenance; video/annotation/manifest nằm trên RAM. Những lệnh sau dành cho chuẩn bị dữ liệu thủ công khi đã có video.

Nguồn: [doc-doc/NExT-QA](https://github.com/doc-doc/NExT-QA). Script `fetch_nextqa_annotations.py` tải CSV chính thức và map ID ở commit `2432e9724f88ed9f40010e2989f104570a91de4e`. Các cột: `video, question, answer, qid, type, a0..a4`; `answer` là chỉ số 0–4. Video có thể nằm trong các thư mục con, được ghép qua tên file không gồm phần mở rộng. Nếu có stem trùng, chương trình báo lỗi thay vì đoán.

```bash
python scripts/fetch_nextqa_annotations.py --split val --output data/annotations
python -m evidencelab prepare --dataset nextqa --split val \
  --annotations data/annotations/val.csv --video-root /path/to/NExTVideo \
  --output data/nextqa-val.jsonl
```

Có thể thêm `--mapping data/annotations/map_vid_vidorID.json`. Mapping JSON là `annotation_video_id -> path_or_filename`, không thay đổi split. Adapter cũng nhận parquet **MC** từ [lmms-eval/NExTQA](https://huggingface.co/datasets/lmms-eval/NExTQA) nếu đúng các cột trên; không dùng parquet OE thay MC. Không lấy test để tune.

## STAR

Lấy `STAR_val.json` từ [repo STAR chính thức](https://github.com/csbobby/STAR_Benchmark#star-data-outline), và video Charades gốc từ [trang Charades](https://prior.allenai.org/projects/charades). Giữ video **nguyên bản**, không truyền các clip đã cắt với timestamp bắt đầu về 0.

```bash
python -m evidencelab prepare --dataset star --split val \
  --annotations /path/to/STAR_val.json --video-root /path/to/Charades_v1_480 \
  --output data/star-val.jsonl
```

Adapter sắp lựa chọn theo `choice_id`, ghép đáp án theo chuỗi `answer`, lấy loại câu từ `question_id`. Decoder chỉ cung cấp frame có timestamp trong **[start, end)**. `situations`, `question_program`, `choice_program`, bounding box, quan hệ và đáp án đúng không đi vào prompt hay bộ chọn khung. Split không có label cho phép suy luận, nhưng không thể tính accuracy hoặc paired comparison; repo chưa xuất định dạng nộp EvalAI.

## Định dạng manifest

JSONL có các trường `id,dataset,split,video,video_sha256,question,choices,answer,question_type,start,end`. `video` là đường dẫn tương đối dưới `--video-root`; `answer` là chỉ số 0-based hoặc null. CLI `prepare` là cách được khuyến nghị để tạo manifest. SHA256 cho biết chính xác bytes của video; đổi transcoding cần manifest/run mới.

Không bỏ dòng lỗi hoặc video bị thiếu một cách tự động vì điều đó làm thay đổi tập đánh giá. Nếu chỉ sở hữu một subset video, lọc annotation trước, giữ file lọc và mô tả subset rõ ràng.

Chế độ Slurm tùy chọn `DATA_MODE=local_archive` nhận ZIP/TAR qua `VIDEO_ARCHIVE` và annotation qua `ANNOTATIONS_FILE`, giải nén vào tmpfs mỗi phiên. Chỉ dùng khi bạn đã có archive trên vùng lưu trữ được cấp, hoặc nguồn RAM/external read-only đọc được từ node GPU. Với NExT-QA và login thiếu disk, dùng `nextqa_auto` để job tự tải vào RAM sau khi được cấp tài nguyên.
