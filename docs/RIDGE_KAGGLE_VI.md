# Chạy RIDGE trên Kaggle T4 ×2

Đã chuẩn bị notebook riêng cho **Molmo2-4B FP16**, chia trọng số lên hai GPU T4. Đây là thực nghiệm mới; không nối tiếp 376 câu trả lời BF16 trên server trường. Gói đã kiểm tra CPU, chưa chạy suy luận thật trên Kaggle.

## 1. Hai file cần dùng

- `outputs/ridge-kaggle-v1/RIDGE_Kaggle_T4x2.ipynb`: notebook để import.
- `outputs/ridge-kaggle-v1/ridge-kaggle-input.zip`: khoảng 7,2 MB, gồm code, score BLIP, trọng số ưu tiên Qwen, metadata và bản sao từ vựng tránh giải nén lồng nhau. Không chứa video, trọng số Molmo2 hoặc token. Dataset cũ khoảng 5,9 MB vẫn dùng được với notebook mới.

## 2. Chuẩn bị trên Kaggle

1. Tạo Dataset mới, upload `ridge-kaggle-input.zip`, đặt **Private**, rồi Create. Kaggle có thể tự giải nén ZIP; notebook hỗ trợ cả hai cách lưu.
2. Tạo notebook, dùng **File → Import Notebook** hoặc chức năng upload notebook, chọn `RIDGE_Kaggle_T4x2.ipynb`.
3. Trong Settings chọn **GPU T4 ×2**, bật **Internet**. Nếu chưa chọn được GPU/Internet, hoàn tất bước xác minh tài khoản mà Kaggle yêu cầu.
4. **Add Input**, tìm và gắn Dataset vừa tạo. Không cần upload video hoặc model.
5. Trong **Add-ons → Secrets**, tạo secret tên chính xác **HF_TOKEN**, nhập Hugging Face read token của tài khoản đã được cấp quyền LongVideoBench. Bật quyền dùng secret cho notebook này. Không dán token vào cell hoặc gửi token qua chat.

## 3. Chạy pilot trước

Chạy lần lượt các cell từ trên xuống. Cell môi trường cài venv riêng; lần đầu sẽ mất thời gian tải thư viện, model và lập chỉ mục video. Notebook tải từng video cần dùng, không tải toàn bộ kho video.

Bản notebook đã sửa lỗi kernel Python ngoài 3.10–3.12: cell đầu chỉ in phiên bản kernel, cell môi trường dùng uv tải **Python 3.12.11** riêng cho worker. Không cần đổi kernel hoặc sửa bộ thư viện đã ghim. Nếu đang dùng bản cũ báo AssertionError này, import lại notebook đã cập nhật và chạy từ đầu; dùng lại Dataset input cũ.

Nếu gặp thiếu `bpe_simple_vocab_16e6.txt.gz` ở cell đầu, dùng bản notebook mới nhất và chạy lại từ đầu với Internet bật. Bộ nạp tự khôi phục đúng file từ vựng từ bản sao an toàn trong gói mới, hoặc nguồn OpenAI CLIP ở commit cố định với gói cũ; chỉ ghi file khi SHA256 khớp bản gốc. Không cần tạo lại Dataset đã upload, không bỏ kiểm tra checksum.

Trong cell chạy cuối, giữ:

```python
MODE = 'pilot'
```

Pilot xử lý **2 câu × 5 biến thể = 10 câu trả lời**, mỗi lần dùng 64 frame. Kết quả mong đợi:

```text
completed_variant_answers: 10
target_variant_answers: 1000
complete: false
```

`complete=false` là bình thường ở pilot. Kiểm tra mỗi biến thể có `n=2`, không lỗi OOM, NaN hoặc device mismatch. Nếu lỗi, tải `session.log` và checkpoint nếu có để chẩn đoán trước khi thay cấu hình.

## 4. Chạy đủ dev200

Khi pilot thành công, sửa ngay trong cell đó:

```python
MODE = 'full'
```

Chỉ chạy lại cell này trong cùng session. Chương trình nạp lại model nhưng giữ 10 kết quả pilot; xử lý phần còn lại. Có 5 biến thể: `sigma2`, `sigma1`, `multiscale_v1`, `vote_only`, `rescue_only`. Không tính lại BLIP hoặc Qwen.

Hoàn tất khi `completed_variant_answers=1000`, `complete=true`, mỗi biến thể `n=200`. Mặc định worker tự dừng sau khoảng 6 giờ giữa các biến thể/câu hỏi; đây không phải cam kết đủ thời gian chạy toàn bộ. Theo dõi thời gian session/quota trong giao diện Kaggle.

## 5. Lưu và tiếp tục

Tải hai file trong Output hoặc lưu version có outputs:

```text
/kaggle/working/ridge-kaggle-v1/ridge-kaggle-checkpoint.zip
/kaggle/working/ridge-kaggle-v1/session.log
```

SQLite commit sau mỗi biến thể; ZIP cập nhật mỗi 5 câu và khi worker thoát bình thường. Khi cần dừng, dùng Interrupt rồi chờ worker lưu xong. Việc nền tảng tắt cưỡng bức có thể khiến ZIP chưa chứa các câu mới nhất; lưu outputs trước khi kết thúc session.

Để tiếp tục ở session mới: tạo private Dataset chứa ZIP checkpoint mới nhất, gắn Dataset này cùng input bundle gốc, chạy lại notebook theo thứ tự và chọn `MODE='full'`. Cell khôi phục hỗ trợ cả checkpoint ZIP và nội dung do Kaggle tự giải nén. Giữ nguyên code và môi trường; khác hợp đồng runtime/device map sẽ bị từ chối để tránh trộn kết quả.

So sánh các biến thể **trong cùng run FP16 này**. Không gộp kết quả với BF16 trên server hoặc diễn giải chênh lệch giữa hai nền tảng như tác động riêng của thuật toán sampling.
