# Chạy bài trong VS Code với Colab

Mở `lab_day10.ipynb`, chọn kernel Colab có T4 (16 GB trở lên). Chạy cell kiểm tra GPU trước khi tải model. Terminal PowerShell trên máy Windows vẫn dùng GPU local; chọn kernel Colab không tự chuyển terminal sang Colab.

Filesystem Colab khác máy Windows. Đưa toàn bộ mã đã hoàn thiện lên runtime (hoặc clone repo của bạn sau khi đẩy mã), rồi đặt `PROJECT` trong cell đầu thành đường dẫn thư mục đó. Clone repo đề gốc sẽ chỉ lấy notebook khung, không có lời giải mới ở máy này.

Các bước notebook: cài dependencies → kiểm tra GPU → mount Drive → lấy dataset bằng Secrets/zip → audit → kiểm tra mask → E0 → E1/E2/E3 → E4 → ảnh và tổng hợp.

Đã xác minh trên Colab: TorchAO có sẵn bản 0.10.0 gây lỗi khi PEFT 0.21.2 gắn LoRA vào các Linear thường của ViT ở E3. Notebook kiểm tra và gỡ TorchAO nếu bản thấp hơn 0.16.0 trong runtime Colab riêng; bài chỉ dùng bitsandbytes. Cấu hình lượng tử hóa cũng phải bỏ qua đường dẫn đầy đủ `model.visual`, vì `visual` không khớp nhánh này ở Transformers 5.19.0.

Adapter và projector lưu vào `CHECKPOINT_ROOT` trên Drive sau mỗi 100 bước. Muốn suy luận lại checkpoint hoàn chỉnh, chạy action `infer` thay vì `run` (action `run` huấn luyện lại). Checkpoint giữa chừng được lưu để bảo toàn tiến độ, nhưng hiện chưa hỗ trợ khôi phục trạng thái optimizer để tiếp tục đúng bước.

Lệnh chạy từ Python kernel, không chạy từ terminal local:

```python
import subprocess, sys
subprocess.run([sys.executable, "lab.py", "run", "--config", "configs/e1.yaml",
                "--checkpoint-root", "/content/drive/MyDrive/day10/adapters"], check=True)
```

Sau khi có kết quả thật: điền REPORT.md, lấy bảng trong report/RESULTS.md, đọc sáu ảnh và output E0, rồi chạy scripts/check_submission.py. E5 không bắt buộc: có reward IoU và test; chưa triển khai GRPO.

Chưa chạy GPU nên chưa có evidence, F1, CI95, nhật ký VRAM hay ảnh dự đoán thật. Không dùng số minh họa làm bằng chứng nộp bài.
