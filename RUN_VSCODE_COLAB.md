# Chạy bài trong VS Code với Colab

Mở `lab_day10.ipynb`, chọn kernel Colab có T4 (16 GB trở lên). Chạy cell kiểm tra GPU trước khi tải model. Terminal PowerShell trên máy Windows vẫn dùng GPU local; chọn kernel Colab không tự chuyển terminal sang Colab.

Filesystem Colab khác máy Windows. Notebook được tạo bằng `notebook_builder.py` chứa gói mã nguồn nén ngay trong cell đầu và giải nén vào `/content/day10`. Vì vậy notebook hoàn thiện có thể chạy độc lập; clone repo đề gốc chỉ lấy khung ban đầu.

Các bước notebook: kiểm tra GPU → giải nén mã và cài dependencies → lấy dataset bằng URL zip công khai → audit → kiểm tra mask → E0 → E1/E2/E3 → E4 → ảnh và tổng hợp. Cell checkpoint dùng Drive nếu đã mount, nếu chưa thì dùng `work/adapters` trong runtime. Tải checkpoint và bằng chứng về máy bằng Colab sidebar → chuột phải tệp → Download.

Đã xác minh trên Colab: TorchAO có sẵn bản 0.10.0 gây lỗi khi PEFT 0.21.2 gắn LoRA vào các Linear thường của ViT ở E3. Notebook kiểm tra và gỡ TorchAO nếu bản thấp hơn 0.16.0 trong runtime Colab riêng; bài chỉ dùng bitsandbytes. Cấu hình lượng tử hóa cũng phải bỏ qua đường dẫn đầy đủ `model.visual`, vì `visual` không khớp nhánh này ở Transformers 5.19.0.

Adapter và projector lưu vào `CHECKPOINT_ROOT` sau mỗi 100 bước. Muốn suy luận lại checkpoint hoàn chỉnh, chạy action `infer` thay vì `run` (action `run` huấn luyện lại). Checkpoint giữa chừng chưa hỗ trợ khôi phục optimizer để tiếp tục đúng bước. `run_experiments.py --start <stage>` tiếp tục từ đầu một giai đoạn sau khi xử lý lỗi; tiến độ và log nằm trong `work/progress.json` và `work/logs/`.

Lệnh chạy từ Python kernel, không chạy từ terminal local:

```python
import subprocess, sys
subprocess.run([sys.executable, "lab.py", "run", "--config", "configs/e1.yaml",
                "--checkpoint-root", "/content/drive/MyDrive/day10/adapters"], check=True)
```

Sau khi có kết quả thật: điền REPORT.md, lấy bảng trong report/RESULTS.md, đọc sáu ảnh và output E0, rồi chạy scripts/check_submission.py. E5 không bắt buộc: có reward IoU và test; chưa triển khai GRPO.

Đã chạy E0 và huấn luyện đủ 300 bước E1 trên Tesla T4; pipeline E1–E4 đang tiếp tục. Xem `REPORT.md` và `report/evidence/` để phân biệt kết quả đã đo với phần còn chờ. Các lần smoke test một bước chỉ để kiểm tra môi trường, không phải bằng chứng nộp bài.

Bộ test gốc có một kiểm tra yêu cầu REPORT.md còn chuỗi `<điền`, mâu thuẫn với check_submission.py yêu cầu xóa chuỗi đó. Sau khi điền báo cáo, chỉ bỏ riêng kiểm tra mẫu bằng `python -m pytest -q -k "not test_report_template_has_student_markers_and_ten_questions"`; không sửa scripts/ hoặc tests/.
