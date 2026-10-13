Họ tên: <điền họ tên>
`dataset_version`: <điền, lấy từ data/splits/meta.json>

Trạng thái: mã E0–E4 đã được triển khai, chưa chạy thực nghiệm trên Colab GPU. Các ô kết quả còn trống có chủ đích; chưa có bằng chứng F1, CI95, VRAM hoặc thời gian huấn luyện. Chạy notebook và dùng `report/RESULTS.md` để cập nhật bảng sau khi có kết quả thật.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Đã kiểm tra cấu hình model thật bằng `runtime_check.py` trên meta device (không tải trọng số): tổng 4.437.815.808 tham số; visual gồm projector là 415.347.712; projector là 109.105.152; language_model gồm embedding là 4.022.468.096. Vì embedding đầu vào và lm_head dùng chung trọng số trong cấu hình này, không cộng lm_head thêm lần nữa. ViT không tính projector có 306.242.560 tham số. E1 chỉ học adapter LoRA của LLM; E2 thêm toàn bộ projector; E3 thêm LoRA của ViT. Regex khớp đúng 252 module LLM và 96 module ViT. Số tham số trainable thực tế sẽ lấy từ run.json sau khi gắn adapter, không dùng tổng tham số backbone làm số trainable.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Chưa chạy E0. Cần đọc `report/evidence/e0.json` và ít nhất 5 output trong `work/e0.jsonl`; chưa thể xác định hai lỗi phổ biến nhất hoặc đưa ví dụ output thực. JSON hợp lệ chỉ chứng minh đúng cấu trúc, không chứng minh đúng lớp và vị trí.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Quy đổi chuẩn: x_pixel=x_norm×width/1000, y_pixel=y_norm×height/1000. Với ảnh 640×360, box [400,400,600,600] tương ứng [256,144,384,216] pixel; kiểm tra CPU đã xác nhận ví dụ này. Nếu coi 400 là pixel thay vì tọa độ chuẩn hóa thì vị trí và kích thước bị đổi, IoU thường giảm mạnh, kéo F1 xuống. Chưa đo mức giảm thực tế. `finish_lab.py summary` sẽ chấm cùng output E0 theo cả norm1000 và pixel, ghi `report/coordinate_check.json`. Đây là kiểm tra thang chấm, không thay thế E4 đổi thang ở cả nhãn/prompt/huấn luyện.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mục tiêu là dự đoán JSON khi đã biết ảnh và yêu cầu. `encode_example` kiểm tra chuỗi prompt là tiền tố chính xác của chuỗi đầy đủ, rồi gán labels=-100 cho toàn bộ tiền tố; `collate` cũng gán -100 cho padding. Token answer và token kết thúc vẫn được giám sát. Tính loss trên prompt/token ảnh sẽ tối ưu việc tái tạo nội dung đầu vào không phải mục tiêu, làm giảm trọng số tương đối của lỗi định vị. `runtime_check.py` đã xác nhận ảnh 640×360 tạo 220 token ảnh, tất cả đều bị mask; labels giải mã thành JSON đúng cộng `<|im_end|>` và newline của template. Chưa chạy ablation về mask nên chưa có số đo tác động F1.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
QLoRA cập nhật adapter float, không tối ưu trực tiếp các trọng số đã đóng gói 4-bit bằng AdamW thông thường. Cấu hình bỏ qua visual và lm_head khi lượng tử hóa. E2/E3 mở các tham số visual.merger và visual.deepstack_merger_list, đưa chúng về float32 trước khi tạo optimizer. Float32 giữ độ chính xác của tham số/gradient cập nhật và tránh GradScaler phải unscale gradient fp16; forward vẫn dùng autocast fp16. Adapter trainable cũng giữ float32. Projector được lưu riêng vào projector.pt cùng adapter và bắt buộc nạp lại khi suy luận E2/E3; mã kiểm tra đủ tên tensor trước khi nạp.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau cùng phiên quay có nội dung rất giống nhau. Chia ngẫu nhiên dễ đưa các frame gần trùng vào train và test; model có thể nhớ cảnh thay vì tổng quát hóa sang phiên mới, dẫn đến F1 quá lạc quan. `lab.py audit` kiểm tra các tập session không giao nhau và đếm ảnh/box theo lớp từ split thực. Chưa tải dataset ở runtime nên chưa xác nhận số đếm của lần chạy này; các số in trong README là số của đề, không phải kết quả tự đo.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Chưa có kết quả huấn luyện nên chưa kết luận setup nào đáng dùng. E2 thêm 109.105.152 tham số projector học đầy đủ; E3 thêm adapter ViT và cần gradient qua visual backbone, nên dự kiến tốn bộ nhớ và thời gian hơn E1. Cần đối chiếu lợi ích F1 với VRAM/thời gian đo thật trong evidence, và xem CI95 trước khi kết luận cải thiện đáng tin. E4 chọn setup nền theo validation, không chọn theo test.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Chưa có output để đo độ rộng CI95. Sau khi chấm, độ rộng bằng cận trên trừ cận dưới. score.py bootstrap theo ảnh; nhiều ảnh cùng phiên vẫn có thể tương quan nên CI theo ảnh không hoàn toàn phản ánh bất định khi đổi phiên quay. Hai CI chồng nhau không tự chứng minh hai setup bằng nhau; muốn kiểm định chênh lệch nên bootstrap ghép cặp trên cùng ảnh hoặc theo session. Không diễn giải một chênh lệch F1 nhỏ là cải thiện chắc chắn chỉ từ một lần chạy.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Chưa có F1 theo lớp hoặc ảnh dự đoán, nên chưa xác định lớp kém nhất. `finish_lab.py figures --name e1` chọn 3 ảnh F1 thấp và 3 ảnh F1 cao, vẽ GT xanh và dự đoán đỏ; captions ghi image_id và F1 ảnh. Cần đọc ảnh thực để phân biệt bỏ sót, box lệch, nhầm lớp, bịa box và JSON hỏng; số mẫu/kích thước box chỉ là giả thuyết giải thích nếu chưa kiểm chứng.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Chọn biến rank: r=4 và r=64, alpha=2r theo đề. `finish_lab.py e4` chọn setup nền có F1 validation cao nhất trong E1–E3 rồi tạo hai config bằng deepcopy; giữ nguyên learning rate, batch, số bước, seed, targets và trạng thái projector. Chưa chạy nên chưa có kết quả hoặc kết luận thực nghiệm. So sánh cả hai evidence và CI95; một seed và một test nhỏ chưa đủ kết luận rank cao luôn tốt hơn.

## Nhìn lại

Phần CPU đã kiểm tra nhãn, reward, ghép batch, tên module model và mask loss bằng processor thật. Phần GPU chưa được xác minh: mức VRAM, tính hữu hạn của loss ở E2/E3, thời gian và chất lượng sau 300 bước. Kernel Colab có filesystem riêng; cần đưa mã lên runtime. Checkpoint hiện lưu adapter/projector và log, chưa lưu optimizer để resume chính xác. E5 là tùy chọn: mới có reward IoU và test, chưa có GRPO hay bằng chứng bonus. Báo cáo vẫn cần họ tên, dataset_version và kết quả thật trước khi nộp.
