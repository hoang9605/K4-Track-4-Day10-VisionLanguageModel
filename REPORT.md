Họ tên: NguyenHaiHoang
`dataset_version`: d10-21e4070301

Trạng thái: đang chạy thực nghiệm E0–E4 trên Colab Tesla T4. E0 và E1 đã chấm test đầy đủ; E2 đang huấn luyện. Các ô chưa có kết quả vẫn để trống cho đến khi có bằng chứng thực.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,0631 | [0,0429; 0,0869] | 1,0868 | 0,0719 | 0,0177 | 0,1626 |
| E1 | LoRA LLM | 33.030.144 | 8,03 | 45,13 | 0,2528 | [0,2187; 0,2913] | 1,3698 | 0,1818 | 0,1798 | 0,5224 |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Đã kiểm tra cấu hình model thật bằng `runtime_check.py` trên meta device (không tải trọng số): tổng 4.437.815.808 tham số; visual gồm projector là 415.347.712; projector là 109.105.152; language_model gồm embedding là 4.022.468.096. Vì embedding đầu vào và lm_head dùng chung trọng số trong cấu hình này, không cộng lm_head thêm lần nữa. ViT không tính projector có 306.242.560 tham số. E1 chỉ học adapter LoRA của LLM; E2 thêm toàn bộ projector; E3 thêm LoRA của ViT. Regex khớp đúng 252 module LLM và 96 module ViT. Số tham số trainable thực tế sẽ lấy từ run.json sau khi gắn adapter, không dùng tổng tham số backbone làm số trainable.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có parse_valid_rate=0,9164: 285/311 output hợp lệ và 26 output không parse được. Hai nhóm lỗi thể hiện trong bằng chứng là bỏ sót đối tượng (553 FN, recall=0,0515) và dự đoán không khớp lớp/IoU (338 FP, riêng crack có 262 FP và 294 FN). Một lỗi cấu trúc quan sát trực tiếp là lặp khóa bbox_2d rồi chuỗi bị cắt trước dấu đóng. Ví dụ nguyên văn của ảnh vlcsnap-00201: `[{"label":"crack","bbox_2d":[0, 457, 1000, 999]}]`. Output này hợp lệ về JSON nhưng box rất rộng; tính hợp lệ không chứng minh đúng vị trí. Output thô đầy đủ được lưu trong work/e0.jsonl; nguyên nhân thị giác cần đối chiếu ảnh và GT.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Quy đổi chuẩn: x_pixel=x_norm×width/1000, y_pixel=y_norm×height/1000. Với ảnh 640×360, box [400,400,600,600] tương ứng [256,144,384,216] pixel; kiểm tra CPU đã xác nhận ví dụ này. Đã chấm cùng 311 output E0 bằng evaluate của bộ chấm gốc: F1=0,0630915 với norm1000 và F1=0 với pixel, lưu trong report/coordinate_check.json. Đọc nhầm thang làm đổi vị trí/kích thước box và mất các cặp IoU≥0,5. Đây là kiểm tra thang chấm, không thay thế E4 đổi thang ở cả nhãn/prompt/huấn luyện.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mục tiêu là dự đoán JSON khi đã biết ảnh và yêu cầu. `encode_example` kiểm tra chuỗi prompt là tiền tố chính xác của chuỗi đầy đủ, rồi gán labels=-100 cho toàn bộ tiền tố; `collate` cũng gán -100 cho padding. Token answer và token kết thúc vẫn được giám sát. Tính loss trên prompt/token ảnh sẽ tối ưu việc tái tạo nội dung đầu vào không phải mục tiêu, làm giảm trọng số tương đối của lỗi định vị. `runtime_check.py` đã xác nhận ảnh 640×360 tạo 220 token ảnh, tất cả đều bị mask; labels giải mã thành JSON đúng cộng `<|im_end|>` và newline của template. Chưa chạy ablation về mask nên chưa có số đo tác động F1.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
QLoRA cập nhật adapter float, không tối ưu trực tiếp các trọng số đã đóng gói 4-bit bằng AdamW thông thường. Cấu hình dùng đường dẫn đầy đủ `llm_int8_skip_modules=["model.visual", "lm_head"]`. Chạy thử trên Colab đã phát hiện `"visual"` trong pseudo code không khớp các tên `model.visual.*` ở Transformers 5.19.0: quantizer vẫn trả `should_convert_module=True`; E2 báo `Attempting to unscale FP16 gradients` và số trainable sai. Đổi sang `model.visual` khiến quy tắc trả False cho cả ViT và hai nhóm merger; mã còn kiểm tra không có Linear4bit trong visual ngay sau nạp model. E2/E3 mở các tham số visual.merger và visual.deepstack_merger_list, đưa chúng về float32 trước optimizer. Float32 giữ độ chính xác cập nhật và cho phép GradScaler unscale gradient; forward vẫn autocast fp16. Adapter trainable cũng giữ float32. Projector được lưu riêng vào projector.pt và bắt buộc nạp lại khi suy luận E2/E3, có kiểm tra đủ tên tensor. Các lần chạy thử chỉ dùng để gỡ lỗi, không được nộp như evidence thực nghiệm.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau cùng phiên quay có nội dung rất giống nhau. Chia ngẫu nhiên dễ đưa các frame gần trùng vào train và test; model có thể nhớ cảnh thay vì tổng quát hóa sang phiên mới, dẫn đến F1 quá lạc quan. Đã chạy prepare_data.py và lab.py audit trên dataset thật tại Colab, phiên bản d10-21e4070301. Train có 1.422 ảnh, 1.074 pothole, 1.750 crack, 691 manhole; validation có 276 ảnh, 66 pothole, 470 crack, 103 manhole; test có 311 ảnh, 121 pothole, 299 crack, 163 manhole. Kiểm tra các tập session không giao nhau đã đạt. Validation chỉ có 66 box pothole nên việc chọn setup dựa trên micro F1 cần đối chiếu thêm F1 từng lớp.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đã có test F1=0,2528, CI95=[0,2187; 0,2913], VRAM=8,0295 GB và thời gian=45,1327 phút. Precision tăng từ 0,0815 lên 0,2552, recall từ 0,0515 lên 0,2504; tuy nhiên FP/ảnh tăng từ 1,0868 lên 1,3698 do model dự đoán nhiều hơn. E2/E3 chưa chấm xong nên chưa kết luận việc mở thêm khối có đáng dùng. E2 thêm 109.105.152 tham số projector học đầy đủ; E3 thêm adapter ViT và cần gradient qua visual backbone, nên dự kiến tốn bộ nhớ và thời gian hơn E1. Cần đối chiếu lợi ích F1 với VRAM/thời gian đo thật trong evidence, và xem CI95 trước khi kết luận cải thiện đáng tin. E4 chọn setup nền theo validation, không chọn theo test.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
E0 có CI95=[0,0429; 0,0869], độ rộng 0,0440, tương đương khoảng 70% F1 điểm 0,0631; mức bất định tương đối lớn. E1 có CI95=[0,2187; 0,2913], độ rộng 0,0726. E1 tăng F1 0,1897 và CI của E1 nằm trên CI E0; đây là bằng chứng cải thiện ở split này, nhưng chưa phải kiểm định ghép cặp. E2–E4 sẽ đối chiếu khi chấm xong. score.py bootstrap theo ảnh; nhiều ảnh cùng phiên vẫn có thể tương quan nên CI theo ảnh không hoàn toàn phản ánh bất định khi đổi phiên quay. Hai CI chồng nhau không tự chứng minh hai setup bằng nhau; muốn kiểm định chênh lệch nên bootstrap ghép cặp trên cùng ảnh hoặc theo session. Không diễn giải một chênh lệch F1 nhỏ là cải thiện chắc chắn chỉ từ một lần chạy.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E0, crack kém nhất: F1=0,0177 so với pothole=0,0719 và manhole=0,1626. Tính từ nhãn test, diện tích box trung vị là pothole 0,4193%, crack 1,9813%, manhole 0,6688% khung hình; train có lần lượt 1.074, 1.750, 691 box (report/box_statistics.json). Crack có nhiều nhãn và box trung vị lớn hơn hai lớp kia, nên không đủ bằng chứng quy lỗi chỉ cho ít mẫu hoặc box nhỏ. Ví dụ vlcsnap-00201: crack GT=[558,736,731,825], nhưng E0 trả [0,457,1000,999] và bỏ sót manhole GT=[856,725,919,789]. Đây là lỗi định vị và bỏ sót thực tế; giả thuyết model gộp vùng nứt/vệt mặt đường rộng cần đối chiếu ảnh. `finish_lab.py figures` sẽ tạo 3 ảnh F1 thấp và 3 ảnh F1 cao của setup chọn bằng validation, vẽ GT xanh và dự đoán đỏ, kèm image_id/F1.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Chọn biến rank: r=4 và r=64, alpha=2r theo đề. `finish_lab.py e4` chọn setup nền có F1 validation cao nhất trong E1–E3 rồi tạo hai config bằng deepcopy; giữ nguyên learning rate, batch, số bước, seed, targets và trạng thái projector. Chưa chạy nên chưa có kết quả hoặc kết luận thực nghiệm. So sánh cả hai evidence và CI95; một seed và một test nhỏ chưa đủ kết luận rank cao luôn tốt hơn.

## Sáu ảnh dự đoán E1 đã kiểm tra

GT màu xanh, dự đoán màu đỏ. Ba ảnh đầu được chọn từ nhóm F1 ảnh thấp nhất, ba ảnh cuối từ nhóm cao nhất, nên đây là ví dụ có chủ đích để phân tích lỗi, không phải mẫu ngẫu nhiên đại diện toàn test. F1 tổng thể lấy từ toàn bộ 311 ảnh.

Ảnh vlcsnap-00201, F1=0: E1 gán crack cho vùng gần manhole và bỏ sót crack GT bên trái. IoU giữa crack dự đoán với crack GT bằng 0; cả hai GT đều không được match đúng lớp.

![E1: box crack lệch vị trí và bỏ sót manhole](report/figures/e1_1.png)

Ảnh vlcsnap-00203, F1=0: hai box crack nhỏ nằm phía trên vùng crack GT rộng; IoU tốt nhất chỉ khoảng 0,002. Lỗi là vị trí và phạm vi vùng hư hại, dù JSON và nhãn lớp đều hợp lệ.

![E1: dự đoán các box nhỏ lệch vùng crack](report/figures/e1_2.png)

Ảnh vlcsnap-00207, F1=0: model nhận đúng vùng crack nhưng box nằm trong GT và quá nhỏ, IoU≈0,394<0,5. Một dự đoán nhìn có vẻ hợp lý vẫn bị tính một FP và một FN theo ngưỡng chấm.

![E1: box crack quá nhỏ, IoU dưới ngưỡng](report/figures/e1_3.png)

Ảnh vlcsnap-2025-02-19-17h14m33s249, F1=1: box manhole đúng lớp, đủ IoU và không có dự đoán thừa.

![E1: manhole được định vị đúng](report/figures/e1_4.png)

Ảnh vlcsnap-2025-02-19-17h14m56s176, F1=1: box manhole khớp vùng mục tiêu; khác biệt nhỏ ở biên không làm mất match.

![E1: biên box manhole đủ khớp GT](report/figures/e1_5.png)

Ảnh vlcsnap-2025-02-19-17h16m03s525, F1=1: pothole gần giữa mặt đường được dự đoán đúng, không có box thừa.

![E1: pothole được định vị đúng](report/figures/e1_6.png)

## Nhìn lại

Phần CPU đã kiểm tra nhãn, reward, ghép batch, tên module model và mask loss bằng processor thật. Smoke test GPU của E1/E2/E3 đều qua sau khi sửa quy tắc bỏ qua visual và gỡ TorchAO cũ; smoke không dùng làm evidence. E1 đủ 300 bước: loss từ 1,1483 xuống 0,5433, 33.030.144 tham số trainable, VRAM đỉnh 8,0295 GB, 45,1327 phút. Validation E1 có F1=0,1872, parse_valid_rate=0,9819; Test E1 đạt F1=0,2528, parse_valid_rate=0,9936, không thiếu dự đoán; F1 tăng 0,1897 so với E0, đạt ngưỡng ≥0,10 của rubric. E2–E4 đang chạy. Kernel Colab có filesystem riêng; notebook hoàn thiện chứa gói mã nguồn để giải nén vào runtime. Checkpoint hiện lưu adapter/projector và log, chưa lưu optimizer để resume chính xác; checkpoint E1 đủ 300 bước đã tải về máy và kiểm tra CRC ZIP. E5 là tùy chọn: mới có reward IoU và test, chưa có GRPO hay bằng chứng bonus. Báo cáo còn chờ kết quả test E2–E4 trước khi nộp.
