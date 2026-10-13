"""Rebuild the completed student notebook without touching scripts/."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(True),
                      outputs=[], execution_count=None))


md("# Day 10 — Qwen3-VL road damage\nMã E0–E4 đã điền trong `lab.py`. Chọn kernel Colab T4 trong VS Code. "
   "Chạy từng cell; mọi subprocess dùng Python của kernel đang chọn. Số liệu chỉ sinh sau khi chạy thật.")
md("## 0. Môi trường Colab trong VS Code")
code('''import os, sys, subprocess
from pathlib import Path
# Colab remote filesystem is separate from the Windows workspace.
# Upload lab.py, finish_lab.py, solution_check.py, configs/, scripts/, tests/ and requirements first,
# or clone YOUR repository containing these completed files into the runtime.
PROJECT = Path.cwd()  # set to the uploaded/cloned project directory if needed
os.chdir(PROJECT)
assert Path("lab.py").exists(), "Upload completed project files to the remote Colab runtime first"
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)
run("-m", "pip", "install", "-r", "requirements-colab.txt")
''')
code('''import torch
print(sys.executable, torch.__version__)
assert torch.cuda.is_available(), "Select Colab GPU kernel"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024**3)
assert torch.cuda.get_device_properties(0).total_memory >= 14*1024**3, "Use T4 16 GB or larger"
run("-m", "pytest", "-q")
run("solution_check.py")
''')
md("## Lưu adapter vào Google Drive\nCheckpoint được ghi thẳng vào Drive mỗi 100 bước, gồm LoRA và projector. Nếu chưa có Drive, đổi `CHECKPOINT_ROOT` sang thư mục bền vững của runtime.")
code('''from google.colab import drive
drive.mount("/content/drive")
CHECKPOINT_ROOT = "/content/drive/MyDrive/day10/adapters"
''')
md("## 1. Kaggle và dữ liệu\nDùng Colab Secrets. Nếu secrets không khả dụng trong kết nối VS Code, dùng dataset zip và `scripts/prepare_data.py --zip PATH`. Không ghi khóa trong notebook.")
code('''from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
run("scripts/prepare_data.py", "--kaggle")
run("lab.py", "audit")
''')
md("## 2. Nhãn YOLO sang JSON\n`objects_to_answer` đổi tọa độ, kẹp biên, sắp theo y1/x1. `solution_check.py` kiểm tra quy đổi norm1000 và pixel.")
md("Kiểm tra 220 token ảnh và mask loss trước khi huấn luyện:")
code('''from lab import encode_example, MODEL_ID
from scripts.labcommon import read_jsonl
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_ID)
entry = read_jsonl("data/splits/train.jsonl")[0]
encoded = encode_example(processor, entry, "data", "norm1000")
image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
n_image = int((encoded["input_ids"] == image_id).sum())
print("Vision tokens:", n_image)
assert n_image == 220, f"Unexpected resize/token count: {n_image}"
print(processor.tokenizer.decode(encoded["labels"][encoded["labels"] != -100]))
del encoded, processor
''')
md("## 3. E0 — zero-shot")
code('run("lab.py", "run", "--config", "configs/e0.yaml", "--checkpoint-root", CHECKPOINT_ROOT)')
md("## 4. Ví dụ huấn luyện và ghép batch\n`encode_example` chỉ giám sát answer; `collate` right-pad các chuỗi, nối pixel_values/image_grid_thw mà không pad ảnh. Kiểm tra CPU chạy ở bước 0.")
md("## 5. Setup LoRA\n`apply_setup` xác nhận 252 module LLM hoặc 348 module LLM+ViT; E2/E3 mở projector ở float32.")
md("## 6. E1–E3 — SFT, validation và test\nCó thể chạy từng thí nghiệm riêng. Khi OOM, đổi cùng một chế độ batch_size=1, grad_accum=8 ở cả E1–E3 trước khi chạy. Không chỉnh chỉ một setup rồi coi là so sánh công bằng.")
for name in ("e1", "e2", "e3"):
    md(f"### {name.upper()}")
    code(f'''run("lab.py", "train", "--config", "configs/{name}.yaml", "--checkpoint-root", CHECKPOINT_ROOT)
run("lab.py", "infer", "--config", "configs/{name}.yaml", "--checkpoint-root", CHECKPOINT_ROOT, "--split", "val")
run("lab.py", "infer", "--config", "configs/{name}.yaml", "--checkpoint-root", CHECKPOINT_ROOT)
''')
md("## 7. Suy luận và chấm E1–E3\nCác cell phía trên nạp lại checkpoint để suy luận validation và test, gồm projector E2/E3. Evidence được sinh bởi score.py gốc của đề.")
md("## 8. E4 — rank 4 và 64\nChọn setup nền theo validation, giữ nguyên mọi thứ ngoài rank và alpha=2r.")
code('''run("finish_lab.py", "e4")
for rank in (4, 64):
    run("lab.py", "run", "--config", f"configs/e4_rank_{rank}.yaml", "--checkpoint-root", CHECKPOINT_ROOT)
''')
md("## Tổng hợp, sáu ảnh đúng/sai và kiểm tra bài nộp\nĐọc output E0, ảnh và số liệu để hoàn thiện các câu phân tích thực nghiệm trong REPORT.md. "
   "E5 GRPO là tùy chọn: đã có reward IoU và test, chưa có vòng GRPO.")
code('''run("finish_lab.py", "summary")
run("finish_lab.py", "figures", "--name", "e1")
from IPython.display import display, Markdown
display(Markdown(Path("report/RESULTS.md").read_text()))
for row in read_jsonl("work/e0.jsonl")[:5]:
    print(row)
run("scripts/check_submission.py")
''')
md("## Tải kết quả về máy\nArchive chứa code, evidence, ảnh, báo cáo; loại data/work/weights. Cần hoàn thiện REPORT.md bằng kết quả thật và thông tin cá nhân trước khi nộp.")
code('''import zipfile
with zipfile.ZipFile("day10_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ("report", "configs"):
        for path in Path(name).rglob("*"):
            if path.is_file():
                archive.write(path)
    for name in ("REPORT.md", "lab.py", "finish_lab.py", "solution_check.py", "lab_day10.ipynb"):
        archive.write(name)
from google.colab import files
files.download("day10_results.zip")
''')

nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
          metadata=dict(accelerator="GPU", colab={"provenance": []},
                        kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                        language_info=dict(name="python")))
for i, cell in enumerate(cells):
    cell["id"] = f"day10-{i:03d}"
Path("lab_day10.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Completed lab_day10.ipynb generated")
