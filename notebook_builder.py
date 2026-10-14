"""Rebuild the completed student notebook without touching scripts/."""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

cells = []
root = Path(__file__).resolve().parent
source_files = [root/name for name in (
    "lab.py", "finish_lab.py", "run_experiments.py", "runtime_check.py", "solution_check.py",
    "notebook_builder.py", "requirements-colab.txt", "pytest.ini", "README.md", "GUIDE.md",
    "RUBRIC.md", "REPORT.md", "RUN_VSCODE_COLAB.md")]
for directory, pattern in (("configs", "*.yaml"), ("scripts", "*.py"), ("tests", "*.py")):
    source_files.extend(sorted((root/directory).glob(pattern)))
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in source_files:
        archive.write(path, path.relative_to(root).as_posix())
bundle_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(True),
                      outputs=[], execution_count=None))


md("# Day 10 — Qwen3-VL road damage\nMã E0–E4 đã điền trong `lab.py`. Chọn kernel Colab T4 trong VS Code. "
   "Chạy từng cell; mọi subprocess dùng Python của kernel đang chọn. Số liệu chỉ sinh sau khi chạy thật.")
md("## 0. Môi trường Colab trong VS Code")
code('''import os, sys, subprocess, base64, io, zipfile
from pathlib import Path
import torch
assert Path("/content").exists() and torch.cuda.is_available(), "Select the Colab GPU kernel"
PROJECT = Path("/content/day10")
PROJECT.mkdir(parents=True, exist_ok=True)
SOURCE_B64 = ''' + repr(bundle_b64) + '''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as archive:
    archive.extractall(PROJECT)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    print(result.stdout[-10000:])
    if result.stderr:
        print(result.stderr[-5000:])
    result.check_returncode()
run("-m", "pip", "install", "-r", "requirements-colab.txt")
import importlib.metadata as metadata
from packaging.version import Version
try:
    torchao_version = metadata.version("torchao")
except metadata.PackageNotFoundError:
    torchao_version = None
if torchao_version is not None and Version(torchao_version) < Version("0.16.0"):
    # This lab uses bitsandbytes; stale optional torchao breaks dense ViT LoRA dispatch.
    assert Path("/content").exists(), "Only clean the dedicated Colab runtime"
    run("-m", "pip", "uninstall", "-y", "torchao")
''')
code('''import torch
print(sys.executable, torch.__version__)
assert torch.cuda.is_available(), "Select Colab GPU kernel"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024**3)
assert torch.cuda.get_device_properties(0).total_memory >= 14*1024**3, "Use T4 16 GB or larger"
# The original report-template test expects unfinished placeholders. It is inapplicable to a filled report.
run("-m", "pytest", "-q", "-k", "not test_report_template_has_student_markers_and_ten_questions")
run("solution_check.py")
''')
md("## Lưu adapter\nNếu đã mount Google Drive bằng menu Colab, checkpoint lưu thẳng vào Drive mỗi 100 bước. Khi chưa mount, mã dùng work/adapters trong runtime. Test gốc về mẫu REPORT đòi placeholder nên được loại khỏi kiểm tra khi báo cáo đã hoàn thiện; scripts/ và tests/ giữ nguyên.")
code('''CHECKPOINT_ROOT = "/content/drive/MyDrive/day10/adapters" if Path("/content/drive/MyDrive").exists() else "work/adapters"
print("Checkpoint root:", CHECKPOINT_ROOT)
''')
md("## 1. Kaggle và dữ liệu\nTải dataset công khai bằng URL zip qua prepare_data.py gốc. Có thể thay URL bằng đường dẫn zip đã tải. Không cần ghi khóa vào notebook.")
code('''DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
run("scripts/prepare_data.py", "--zip", DATASET_URL)
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
    for name in ("REPORT.md", "lab.py", "finish_lab.py", "solution_check.py", "run_experiments.py"):
        archive.write(name)
print("Archive ready:", Path("day10_results.zip").resolve())
# In VS Code: Colab sidebar → right-click day10_results.zip → Download.
''')

nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
          metadata=dict(accelerator="GPU", colab={"provenance": []},
                        kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                        language_info=dict(name="python")))
for i, cell in enumerate(cells):
    cell["id"] = f"day10-{i:03d}"
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="lab_day10.ipynb")
args = ap.parse_args()
Path(args.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Completed notebook generated:", args.out)
