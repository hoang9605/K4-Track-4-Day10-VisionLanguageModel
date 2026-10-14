"""Student implementation: python lab.py --help. GPU operations import lazily."""
import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import iou_xyxy, parse_output

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
UNQUANTIZED_MODULES = ["model.visual", "lm_head"]
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    boxes = []
    for obj in objects:
        xc, yc, w, h = obj["yolo"]
        values = [(xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy]
        box = [max(0, min(limit, round(v))) for v, limit in zip(values, (sx, sy, sx, sy))]
        boxes.append({"label": obj["label"], "bbox_2d": box})
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def prompt(coord="norm1000"):
    scale = "integer coordinates normalized to 0-1000 independently for image width and height" if coord == "norm1000" else "integer pixel coordinates in the original image"
    return ("Locate all road damage: pothole, crack, manhole. Return ONLY a JSON list of "
            '{"label":"pothole|crack|manhole","bbox_2d":[x1,y1,x2,y2]}. '
            f"Use {scale}, top-left and bottom-right corners. Return [] if none.")


def user_message(image, coord):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt(coord)}]}


def encode_example(processor, entry, data_dir, coord):
    import torch
    from PIL import Image
    with Image.open(Path(data_dir) / entry["image_path"]) as source:
        image = source.convert("RGB")
    user = user_message(image, coord)
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template([user, {"role": "assistant", "content": answer}],
                                          add_generation_prompt=False, **kwargs)
    n = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :n], prefix["input_ids"]):
        raise ValueError("Chat template prefix mismatch: cannot safely mask loss")
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    return dict(full)


def collate(examples, pad_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    for key in examples[0]:
        values = [e[key] for e in examples]
        if key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels", "token_type_ids"):
            pad = -100 if key == "labels" else pad_id if key == "input_ids" else 0
            batch[key] = pad_sequence([v.squeeze(0) for v in values], batch_first=True, padding_value=pad)
        else:
            batch[key] = torch.cat(values, dim=0)
    return batch


def reward(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.
    if not pred and not gt_boxes:
        return 1.
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.
    for value, i, j in pairs:
        if value < .5:
            break
        if i not in used_g and j not in used_p:
            total += value
            used_g.add(i)
            used_p.add(j)
    return .2 + .8 * total / max(len(gt_boxes), len(pred), 1)


def load_base(config):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Connect a Colab T4 GPU kernel before running experiments")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=UNQUANTIZED_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.get("model_id", MODEL_ID), quantization_config=quant,
        dtype=torch.float16, device_map={"": 0})
    from bitsandbytes.nn import Linear4bit
    quantized_visual = [name for name, module in model.model.visual.named_modules()
                        if isinstance(module, Linear4bit)]
    if quantized_visual:
        raise RuntimeError(f"Visual must remain dense fp16; quantized modules: {quantized_visual[:5]}")
    processor = AutoProcessor.from_pretrained(config.get("model_id", MODEL_ID))
    processor.tokenizer.padding_side = "left"
    return model, processor


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def apply_setup(model, config):
    import torch
    from peft import LoraConfig, get_peft_model
    from peft import prepare_model_for_kbit_training
    # Freeze and cast non-quantized weights, then keep frozen ViT in fp16.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.visual.to(dtype=torch.float16)
    lora = config["lora"]
    regex = LLM_TARGET if lora["targets"] == "llm" else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    matched = [name for name, _ in model.named_modules() if re.fullmatch(regex, name)]
    expected = 252 if lora["targets"] == "llm" else 348
    if len(matched) != expected:
        raise RuntimeError(f"Expected {expected} LoRA modules, got {len(matched)}; inspect model tree")
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                              lora_dropout=lora["dropout"], target_modules=regex,
                              task_type="CAUSAL_LM"))
    for name, param in model.named_parameters():
        if is_projector(name) and config["train_projector"]:
            param.data = param.data.float()
            param.requires_grad = True
        # GradScaler requires fp32 trainable parameters (including ViT adapters).
        if param.requires_grad:
            param.data = param.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, matched


def save_checkpoint(model, processor, directory, run):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    extra = {n: p.detach().cpu() for n, p in model.named_parameters()
             if p.requires_grad and is_projector(n)}
    if extra:
        torch.save(extra, directory / "projector.pt")
    (directory / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


def train(config, data_dir, directory):
    import torch
    from torch.utils.data import DataLoader
    from transformers import set_seed
    set_seed(config.get("seed", 42))
    model, processor = load_base(config)
    model, matched = apply_setup(model, config)
    entries = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    fraction = config.get("train_fraction", 1.)
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    random.Random(config.get("seed", 42)).shuffle(entries)
    entries = entries[:max(1, round(len(entries)*fraction))]
    coord = config.get("coord", "norm1000")
    # Lazy encoding avoids retaining the entire image tensor dataset in RAM.
    class Examples:
        def __len__(self):
            return len(entries)

        def __getitem__(self, i):
            return encode_example(processor, entries[i], data_dir, coord)

    loader = DataLoader(Examples(), batch_size=config["batch_size"], shuffle=True,
                        collate_fn=partial(collate, pad_id=processor.tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=0.)
    steps, accum = config["max_steps"], config["grad_accum"]
    warmup = max(1, round(.05*steps))
    def lr_factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    iterator = iter(loader)
    logs = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps+1):
        loss_sum = 0.
        for _ in range(accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss_sum += loss.item()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        logs.append({"step": step, "loss": loss_sum, "lr": scheduler.get_last_lr()[0]})
        if step == 1 or step % 10 == 0:
            print(f"{config['name']} step={step}/{steps} loss={loss_sum:.4f}", flush=True)
        if step % 100 == 0 or step == steps:
            run = dict(config=config, trainable_params=sum(p.numel() for p in params),
                       peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3,
                       train_minutes=(time.perf_counter()-started)/60,
                       matched_modules=matched, n_train=len(entries), steps_completed=step,
                       loss_log=logs,
                       dataset_version=json.loads((Path(data_dir)/"splits/meta.json").read_text())["dataset_version"])
            save_checkpoint(model, processor, directory, run)
    del model, optimizer, params, loader, iterator, batch, loss
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def infer(config, data_dir, directory, output, split="test", limit=None):
    import torch
    from PIL import Image
    from peft import PeftModel
    model, processor = load_base(config)
    if "lora" in config:
        saved = json.loads((Path(directory)/"run.json").read_text())
        current_version = json.loads((Path(data_dir)/"splits/meta.json").read_text())["dataset_version"]
        if saved["dataset_version"] != current_version or saved["config"] != config:
            raise ValueError("Checkpoint config or dataset_version mismatch")
        if saved["steps_completed"] != config["max_steps"]:
            raise ValueError("Incomplete checkpoint: finish training before scoring")
        model = PeftModel.from_pretrained(model, directory)
        if config["train_projector"]:
            extra = torch.load(Path(directory)/"projector.pt", map_location="cpu", weights_only=True)
            expected = {n for n, _ in model.named_parameters() if is_projector(n)}
            if set(extra) != expected:
                raise ValueError("Missing or unexpected projector weights")
            for name, param in model.named_parameters():
                if name in extra:
                    param.data = extra[name].to(device=param.device, dtype=torch.float32)
    model.eval()
    model.config.use_cache = True
    entries = read_jsonl(Path(data_dir)/f"splits/{split}.jsonl")
    if limit is not None:
        entries = entries[:limit]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f, torch.inference_mode():
        size = config.get("inference_batch_size", 4)
        for start in range(0, len(entries), size):
            group = entries[start:start+size]
            images = []
            for e in group:
                with Image.open(Path(data_dir)/e["image_path"]) as im:
                    images.append(im.convert("RGB"))
            conversations = [[user_message(im, config.get("coord", "norm1000"))] for im in images]
            inputs = processor.apply_chat_template(conversations, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**inputs, max_new_tokens=config.get("max_new_tokens", 256), do_sample=False)
            raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, text in zip(group, raw):
                f.write(json.dumps({"image_id": e["image_id"], "raw": text})+"\n")
            f.flush()
            print(f"{config['name']} {split}: {start+len(group)}/{len(entries)}", flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def score(config, data_dir, directory, predictions, evidence, split):
    args = [sys.executable, "scripts/score.py", "--pred", str(predictions), "--name", config["name"],
            "--out", str(evidence), "--splits-dir", str(Path(data_dir)/"splits"), "--split", split,
            "--coord", config.get("coord", "norm1000")]
    if "lora" in config:
        run = json.loads((Path(directory)/"run.json").read_text())
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            args += ["--meta", f"{key}={run[key]}"]
    subprocess.run(args, check=True)


def audit(data_dir, output="report/figures"):
    from PIL import Image, ImageDraw
    sessions = []
    for split in ("train", "val", "test"):
        entries = read_jsonl(Path(data_dir)/f"splits/{split}.jsonl")
        sessions.append({e["session"] for e in entries})
        print(split, len(entries), {c: sum(o["label"] == c for e in entries for o in e["objects"]) for c in CLASSES})
    assert all(not sessions[i] & sessions[j] for i in range(3) for j in range(i+1, 3)), "Session leakage"
    Path(output).mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(read_jsonl(Path(data_dir)/"splits/train.jsonl")[:6]):
        with Image.open(Path(data_dir)/e["image_path"]) as im:
            im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for box in json.loads(objects_to_answer(e["objects"], "pixel", e["width"], e["height"])):
            draw.rectangle(box["bbox_2d"], outline="lime", width=2)
            draw.text(tuple(box["bbox_2d"][:2]), box["label"], fill="lime")
        im.save(Path(output)/f"ground_truth_{i+1}.png")


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["audit", "train", "infer", "run"])
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint-root", default="work/adapters")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.action == "audit":
        return audit(a.data_dir)
    config = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    name = config["name"].lower()
    if name in ("e4", "e5"):
        ap.error("Choose a concrete E4 config; E5 GRPO is optional and not implemented")
    directory = Path(a.checkpoint_root)/name
    if a.action in ("train", "run") and "lora" in config:
        train(config, a.data_dir, directory)
    if a.action in ("infer", "run"):
        suffix = "" if a.split == "test" and a.limit is None else f"_{a.split}_debug"
        pred = Path("work")/f"{name}{suffix}.jsonl"
        infer(config, a.data_dir, directory, pred, a.split, a.limit)
        if a.limit is None:
            ev = Path("report/evidence" if a.split == "test" else "work/validation")/f"{name}.json"
            score(config, a.data_dir, directory, pred, ev, a.split)


if __name__ == "__main__":
    main()
