"""Prepare E4 from validation, draw predictions, summarize real evidence."""
import argparse
import copy
import json
import sys
from pathlib import Path

from lab import objects_to_answer
from scripts.labcommon import read_jsonl
from scripts.score import evaluate, parse_output


def prepare_e4():
    import yaml
    candidates = []
    for name in ("e1", "e2", "e3"):
        evidence = Path(f"work/validation/{name}.json")
        if not evidence.exists():
            raise FileNotFoundError(f"Run full validation first: {evidence}")
        d = json.loads(evidence.read_text())
        if d["split"] != "val" or d["missing_predictions"]:
            raise ValueError(f"Invalid validation evidence: {evidence}")
        candidates.append((d["micro"]["f1"], name))
    _, best = max(candidates)
    base = yaml.safe_load(Path(f"configs/{best}.yaml").read_text(encoding="utf-8"))
    for rank in (4, 64):
        config = copy.deepcopy(base)
        config["name"] = f"E4_rank_{rank}"
        config["lora"].update(r=rank, alpha=2*rank)
        Path(f"configs/e4_rank_{rank}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    Path("report").mkdir(exist_ok=True)
    Path("report/e4_selection.json").write_text(json.dumps({"selected_on": "val", "base": best,
        "validation_f1": dict((name, f1) for f1, name in candidates), "variable": "rank",
        "values": [4, 64]}, indent=2), encoding="utf-8")
    print(f"E4 base={best}, selected on validation; only rank/associated alpha vary")


def figures(name, data_dir):
    from PIL import Image, ImageDraw
    entries = read_jsonl(Path(data_dir)/"splits/test.jsonl")
    preds = {e["image_id"]: e["raw"] for e in read_jsonl(f"work/{name}.jsonl")}
    coord = json.loads(Path(f"report/evidence/{name}.json").read_text())["coord"]
    ranked = []
    for e in entries:
        score = evaluate([e], preds, coord=coord, n_boot=40)
        ranked.append((score["micro"]["f1"], e))
    ranked.sort(key=lambda pair: (pair[0], pair[1]["image_id"]))
    selected = ranked[:3] + ranked[-3:]
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)
    captions = []
    for i, (f1, e) in enumerate(selected, 1):
        with Image.open(Path(data_dir)/e["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for b in json.loads(objects_to_answer(e["objects"], "pixel", e["width"], e["height"])):
            draw.rectangle(b["bbox_2d"], outline="lime", width=3)
            draw.text(tuple(b["bbox_2d"][:2]), "GT "+b["label"], fill="lime")
        parsed = parse_output(preds[e["image_id"]])
        for b in parsed or []:
            box = list(b["bbox"])
            if coord == "norm1000":
                box = [v*s/1000 for v, s in zip(box, (e["width"], e["height"])*2)]
            draw.rectangle(box, outline="red", width=2)
            draw.text(tuple(box[:2]), "PRED "+b["label"], fill="red")
        filename = f"{name}_{i}.png"
        image.save(out/filename)
        captions.append(f"![{e['image_id']}: F1={f1:.3f}; GT xanh, dự đoán đỏ](figures/{filename})")
    (out/f"{name}_captions.md").write_text("\n\n".join(captions), encoding="utf-8")


def summary(data_dir):
    evidence = sorted(Path("report/evidence").glob("e*.json"))
    if not evidence:
        raise FileNotFoundError("No scored experiments; cannot generate measured results")
    rows = []
    for p in evidence:
        d = json.loads(p.read_text())
        if d["split"] != "test" or d["missing_predictions"]:
            raise ValueError(f"Invalid test evidence {p}")
        m = d["meta"]
        vals = [d["name"], str(m.get("trainable_params", 0)), str(m.get("peak_vram_gb", "—")),
                str(m.get("train_minutes", "—")), f"{d['micro']['f1']:.4f}",
                f"[{d['f1_ci95'][0]:.4f}, {d['f1_ci95'][1]:.4f}]", f"{d['fp_per_image']:.4f}"]
        vals += [f"{d['per_class'][c]['f1']:.4f}" for c in ("pothole", "crack", "manhole")]
        rows.append("| "+" | ".join(vals)+" |")
    text = "| Mã | Tham số train | VRAM GB | Phút | F1 | CI95 | FP/ảnh | pothole | crack | manhole |\n"
    text += "|"+" --- |"*10+"\n"+"\n".join(rows)
    Path("report/RESULTS.md").write_text(text, encoding="utf-8")
    # Empirically check the coordinate mismatch without creating another test submission.
    entries = read_jsonl(Path(data_dir)/"splits/test.jsonl")
    raw = {p["image_id"]: p["raw"] for p in read_jsonl("work/e0.jsonl")}
    compare = {coord: evaluate(entries, raw, coord=coord)["micro"]["f1"] for coord in ("norm1000", "pixel")}
    Path("report/coordinate_check.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print("Wrote report/RESULTS.md and report/coordinate_check.json from actual predictions")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["e4", "figures", "summary"])
    ap.add_argument("--name", default="e1")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    if a.action == "e4":
        prepare_e4()
    elif a.action == "figures":
        figures(a.name, a.data_dir)
    else:
        summary(a.data_dir)


if __name__ == "__main__":
    main()
