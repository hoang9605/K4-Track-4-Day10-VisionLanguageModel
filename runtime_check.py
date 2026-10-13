"""Check actual HF model tree and multimodal loss masking without model weights."""
import re
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from lab import MODEL_ID, LLM_TARGET, VIT_TARGET, encode_example, is_projector


def main():
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    llm = [n for n, _ in model.named_modules() if re.fullmatch(LLM_TARGET, n)]
    vit = [n for n, _ in model.named_modules() if re.fullmatch(VIT_TARGET, n)]
    assert len(llm) == 252, len(llm)
    assert len(vit) == 96, len(vit)
    sizes = {
        "total": sum(p.numel() for p in model.parameters()),
        "visual_including_projector": sum(p.numel() for n, p in model.named_parameters() if "visual." in n),
        "projector": sum(p.numel() for n, p in model.named_parameters() if is_projector(n)),
        "language_model": sum(p.numel() for n, p in model.named_parameters() if "language_model." in n),
    }
    print("Actual architecture counts:", sizes, "LoRA targets:", len(llm), len(vit))
    import tempfile
    from pathlib import Path
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    with tempfile.TemporaryDirectory() as tmp:
        Image.new("RGB", (640, 360)).save(Path(tmp)/"test.jpg")
        e = dict(image_path="test.jpg", width=640, height=360,
                 objects=[dict(label="pothole", yolo=[.5, .5, .2, .2])])
        full = encode_example(processor, e, tmp, "norm1000")
        supervised = processor.tokenizer.decode(full["labels"][full["labels"] != -100])
        assert supervised == '[{"label":"pothole","bbox_2d":[400,400,600,600]}]<|im_end|>\n', repr(supervised)
        n_image = int((full["input_ids"] == config.image_token_id).sum())
        assert n_image == 220, n_image
        assert torch.all(full["labels"][full["input_ids"] == config.image_token_id] == -100)
        print("Vision tokens:", n_image, "Answer-only labels:", repr(supervised))
    print("Model tree, 220 image tokens, prefix and answer loss mask: PASS")


if __name__ == "__main__":
    main()
