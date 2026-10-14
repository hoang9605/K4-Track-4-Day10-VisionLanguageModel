"""CPU checks for student code; original tests/ remains untouched."""
import json
import unittest


class SolutionChecks(unittest.TestCase):
    def test_visual_subtree_is_excluded_from_quantization(self):
        from lab import UNQUANTIZED_MODULES
        from transformers.quantizers.quantizers_utils import should_convert_module
        for name in ("model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
                     "model.visual.deepstack_merger_list.0.linear_fc1", "lm_head"):
            self.assertFalse(should_convert_module(name, UNQUANTIZED_MODULES), name)
        self.assertTrue(should_convert_module("model.language_model.layers.0.self_attn.q_proj",
                                            UNQUANTIZED_MODULES))

    def test_coordinates_and_order(self):
        from lab import objects_to_answer
        objects = [{"label": "crack", "yolo": [.5, .5, .2, .2]},
                   {"label": "pothole", "yolo": [.05, .05, .2, .2]}]
        boxes = json.loads(objects_to_answer(objects))
        self.assertEqual(boxes[0]["bbox_2d"], [0, 0, 150, 150])
        self.assertEqual(boxes[1]["bbox_2d"], [400, 400, 600, 600])
        self.assertEqual(json.loads(objects_to_answer(objects, "pixel", 640, 360))[1]["bbox_2d"],
                         [256, 144, 384, 216])
        self.assertEqual(objects_to_answer([]), "[]")

    def test_reward(self):
        from lab import reward
        gt = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]
        good = {"label": "pothole", "bbox_2d": [100, 100, 200, 200]}
        self.assertEqual(reward(json.dumps([good]), gt), 1.)
        self.assertEqual(reward("broken", gt), 0.)
        self.assertEqual(reward("[]", []), 1.)
        self.assertEqual(reward("[]", gt), .2)
        self.assertEqual(reward(json.dumps([{**good, "label": "crack"}]), gt), .2)
        self.assertAlmostEqual(reward(json.dumps([good, good]), gt), .6)

    def test_collation(self):
        import torch
        from lab import collate
        examples = []
        for n in (2, 4):
            examples.append({"input_ids": torch.arange(n).unsqueeze(0),
                             "attention_mask": torch.ones(1, n, dtype=torch.long),
                             "labels": torch.tensor([[-100] + list(range(1, n))]),
                             "pixel_values": torch.ones(3, 8),
                             "image_grid_thw": torch.tensor([[1, 2, 2]])})
        batch = collate(examples, 99)
        self.assertEqual(batch["input_ids"].tolist()[0], [0, 1, 99, 99])
        self.assertEqual(batch["labels"].tolist()[0], [-100, 1, -100, -100])
        self.assertEqual(tuple(batch["pixel_values"].shape), (6, 8))
        self.assertEqual(tuple(batch["image_grid_thw"].shape), (2, 3))

    def test_e4_changes_only_rank_and_alpha_from_validation_winner(self):
        import copy
        import os
        import tempfile
        from pathlib import Path
        import yaml
        from finish_lab import prepare_e4
        original_dir = Path.cwd()
        configs = {name: yaml.safe_load((original_dir/f"configs/{name}.yaml").read_text(encoding="utf-8"))
                   for name in ("e1", "e2", "e3")}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("work/validation").mkdir(parents=True)
                Path("configs").mkdir()
                for name, f1 in (("e1", .1), ("e2", .8), ("e3", .3)):
                    Path(f"configs/{name}.yaml").write_text(yaml.safe_dump(configs[name]), encoding="utf-8")
                    Path(f"work/validation/{name}.json").write_text(json.dumps(
                        dict(split="val", missing_predictions=0, micro=dict(f1=f1))), encoding="utf-8")
                prepare_e4()
                for rank in (4, 64):
                    generated = yaml.safe_load(Path(f"configs/e4_rank_{rank}.yaml").read_text())
                    expected = copy.deepcopy(configs["e2"])
                    expected["name"] = f"E4_rank_{rank}"
                    expected["lora"].update(r=rank, alpha=2*rank)
                    self.assertEqual(generated, expected)
            finally:
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
