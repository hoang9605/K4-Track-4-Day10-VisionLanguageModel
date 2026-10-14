"""Run required experiments sequentially; persist progress and per-stage logs."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-root", default="work/adapters")
    ap.add_argument("--start", default="audit", help="Resume at a named stage after resolving a failure")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    log_dir = root / "work/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stages = [("audit", ["lab.py", "audit"]), ("processor", ["runtime_check.py"])]
    import yaml
    for name in ("e1", "e2", "e3"):
        config = yaml.safe_load((root/f"configs/{name}.yaml").read_text(encoding="utf-8"))
        config.update(name=f"SMOKE_{name.upper()}", max_steps=1)
        path = root / f"work/smoke_{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        stages.append((f"smoke_{name}", ["lab.py", "train", "--config", str(path)]))
    stages.append(("e0", ["lab.py", "run", "--config", "configs/e0.yaml"]))
    for name in ("e1", "e2", "e3"):
        stages.extend([
            (name+"_train", ["lab.py", "train", "--config", f"configs/{name}.yaml"]),
            (name+"_val", ["lab.py", "infer", "--config", f"configs/{name}.yaml", "--split", "val"]),
            (name+"_test", ["lab.py", "infer", "--config", f"configs/{name}.yaml"]),
        ])
    stages.extend([
        ("e4_select", ["finish_lab.py", "e4"]),
        ("e4_rank_4", ["lab.py", "run", "--config", "configs/e4_rank_4.yaml"]),
        ("e4_rank_64", ["lab.py", "run", "--config", "configs/e4_rank_64.yaml"]),
        ("summary", ["finish_lab.py", "summary"]),
        ("figures", ["finish_lab.py", "figures"]),
    ])
    names = [name for name, _ in stages]
    if a.start not in names:
        ap.error(f"--start must be one of {names}")
    completed = []
    status_path = root / "work/progress.json"

    def status(stage, state, **extra):
        data = dict(stage=stage, state=state, completed=completed,
                    updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(status_path)
        print(json.dumps(data), flush=True)

    for name, args in stages[names.index(a.start):]:
        if args[0] == "lab.py" and args[1] != "audit":
            args += ["--checkpoint-root", a.checkpoint_root]
        if name == "figures":
            best = json.loads((root/"report/e4_selection.json").read_text())["base"]
            args += ["--name", best]
        logfile = log_dir / f"{name}.log"
        with logfile.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen([sys.executable, "-u", *args], stdout=stream,
                                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            status(name, "running", pid=child.pid, log=str(logfile.relative_to(root)))
            code = child.wait()
        if code:
            status(name, "failed", returncode=code, log=str(logfile.relative_to(root)))
            sys.exit(code)
        completed.append(name)
    status("complete", "complete")


if __name__ == "__main__":
    main()
