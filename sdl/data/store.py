"""Data twin, part 1: the trace store.

Every run, simulated or real, accepted or rejected, is written in the same
format: one folder per run with a JSON record and the raw frames, plus one line
in index.jsonl. The `backend` field keeps simulated and real data apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


class TraceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def write(self, record: dict, frames: list | None = None) -> Path:
        run_dir = self.root / record["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        record = dict(record)
        record["frames"] = []
        for i, frame in enumerate(frames or []):
            name = f"frame_{i}.png"
            PILImage.fromarray(np.asarray(frame, dtype=np.uint8)).save(run_dir / name)
            record["frames"].append(name)
        with open(run_dir / "record.json", "w") as fh:
            json.dump(record, fh, indent=2)
        with open(self.index, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        return run_dir

    def records(self, backend: str | None = None, accepted_only: bool = True) -> list:
        if not self.index.exists():
            return []
        out = []
        with open(self.index) as fh:
            for line in fh:
                r = json.loads(line)
                if backend and r.get("backend", {}).get("backend") != backend:
                    continue
                if accepted_only and not r.get("accepted"):
                    continue
                out.append(r)
        return out

    def frame(self, run_id: str, i: int = -1) -> np.ndarray:
        with open(self.root / run_id / "record.json") as fh:
            names = json.load(fh)["frames"]
        return np.asarray(PILImage.open(self.root / run_id / names[i]).convert("RGB"))
