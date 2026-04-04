"""
src/utils/logging.py

Structured experiment logger. Every metric at every epoch is persisted
to a JSON file alongside console output. This is the record that
generates paper tables and learning curve plots.
"""

import json
import time
from pathlib import Path
from typing import Dict


class ExperimentLogger:
    """
    Logs metrics to:
      - Console (formatted table)
      - <exp_dir>/metrics.json  (full history, machine-readable)
      - <exp_dir>/summary.json  (best metrics per split, for tables)
    """

    def __init__(self, exp_dir: str, model_name: str, config: Dict) -> None:
        self.exp_dir    = Path(exp_dir)
        self.model_name = model_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self._history: list = []
        self._best: Dict    = {}
        self._start_time    = time.time()

        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"  Experiment: {model_name}")
        print(f"  Directory:  {exp_dir}")
        print(f"{'='*60}")

    def log(self, epoch: int, split: str, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch/split."""
        record = {"epoch": epoch, "split": split, "time": time.time() - self._start_time}
        record.update(metrics)
        self._history.append(record)

        key = f"{split}_acc"
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            if key not in self._best or acc > self._best[key]:
                self._best[key]                  = acc
                self._best[f"{split}_epoch"]     = epoch
                self._best[f"{split}_metrics"]   = metrics

        self._flush()
        self._print_row(epoch, split, metrics)

    def log_final(self, split: str, metrics: Dict[str, float]) -> None:
        """Log final test-set results (called once at end of training)."""
        print(f"\n{'─'*60}")
        print(f"  FINAL RESULTS — {split.upper()}")
        for k, v in metrics.items():
            print(f"    {k:<22} {v:.4f}")
        print(f"{'─'*60}\n")
        self._best[f"final_{split}"] = metrics
        self._flush()

    def _print_row(self, epoch: int, split: str, metrics: Dict) -> None:
        acc  = metrics.get("accuracy", float("nan"))
        loss = metrics.get("loss",     float("nan"))
        f1   = metrics.get("f1_macro", float("nan"))
        print(f"  Ep {epoch:>3d} | {split:<12} | "
              f"loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}")

    def _flush(self) -> None:
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(self._history, f, indent=2, default=str)
        with open(self.exp_dir / "summary.json", "w") as f:
            json.dump(self._best, f, indent=2, default=str)

    @property
    def history(self) -> list:
        return self._history

    @property
    def best(self) -> Dict:
        return self._best