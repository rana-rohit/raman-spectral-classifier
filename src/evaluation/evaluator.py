"""
src/evaluation/evaluator.py

Full evaluation suite. Produces the complete results matrix:
  rows = models, columns = evaluation splits (in-domain + OOD)

Also computes:
  - Transfer gaps per model per OOD split
  - McNemar's test for pairwise model comparison
  - Per-class error analysis on clinical splits
  - Saves everything to a machine-readable JSON for paper table generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_transfer_gap,
)


class ModelEvaluator:
    """
    Runs a trained model against all evaluation splits and
    produces the complete results matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        n_classes: int = 30,
        device: str = "cpu",
    ) -> None:
        self.model      = model
        self.model_name = model_name
        self.n_classes  = n_classes
        self.device     = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.results: Dict = {"model": model_name, "splits": {}}

    # ------------------------------------------------------------------ #
    # Single-split evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_split(
        self,
        loader: DataLoader,
        split_name: str,
    ) -> Dict:
        all_logits, all_targets = [], []

        for x, y in loader:
            x = x.to(self.device)
            logits = self.model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)

        all_logits  = torch.cat(all_logits,  dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Convert to numpy for processing
        y_pred = all_logits.argmax(dim=-1).cpu().numpy()
        y_true = all_targets.cpu().numpy()

        # 🔥 Apply clinical subset logic ONLY for clinical splits
        if "clinical" in split_name:
            from src.evaluation.clinical_utils import clinical_subset_eval

            y_true, y_pred = clinical_subset_eval(y_true, y_pred)

            # Debug (VERY IMPORTANT)
            print(f"[DEBUG] {split_name} mapped labels:", np.unique(y_true))

            # Convert back to tensors
            all_logits = torch.from_numpy(
                np.eye(5)[y_pred]
            ).float()
            all_targets = torch.from_numpy(y_true)

            n_classes = 5
        else:
            n_classes = self.n_classes

        # Compute metrics with correct label space
        metrics = compute_metrics(all_logits, all_targets, n_classes)

        cm, present_classes = compute_confusion_matrix(
            all_logits, all_targets, n_classes
        )

        self.results["splits"][split_name] = {
            "metrics":          metrics,
            "confusion_matrix": cm.tolist(),
            "present_classes":  present_classes,
            "n_samples":        len(all_targets),
        }

        # Store raw predictions for McNemar's test
        self.results["splits"][split_name]["predictions"] = \
            all_logits.argmax(dim=-1).numpy().tolist()
        self.results["splits"][split_name]["targets"] = \
            all_targets.numpy().tolist()

        return metrics

    # ------------------------------------------------------------------ #
    # Full evaluation across all splits
    # ------------------------------------------------------------------ #

    def evaluate_all(
        self,
        loaders: Dict,
        source_test_key: str = "test",
    ) -> Dict:
        """
        Evaluate on test split + all OOD splits.
        Computes transfer gaps automatically.
        """
        # In-domain test
        print(f"\n  [{self.model_name}] Evaluating {source_test_key}...")
        test_metrics = self.evaluate_split(loaders[source_test_key], source_test_key)
        print(f"    acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1_macro']:.4f}")

        # OOD splits
        for ood_name, ood_loader in loaders.get("ood", {}).items():
            print(f"  [{self.model_name}] Evaluating {ood_name}...")
            ood_metrics = self.evaluate_split(ood_loader, ood_name)
            gap = compute_transfer_gap(test_metrics, ood_metrics)
            self.results["splits"][ood_name]["transfer_gap"] = gap
            print(f"    acc={ood_metrics['accuracy']:.4f}  "
                  f"f1={ood_metrics['f1_macro']:.4f}  gap={gap:+.4f}")

        self.results["summary"] = self._build_summary(source_test_key)
        return self.results

    # ------------------------------------------------------------------ #
    # McNemar's test (pairwise model comparison)
    # ------------------------------------------------------------------ #

    @staticmethod
    def mcnemar_test(
        preds_a: List[int],
        preds_b: List[int],
        targets: List[int],
    ) -> Dict:
        """
        McNemar's test for two classifiers on the same test set.

        Tests whether classifier A and B differ significantly in their
        error patterns (not just accuracy).

        Returns: {statistic, p_value, significant (p < 0.05)}
        """
        import math

        preds_a  = np.array(preds_a)
        preds_b  = np.array(preds_b)
        targets  = np.array(targets)

        correct_a = preds_a == targets
        correct_b = preds_b == targets

        # Discordant counts
        n_ab = ((correct_a) & (~correct_b)).sum()   # A correct, B wrong
        n_ba = ((~correct_a) & (correct_b)).sum()   # B correct, A wrong

        n_discordant = n_ab + n_ba
        if n_discordant == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        # McNemar statistic with continuity correction
        statistic = (abs(n_ab - n_ba) - 1) ** 2 / n_discordant

        # Chi-squared(1) p-value approximation
        p_value = _chi2_sf(statistic, df=1)

        return {
            "statistic":   float(statistic),
            "p_value":     float(p_value),
            "significant": bool(p_value < 0.05),
            "n_a_wins":    int(n_ab),
            "n_b_wins":    int(n_ba),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save full results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Remove raw predictions before saving (large, not needed for tables)
        clean = json.loads(json.dumps(self.results))
        for split_data in clean.get("splits", {}).values():
            split_data.pop("predictions", None)
            split_data.pop("targets", None)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  Results saved to {path}")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _build_summary(self, test_key: str) -> Dict:
        summary = {"model": self.model_name}
        for split_name, data in self.results["splits"].items():
            m = data["metrics"]
            summary[split_name] = {
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "mcc":      m["mcc"],
            }
            if "transfer_gap" in data:
                summary[split_name]["transfer_gap"] = data["transfer_gap"]
        return summary


# ------------------------------------------------------------------ #
# Multi-model comparison
# ------------------------------------------------------------------ #

def compare_models(
    results_list: List[Dict],
    split_names: List[str],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted comparison table from multiple model results.

    Args:
        results_list: List of result dicts from ModelEvaluator.evaluate_all()
        split_names:  Column order for the table
        save_path:    If set, also saves the table as a text file

    Returns: Formatted table string
    """
    header = f"{'Model':<18}" + "".join(
        f"  {s[:10]:<12}" for s in split_names
    )
    sep = "─" * len(header)
    rows = [sep, header, sep]

    for res in results_list:
        model_name = res["model"]
        row = f"{model_name:<18}"
        for split in split_names:
            acc = res.get("summary", {}).get(split, {}).get("accuracy", float("nan"))
            row += f"  {acc:.4f}      "
        rows.append(row)

    rows.append(sep)
    table = "\n".join(rows)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(table)

    return table


# ------------------------------------------------------------------ #
# Chi-squared survival function (pure numpy, no scipy dependency)
# ------------------------------------------------------------------ #

def _chi2_sf(x: float, df: int = 1) -> float:
    """
    Chi-squared(df) survival function P(X > x).
    Uses regularised incomplete gamma function via series expansion.
    Accurate for df=1 (McNemar's test).
    """
    import math
    if x <= 0:
        return 1.0
    # For df=1: chi2_sf(x) = erfc(sqrt(x/2))
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    # General case via lower incomplete gamma
    k = df / 2
    return 1.0 - _regularised_gamma(k, x / 2)


def _regularised_gamma(a: float, x: float, n_terms: int = 100) -> float:
    """Lower regularised incomplete gamma P(a, x) via series expansion."""
    import math
    if x == 0:
        return 0.0
    term = math.exp(-x + a * math.log(x) - math.lgamma(a + 1))
    result = term
    for n in range(1, n_terms):
        term *= x / (a + n)
        result += term
        if abs(term) < 1e-12 * abs(result):
            break
    return min(result, 1.0)