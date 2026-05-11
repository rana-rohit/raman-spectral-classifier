"""
src/evaluation/evaluator.py

Evaluation suite for:

1. Reference-domain evaluation
2. Clinical OOD evaluation
3. Compact transfer-space metrics

Supports semantic restoration between:

- compact transfer labels
- sparse clinical labels
- clinical treatment semantics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_transfer_gap,
    confidence_vote_predictions,
)

from src.evaluation.visualization import (
    save_confusion_matrix_figure,
)

from metadata.clinical import (
    CLINICAL_LABELS,
    CLINICAL_LABEL_INVERSE_REMAP
)

class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        n_classes: int = 30,
        device: str = "cpu",
        cfg: dict | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.n_classes = n_classes
        self.cfg = cfg or {}
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        clinical_sparse_ids = (
            self.cfg.get("task", {}).get(
                "clinical_sparse_global_ids",
                []
            )
        )

        stage = self.cfg.get("task", {}).get(
            "stage",
            None,
        )
        if stage is None:
            raise ValueError(
                "Missing task.stage in evaluator config"
            )
        if stage == "transfer_5class":
            assert self.n_classes == len(clinical_sparse_ids), (

                "Evaluator n_classes must match number of "
                "clinical sparse IDs, "

                f"got {self.n_classes} vs "
                f"{len(clinical_sparse_ids)}"
            )

        aux_cfg = self.cfg.get("multitask", {}).get("auxiliary_clinical_head", {})
        self.aux_enabled = aux_cfg.get("enabled", False)
        self.aux_blend = aux_cfg.get("clinical_blend", 0.5)

        self.results: Dict = {"model": model_name, "splits": {}}

        self.output_dir = Path(
            self.cfg.get(
                "experiment",
                {}
            ).get(
                "save_dir",
                "results"
            )
        )
        
        print("EVALUATOR OUTPUT DIR:", self.output_dir)

    @torch.no_grad()
    def evaluate_split(
        self,
        loader: DataLoader,
        split_name: str,
    ) -> Dict:
        all_main_logits, all_targets = [], []

        for batch in loader:
            x, y = self._parse_batch(batch)
            outputs = self._normalize_outputs(self.model(x))
            all_main_logits.append(outputs["main_logits"].cpu())
            all_targets.append(y.cpu())

        main_logits = torch.cat(all_main_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        self._assert_logits_and_targets(main_logits, targets, split_name)
        eval_logits, eval_targets = main_logits, targets
        n_classes = self.n_classes

        metrics = compute_metrics(eval_logits, eval_targets, n_classes)

        # --------------------------------------------------------
        # Patient-level majority voting
        #
        # Clinical datasets contain multiple spectra per
        # isolate/patient.
        #
        # Aggregate spectrum predictions into clinically
        # realistic patient-level predictions.
        # --------------------------------------------------------

        group_metrics = {}

        if split_name == "test":
            spectra_per_group = 100

        elif split_name == "2018clinical":
            spectra_per_group = 400

        elif split_name == "2019clinical":
            spectra_per_group = 100

        else:
            spectra_per_group = None

        if spectra_per_group is not None:

            preds_np = eval_logits.argmax(dim=-1).cpu().numpy()
            targets_np = eval_targets.cpu().numpy()

            assert len(preds_np) % spectra_per_group == 0, (
                f"{split_name} size must be divisible by "
                f"{spectra_per_group}"
            )

            group_preds, group_targets = confidence_vote_predictions(
                logits=torch.softmax(eval_logits, dim=-1).cpu().numpy(),
                targets=targets_np,
                sample_ids=None,
                spectra_per_group=spectra_per_group,
            )

            group_results = {
                "accuracy": accuracy_score(group_targets, group_preds),
                "f1_macro": f1_score(
                    group_targets,
                    group_preds,
                    average="macro",
                ),
                "mcc": matthews_corrcoef(
                    group_targets,
                    group_preds,
                ),
            }

            group_metrics = {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in group_results.items()
            }
            group_metrics["predictions"] = group_preds.tolist()
            group_metrics["targets"] = group_targets.tolist()

            if split_name == "test":
                group_name = "isolate"
            else:
                group_name = "patient"

            print(
                f"    {group_name}_acc="
                f"{group_results['accuracy']:.4f}  "
                f"{group_name}_f1="
                f"{group_results['f1_macro']:.4f}"
            )

        cm, present_classes = compute_confusion_matrix(eval_logits, eval_targets, n_classes)
        inverse_class_map = getattr(loader.dataset, "inverse_class_map", {})
        
        # Restore original sparse clinical IDs from
        # compact transfer-space labels.
        original_present_classes = [
            inverse_class_map.get(int(cls), int(cls))
            for cls in present_classes
        ]
        
        # --------------------------------------------------------
        # Semantic restoration layer
        # Convert compact transfer labels:
        #   [0,1,2,3,4]
        #
        # back into sparse clinical IDs:
        #   [0,2,3,5,6]
        #   
        # and attach human-readable
        # treatment/species semantics.
        # --------------------------------------------------------
        stage = self.cfg.get(
            "task",
            {},
        ).get(
            "stage",
            None,
        )
        if stage is None:
            raise ValueError(
                "Missing task.stage in evaluator config"
            )
        clinical_semantics = {}

        if stage == "transfer_5class":

            for compact_idx in present_classes:

                compact_idx = int(compact_idx)

                if compact_idx in CLINICAL_LABEL_INVERSE_REMAP:

                    original_label = (
                        CLINICAL_LABEL_INVERSE_REMAP[
                            compact_idx
                        ]
                    )

                    clinical_semantics[
                        str(compact_idx)
                    ] = {
                        "original_label": original_label,
                        "species": CLINICAL_LABELS[
                            original_label
                        ]["clinical_species"],
                        "treatment": CLINICAL_LABELS[
                            original_label
                        ]["global_treatment"],
                    }
        
        self.results["splits"][split_name] = {
            "metrics": {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in metrics.items()
            },
            "semantic_space": self.cfg.get(
                "model",
                {},
            ).get(
                "semantic_space",
                "unknown",
            ),
            "n_classes": int(self.n_classes),
            "confusion_matrix": cm.tolist(),
            "present_classes": [int(x) for x in present_classes],
            "original_present_classes": [int(x) for x in original_present_classes],
            "clinical_semantics": clinical_semantics,
            "n_samples": int(len(eval_targets)),
            "predictions": eval_logits.argmax(dim=-1).numpy().tolist(),
            "targets": eval_targets.numpy().tolist(),
            "group_metrics": group_metrics,
            "compact_to_sparse_label_map": (
                {
                    str(compact_idx): int(original_idx)
                    for compact_idx, original_idx
                    in inverse_class_map.items()
                }
                if stage == "transfer_5class"
                else {}
            ),
        }
        # Automatic confusion matrix visualization generation

        figure_dir = (
            self.output_dir
            / "confusion_matrices"
            / split_name
        )

        # Spectrum-level confusion

        TREATMENT_LABELS = {
            0: "Vancomycin",
            1: "Ceftriaxone",
            2: "Penicillin",
            3: "Daptomycin",
            4: "Meropenem",
            5: "Ciprofloxacin",
            6: "TZP",
            7: "Caspofungin",
        }

        class_labels = [
            TREATMENT_LABELS.get(int(x), str(x))
            for x in present_classes
        ]

        save_confusion_matrix_figure(
            targets=eval_targets.numpy(),
            predictions=eval_logits.argmax(dim=-1).numpy(),
            class_labels=class_labels,
            save_path=(
                figure_dir
                / "spectrum_confusion.png"
            ),
            title=(
                f"{split_name} "
                "Spectrum-Level Confusion"
            ),
            normalize=False,
        )

        save_confusion_matrix_figure(
            targets=eval_targets.numpy(),
            predictions=eval_logits.argmax(dim=-1).numpy(),
            class_labels=class_labels,
            save_path=(
                figure_dir
                / "spectrum_confusion_normalized.png"
            ),
            title=(
                f"{split_name} "
                "Spectrum-Level Confusion "
                "(Normalized)"
            ),
            normalize=True,
        )

        # Group-level confusion

        if group_metrics:

            save_confusion_matrix_figure(
                targets=group_metrics["targets"],
                predictions=group_metrics["predictions"],
                class_labels=class_labels,
                save_path=(
                    figure_dir
                    / "group_confusion.png"
                ),
                title=(
                    f"{split_name} "
                    "Group-Level Confusion"
                ),
                normalize=False,
            )

            save_confusion_matrix_figure(
                targets=group_metrics["targets"],
                predictions=group_metrics["predictions"],
                class_labels=class_labels,
                save_path=(
                    figure_dir
                    / "group_confusion_normalized.png"
                ),
                title=(
                    f"{split_name} "
                    "Group-Level Confusion "
                    "(Normalized)"
                ),
                normalize=True,
            )
        return metrics
    
    # --------------------------------------------------------
    # Evaluation protocol
    #
    # test:
    #   reference-domain transfer evaluation
    #
    # ood:
    #   clinical-domain transfer evaluation
    #
    # During transfer-learning experiments:
    #
    # BOTH evaluations operate in:
    # compact transfer-space
    #
    # NOT isolate-space.
    # --------------------------------------------------------
    def evaluate_all(
        self,
        loaders: Dict,
        source_test_key: str = "test",
    ) -> Dict:
        print(f"\n  [{self.model_name}] Evaluating {source_test_key}...")
        test_metrics = self.evaluate_split(loaders[source_test_key], source_test_key)
        print(f"    acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1_macro']:.4f}")

        for ood_name, ood_loader in loaders.get("ood", {}).items():
            print(f"  [{self.model_name}] Evaluating {ood_name}...")
            ood_metrics = self.evaluate_split(ood_loader, ood_name)
            gap = compute_transfer_gap(test_metrics, ood_metrics)
            self.results["splits"][ood_name]["transfer_gap"] = gap
            print(
                f"    acc={ood_metrics['accuracy']:.4f}  "
                f"f1={ood_metrics['f1_macro']:.4f}  gap={gap:+.4f}"
            )

        self.results["summary"] = self._build_summary(source_test_key)
        return self.results

    @staticmethod
    def mcnemar_test(
        preds_a: List[int],
        preds_b: List[int],
        targets: List[int],
    ) -> Dict:
        import math

        preds_a = np.array(preds_a)
        preds_b = np.array(preds_b)
        targets = np.array(targets)

        correct_a = preds_a == targets
        correct_b = preds_b == targets
        n_ab = ((correct_a) & (~correct_b)).sum()
        n_ba = ((~correct_a) & (correct_b)).sum()
        n_discordant = n_ab + n_ba
        if n_discordant == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        statistic = (abs(n_ab - n_ba) - 1) ** 2 / n_discordant
        p_value = _chi2_sf(statistic, df=1)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "n_a_wins": int(n_ab),
            "n_b_wins": int(n_ba),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        clean = json.loads(json.dumps(self.results))
        for split_data in clean.get("splits", {}).values():
            split_data.pop("predictions", None)
            split_data.pop("targets", None)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  Results saved to {path}")

    def _build_summary(self, test_key: str) -> Dict:
        summary = {"model": self.model_name}
        for split_name, data in self.results["splits"].items():
            metrics = data["metrics"]
            summary[split_name] = {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "mcc": metrics["mcc"],
            }
            if "transfer_gap" in data:
                summary[split_name]["transfer_gap"] = data["transfer_gap"]
        return summary

    def _normalize_outputs(self, outputs) -> dict[str, torch.Tensor | None]:
        if torch.is_tensor(outputs):
            return {
                "main_logits": outputs,
                "aux_logits": None,
                "features": None,
            }
        if isinstance(outputs, dict):
            return {
                "main_logits": outputs["main_logits"],
                "aux_logits": outputs.get("aux_logits"),
                "features": outputs.get("features"),
            }
        raise TypeError(f"Unsupported model output type: {type(outputs)!r}")

    def _parse_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["x1"].to(self.device), batch["y"].to(self.device)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0].to(self.device), batch[1].to(self.device)
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def _assert_logits_and_targets(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        split_name: str,
    ) -> None:

        assert logits.shape[-1] == self.n_classes, (
            f"{split_name} logits must have "
            f"{self.n_classes} classes, "
            f"got shape {tuple(logits.shape)}"
        )

        if targets.numel() == 0:
            raise AssertionError(
                f"{split_name} has no targets"
            )

        assert int(targets.min()) >= 0, (
            f"{split_name} targets must be non-negative, "
            f"got {int(targets.min())}"
        )

        assert int(targets.max()) < self.n_classes, (
            f"{split_name} targets must be < "
            f"{self.n_classes}, "
            f"got {int(targets.max())}"
        )

        unique_targets = sorted(
            torch.unique(targets).cpu().tolist()
        )

        # --------------------------------------------------------
        # Compact transfer-space integrity check
        #
        # Transfer tasks MUST use compact labels:
        # [0,1,2,3,4]
        #
        # Sparse clinical labels:
        # [0,2,3,5,6]
        #
        # should never appear at evaluator stage.
        # --------------------------------------------------------

        stage = self.cfg.get(
            "task",
            {},
        ).get(
            "stage",
            "transfer_5class",
        )

        if stage == "transfer_5class":

            expected = [0, 1, 2, 3, 4]

            unexpected = set(unique_targets) - set(expected)

            if unexpected:
                raise AssertionError(

                    f"{split_name} contains unexpected "
                    f"compact transfer labels: "
                    f"{sorted(unexpected)}"
                )


def compare_models(
    results_list: List[Dict],
    split_names: List[str],
    save_path: Optional[str] = None,
) -> str:
    header = f"{'Model':<18}" + "".join(f"  {split[:10]:<12}" for split in split_names)
    sep = "-" * len(header)
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


def _chi2_sf(x: float, df: int = 1) -> float:
    import math

    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    k = df / 2
    return 1.0 - _regularised_gamma(k, x / 2)


def _regularised_gamma(a: float, x: float, n_terms: int = 100) -> float:
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
