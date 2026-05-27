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
from dataclasses import dataclass
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
from src.utils.split_modes import resolve_split_mode

from metadata.clinical import (
    CLINICAL_LABELS,
    CLINICAL_LABEL_INVERSE_REMAP
)

@dataclass
class SplitEvaluationArtifact:
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    features: Optional[np.ndarray]
    grouped_predictions: Optional[np.ndarray]
    grouped_targets: Optional[np.ndarray]
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    present_classes: List[int]


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

        self.results: Dict = {
            "model": model_name,
            "split_mode": resolve_split_mode(self.cfg),
            "splits": {},
        }
        self.artifacts: Dict[str, SplitEvaluationArtifact] = {}

        self.output_dir = Path(
            self.cfg.get(
                "experiment",
                {}
            ).get(
                "save_dir",
                "results"
            )
        )

    @torch.no_grad()
    def _forward_pass(
        self,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        all_main_logits, all_targets = [], []
        features_list = []
        has_features = True

        for batch in loader:
            x, y = self._parse_batch(batch)
            outputs = self._normalize_outputs(self.model(x))
            all_main_logits.append(outputs["main_logits"].cpu())
            all_targets.append(y.cpu())
            feats = outputs.get("features")
            if feats is None:
                has_features = False
            else:
                features_list.append(feats.detach().cpu())

        main_logits = torch.cat(all_main_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        features = None
        if has_features and features_list:
            features = torch.cat(features_list, dim=0).numpy()

        return main_logits, targets, features

    @torch.no_grad()
    def evaluate_split(
        self,
        loader: DataLoader,
        split_name: str,
    ) -> Dict:
        main_logits, targets, features = self._forward_pass(loader)
        self._assert_logits_and_targets(main_logits, targets, split_name)
        eval_logits, eval_targets = main_logits, targets
        n_classes = self.n_classes

        metrics = compute_metrics(eval_logits, eval_targets, n_classes)
        probabilities = torch.softmax(eval_logits, dim=-1).cpu().numpy()
        predictions = eval_logits.argmax(dim=-1).cpu().numpy()

        # --------------------------------------------------------
        # Patient-level majority voting
        #
        # Clinical datasets contain multiple spectra per
        # isolate/patient.
        #
        # Aggregate spectrum predictions into clinically
        # realistic patient-level predictions.
        # --------------------------------------------------------

        from src.utils.logging import SPECTRA_PER_GROUP
        group_metrics = {}
        grouped_cfg = self.cfg.get("evaluation", {}).get("grouped", {})
        grouped_enabled = grouped_cfg.get("enabled", True)
        spectra_per_group_map = grouped_cfg.get(
            "spectra_per_group",
            SPECTRA_PER_GROUP,
        )
        spectra_per_group = spectra_per_group_map.get(split_name) if grouped_enabled else None

        group_preds = None
        group_targets = None

        if spectra_per_group is not None:

            preds_np = predictions
            targets_np = eval_targets.cpu().numpy()

            assert len(preds_np) % spectra_per_group == 0, (
                f"{split_name} size must be divisible by "
                f"{spectra_per_group}"
            )

            group_preds, group_targets = confidence_vote_predictions(
                logits=probabilities,
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
            group_metrics["n_groups"] = int(len(group_targets))
            group_metrics["spectra_per_group"] = int(spectra_per_group)
            group_metrics["predictions"] = group_preds.tolist()
            group_metrics["targets"] = group_targets.tolist()

            if split_name == "test":
                group_name = "isolate"
            else:
                group_name = "patient"

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
        
        include_predictions = bool(
            self.cfg.get("evaluation", {}).get("include_predictions", False)
        )

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

        if include_predictions:
            self.results["splits"][split_name]["predictions"] = predictions.tolist()
            self.results["splits"][split_name]["targets"] = eval_targets.cpu().numpy().tolist()
        # Automatic confusion matrix visualization generation

        figure_dir = (
            self.output_dir
            / "confusion_matrices"
            / split_name
        )

        # Spectrum-level confusion

        # --------------------------------------------------------
        # Stage-aware class label restoration
        # --------------------------------------------------------

        ISOLATE_LABELS = {
            0: "C. albicans",
            1: "C. glabrata",
            2: "K. aerogenes",
            3: "E. coli 1",
            4: "E. coli 2",
            5: "E. faecium",
            6: "E. faecalis 1",
            7: "E. faecalis 2",
            8: "E. cloacae",
            9: "K. pneumoniae 1",
            10: "K. pneumoniae 2",
            11: "P. mirabilis",
            12: "P. aeruginosa 1",
            13: "P. aeruginosa 2",
            14: "MSSA 1",
            15: "MSSA 3",
            16: "MRSA 1",
            17: "MRSA 2",
            18: "MSSA 2",
            19: "S. enterica",
            20: "S. epidermidis",
            21: "S. lugdunensis",
            22: "S. marcescens",
            23: "S. pneumoniae 2",
            24: "S. pneumoniae 1",
            25: "S. sanguinis",
            26: "Group A Strep.",
            27: "Group B Strep.",
            28: "Group C Strep.",
            29: "Group G Strep.",
        }

        TREATMENT_LABELS = {
            0: "Meropenem",
            1: "Ciprofloxacin",
            2: "TZP",
            3: "Vancomycin",
            4: "Ceftriaxone",
            5: "Penicillin",
            6: "Daptomycin",
            7: "Caspofungin",
        }

        if self.n_classes == 30:
            label_map = ISOLATE_LABELS
        else:
            label_map = TREATMENT_LABELS

        class_labels = [
            label_map.get(int(x), str(x))
            for x in present_classes
        ]

        if self.cfg.get("evaluation", {}).get("save_confusion_matrices", True):
            save_confusion_matrix_figure(
                targets=eval_targets.numpy(),
                predictions=predictions,
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
                        / "group_confusion_normalized.png"
                    ),
                    title=(
                        f"{split_name} "
                        "Group-Level Confusion "
                        "(Normalized)"
                    ),
                    normalize=True,
                )
        self.artifacts[split_name] = SplitEvaluationArtifact(
            logits=eval_logits.cpu().numpy(),
            probabilities=probabilities,
            predictions=predictions,
            targets=eval_targets.cpu().numpy(),
            features=features,
            grouped_predictions=group_preds,
            grouped_targets=group_targets,
            metrics=metrics,
            confusion_matrix=cm,
            present_classes=[int(x) for x in present_classes],
        )

        return metrics

    @torch.no_grad()
    def collect_artifact(
        self,
        loader: DataLoader,
        split_name: str,
    ) -> SplitEvaluationArtifact:
        main_logits, targets, features = self._forward_pass(loader)
        self._assert_logits_and_targets(main_logits, targets, split_name)
        probabilities = torch.softmax(main_logits, dim=-1).cpu().numpy()
        predictions = main_logits.argmax(dim=-1).cpu().numpy()
        cm, present_classes = compute_confusion_matrix(main_logits, targets, self.n_classes)

        artifact = SplitEvaluationArtifact(
            logits=main_logits.cpu().numpy(),
            probabilities=probabilities,
            predictions=predictions,
            targets=targets.cpu().numpy(),
            features=features,
            grouped_predictions=None,
            grouped_targets=None,
            metrics={},
            confusion_matrix=cm,
            present_classes=[int(x) for x in present_classes],
        )
        self.artifacts[split_name] = artifact
        return artifact
    
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
        print(f"  [{self.model_name}] Evaluating split: {source_test_key}...")
        test_metrics = self.evaluate_split(loaders[source_test_key], source_test_key)

        for ood_name, ood_loader in loaders.get("ood", {}).items():
            print(f"  [{self.model_name}] Evaluating split: {ood_name}...")
            ood_metrics = self.evaluate_split(ood_loader, ood_name)
            gap = compute_transfer_gap(test_metrics, ood_metrics)
            self.results["splits"][ood_name]["transfer_gap"] = gap

        self.results["summary"] = self._build_summary(source_test_key)

        # Call the centralized print_evaluation_summary for highly structured reporting
        from src.utils.logging import print_evaluation_summary
        stage = self.cfg.get("task", {}).get("stage", "transfer_5class")
        label_space = self.cfg.get("task", {}).get("label_space", "")
        clinical_sparse_ids = self.cfg.get("task", {}).get("clinical_sparse_global_ids", [])
        
        output_files = {}
        for sname, sdata in self.results["splits"].items():
            cf_dir = self.output_dir / "confusion_matrices" / sname
            output_files[f"Confusion Matrix ({sname})"] = str(cf_dir / "spectrum_confusion_normalized.png")
            if sdata.get("group_metrics"):
                output_files[f"Group Confusion Matrix ({sname})"] = str(cf_dir / "group_confusion_normalized.png")
                
        checkpoint_path = None
        save_dir = self.cfg.get("experiment", {}).get("save_dir") or str(self.output_dir)
        try:
            from src.utils.checkpoint import resolve_best_checkpoint_path
            checkpoint_path = resolve_best_checkpoint_path(save_dir)
        except Exception:
            pass
            
        print_evaluation_summary(
            stage=stage,
            model_name=self.model_name,
            model_cfg=self.cfg.get("model", {}),
            label_space=label_space,
            clinical_sparse_ids=clinical_sparse_ids,
            checkpoint_path=checkpoint_path,
            split_metrics=self.results["splits"],
            output_files=output_files
        )

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
        export = {
            "model": self.results.get("model"),
            "split_mode": self.results.get("split_mode"),
            "splits": {},
            "summary": self.results.get("summary"),
        }
        for split_name, split_data in self.results.get("splits", {}).items():
            clean = {k: v for k, v in split_data.items() if k not in {"predictions", "targets"}}
            export["splits"][split_name] = clean
        with open(path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"  Results saved to {path}")

    def _build_summary(self, test_key: str) -> Dict:
        summary = {
            "model": self.model_name,
            "split_mode": self.results.get("split_mode"),
        }
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
