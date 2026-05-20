"""
src/utils/logging.py

Structured experiment logger and research-grade reporting system.
Every metric at every epoch is persisted to a JSON file alongside console output.
Provides centralized reporting functions to output stage-aware, visually
structured summaries of training, evaluation, and inference.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


# ============================================================
# GROUPED EVALUATION CONFIG
# ============================================================
# Spectra-per-group is a property of the dataset acquisition
# structure. Shared between the evaluator and the provenance
# reporter so both always agree on grouping boundaries.

SPECTRA_PER_GROUP = {
    "test": 100,
    "2018clinical": 400,
    "2019clinical": 100,
}


def _introspect_loader(
    loader,
    split_name: str,
    spectra_per_group_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Extract runtime statistics from a DataLoader's underlying dataset.

    Returns a dict with:
      n_samples, n_classes, class_distribution,
      spectra_per_group, n_groups.
    """
    dataset = loader.dataset
    info: Dict[str, Any] = {}

    try:
        info["n_samples"] = len(dataset)
    except Exception:
        info["n_samples"] = None

    y = getattr(dataset, "y", None)
    if y is not None:
        y_arr = np.asarray(y)
        unique, counts = np.unique(y_arr, return_counts=True)
        info["n_classes"] = int(len(unique))
        info["class_distribution"] = dict(
            zip(unique.astype(int).tolist(), counts.tolist())
        )
    else:
        info["n_classes"] = None
        info["class_distribution"] = {}

    if spectra_per_group_map is None:
        spectra_per_group_map = SPECTRA_PER_GROUP
    spg = spectra_per_group_map.get(split_name)
    info["spectra_per_group"] = spg
    if spg and info["n_samples"]:
        info["n_groups"] = info["n_samples"] // spg
    else:
        info["n_groups"] = None

    return info


def print_split_provenance(
    loaders: Dict,
    cfg: Dict[str, Any],
    context: str = "training",
) -> None:
    """
    Print a dynamic, stage-aware provenance report of all active
    data splits currently loaded in memory.

    Introspects ACTUAL runtime loader objects — no hardcoded
    statistics.  Every number shown is derived from live arrays.

    Args:
        loaders:  The loaders dict returned by build_all_loaders.
        cfg:      The merged experiment config dict.
        context:  One of "training", "evaluation", "finetuning".
    """
    stage = cfg.get("task", {}).get("stage", "unknown")
    label_space = cfg.get("task", {}).get("label_space", "unknown")
    grouped_cfg = cfg.get("evaluation", {}).get("grouped", {})
    grouped_enabled = grouped_cfg.get("enabled", True)
    spectra_per_group_map = grouped_cfg.get(
        "spectra_per_group",
        SPECTRA_PER_GROUP,
    )
    if not grouped_enabled:
        spectra_per_group_map = {}

    stage_map = {
        "pretrain_30class": ("Stage 1", "Isolate-Space Pretraining"),
        "pretrain_treatment_8class": ("Stage 2", "Semantic Treatment Transfer"),
        "transfer_5class": ("Stage 3", "Clinical Transfer / Domain Adaptation"),
    }
    stage_num, stage_desc = stage_map.get(stage, ("?", stage))

    space_map = {
        "isolate_space": "isolate_space (30 reference isolates)",
        "global_treatment_space": "global_treatment_space (8 empiric treatments)",
        "sparse_global_treatment_space": "compact_transfer_space (5 clinical treatments)",
    }
    space_display = space_map.get(label_space, label_space)

    border = "=" * 60

    print(f"\n{border}")
    print("ACTIVE DATA SPLITS")
    print("=" * 18)
    print()

    # Ordered list of top-level loader keys to report
    report_order = [
        "train", "source_val", "finetune",
        "clinical_train", "clinical_val", "test",
    ]

    split_cfg = cfg.get("splits", {})
    role_labels = {
        "train": "SOURCE (train)",
        "source_val": "SOURCE (val)",
        "clinical_train": "OOD (adapt-train)",
        "clinical_val": "OOD (adapt-val)",
    }

    # Table header
    hdr = (
        f"  {'Split':<18} {'Role':<20} "
        f"{'Samples':>8} {'Classes':>8} {'Groups':>8} {'Spec/Grp':>9}"
    )
    sep = (
        f"  {'-'*18} {'-'*20} "
        f"{'-'*8} {'-'*8} {'-'*8} {'-'*9}"
    )
    print(hdr)
    print(sep)

    def _row(name: str, loader) -> None:
        info = _introspect_loader(loader, name, spectra_per_group_map)
        ns = f"{info['n_samples']:,}" if info["n_samples"] is not None else "?"
        nc = str(info["n_classes"]) if info["n_classes"] is not None else "?"
        ng = str(info["n_groups"]) if info["n_groups"] is not None else "--"
        sg = str(info["spectra_per_group"]) if info["spectra_per_group"] else "--"

        if name in role_labels:
            role = role_labels[name]
        elif name in split_cfg:
            role = split_cfg[name].get("role", "unknown").upper()
        else:
            role = "OOD_EVAL"

        print(f"  {name:<18} {role:<20} {ns:>8} {nc:>8} {ng:>8} {sg:>9}")

    for name in report_order:
        if name in loaders and name != "val":
            _row(name, loaders[name])

    ood = loaders.get("ood", {})
    if isinstance(ood, dict):
        for ood_name, ood_loader in ood.items():
            _row(ood_name, ood_loader)

    # -- Label space --
    print()
    print(f"  {stage_num}: {stage_desc}")
    print(f"  Semantic Space: {space_display}")

    # -- Clinical mapping (Stage 3) --
    clinical_ids = cfg.get("task", {}).get("clinical_sparse_global_ids", [])
    if clinical_ids:
        try:
            from metadata.ontology import GLOBAL_TREATMENTS, COMPACT_LABEL_MAP
            print()
            print("  Compact Label Mapping:")
            for gid in sorted(int(g) for g in clinical_ids):
                compact = COMPACT_LABEL_MAP.get(gid, "?")
                treatment = GLOBAL_TREATMENTS.get(gid, "?")
                print(f"    Global {gid} ({treatment}) -> Compact {compact}")
        except ImportError:
            pass

    # -- Grouping details --
    grouped = []
    all_names = [n for n in report_order if n in loaders] + list(
        ood.keys() if isinstance(ood, dict) else []
    )
    for name in all_names:
        spg = spectra_per_group_map.get(name)
        if spg is None:
            continue
        ldr = ood.get(name) if name in (ood if isinstance(ood, dict) else {}) else loaders.get(name)
        if ldr is not None:
            ns = len(ldr.dataset)
            grouped.append((name, spg, ns // spg))

    if grouped:
        print()
        print("  Grouped Evaluation:")
        for name, spg, ng in grouped:
            entity = "patient" if "clinical" in name else "isolate/replicate"
            print(f"    {name}: {spg} spectra per {entity} -> {ng} groups")

    print(f"\n{border}\n")


def print_stage_header(stage: str, task_name: str = "") -> None:
    """
    Print a prominent, stage-aware section header.
    
    Distinguishes Stage 1, 2, and 3 clearly.
    """
    if stage == "pretrain_30class":
        stage_num = 1
        stage_name = "ISOLATE-SPACE PRETRAINING"
    elif stage == "pretrain_treatment_8class":
        stage_num = 2
        stage_name = "SEMANTIC TRANSFER EVALUATION"
    elif stage == "transfer_5class":
        stage_num = 3
        stage_name = "CLINICAL TRANSFER EVALUATION"
    else:
        stage_num = "?"
        stage_name = stage.upper().replace("_", " ")

    title = f"STAGE {stage_num}: {stage_name}"
    border = "=" * 60
    sub_border = "=" * len(title)
    
    print(f"\n{border}")
    print(title)
    print(sub_border)
    if task_name:
        print(f"Active Task: {task_name}")
    print(f"{border}\n")


def print_model_summary(model_name: str, model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Print model family and key structural hyperparameters.
    
    Outputs format: TCN (dilations=[1,2,4], dropout=0.3)
    """
    if model_cfg is None:
        model_cfg = {}
    
    print("MODEL:")
    name_display = model_cfg.get("name", model_name).upper()
    
    # Collect key model properties
    details = []
    if "dilations" in model_cfg:
        details.append(f"dilations={model_cfg['dilations']}")
    if "dropout" in model_cfg:
        details.append(f"dropout={model_cfg['dropout']}")
    if "kernel_size" in model_cfg:
        details.append(f"kernel_size={model_cfg['kernel_size']}")
    if "use_se" in model_cfg:
        details.append(f"use_se={model_cfg['use_se']}")
    if "use_cbam" in model_cfg:
        details.append(f"use_cbam={model_cfg['use_cbam']}")
    if "n_blocks" in model_cfg:
        details.append(f"n_blocks={model_cfg['n_blocks']}")
    if "n_heads" in model_cfg:
        details.append(f"n_heads={model_cfg['n_heads']}")
        
    if details:
        # Format list outputs without unnecessary spacing for clean look
        detail_strs = []
        for d in details:
            k, v = d.split("=", 1)
            # Remove spaces inside representation of lists
            v_clean = v.replace(" ", "")
            detail_strs.append(f"{k}={v_clean}")
        print(f"{name_display} ({', '.join(detail_strs)})")
    else:
        print(name_display)
    print()


def print_feature_summary(cfg: Dict[str, Any]) -> None:
    """Print the active optional research systems for reproducibility."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    supcon_cfg = train_cfg.get("supcon", {})
    eval_cfg = cfg.get("evaluation", {})
    grouped_cfg = eval_cfg.get("grouped", {})
    finetune_cfg = train_cfg.get("finetune", {})

    supcon_enabled = bool(supcon_cfg.get("enabled", False))
    two_stage_enabled = bool(train_cfg.get("two_stage", False))
    finetune_enabled = bool(finetune_cfg.get("enabled", False))
    dann_enabled = bool(train_cfg.get("use_dann", False) or train_cfg.get("dann", {}).get("enabled", False))
    coral_enabled = bool(train_cfg.get("use_coral", False) or train_cfg.get("coral", {}).get("enabled", False))

    if two_stage_enabled:
        paradigm = "two_stage_supcon"
    elif supcon_enabled:
        contrastive_weight = float(supcon_cfg.get("weight", 0.0))
        classification_weight = float(supcon_cfg.get("classification_weight", 1.0))
        paradigm = "pure_supcon" if classification_weight == 0.0 else "joint_supcon"
        if contrastive_weight == 0.0:
            paradigm = "supervised"
    else:
        paradigm = "supervised"

    print("ACTIVE FEATURES:")
    print(f"Training Paradigm: {paradigm}")
    print(f"SupCon:            {supcon_enabled}")
    print(f"Two-stage:         {two_stage_enabled}")
    print(f"SE:                {bool(model_cfg.get('use_se', False))}")
    print(f"CBAM:              {bool(model_cfg.get('use_cbam', False))}")
    print(f"DANN:              {dann_enabled}")
    print(f"CORAL:             {coral_enabled}")
    print(f"Freeze Backbone:   {bool(train_cfg.get('freeze_backbone', False))}")
    print(f"Finetune:          {finetune_enabled}")
    print(f"Grouped Eval:      {bool(grouped_cfg.get('enabled', True))}")
    print()


def print_label_space_info(label_space: str, clinical_sparse_ids: Optional[list] = None) -> None:
    """Print detailed, stage-aware explanation of the target label space."""
    print("LABEL SPACE:")
    print(label_space)
    if clinical_sparse_ids:
        print(f"Clinical Sparse IDs: {clinical_sparse_ids}")
        from metadata.ontology import CLINICAL_LABELS
        for idx in clinical_sparse_ids:
            if int(idx) in CLINICAL_LABELS:
                info = CLINICAL_LABELS[int(idx)]
                print(f"  {idx} -> {info['global_treatment']} ({info['clinical_species']})")
    print()


def print_metric_block(title: str, metrics: Dict[str, float], is_grouped: bool = False) -> None:
    """Print a structured block of evaluation metrics."""
    print(f"## {title}")
    print()
    
    # Extract keys safely
    loss = metrics.get("loss", None)
    acc = metrics.get("accuracy", metrics.get("acc", None))
    f1 = metrics.get("f1_macro", metrics.get("f1", None))
    mcc = metrics.get("mcc", None)
    roc_auc = metrics.get("roc_auc", None)
    
    prefix = "Grouped " if is_grouped else ""
    
    if loss is not None:
        print(f"{prefix}Loss: {loss:.4f}")
    if acc is not None:
        print(f"{prefix}Accuracy: {acc:.4f}")
    if f1 is not None:
        print(f"{prefix}Macro F1: {f1:.4f}")
    if mcc is not None:
        print(f"{prefix}MCC: {mcc:.4f}")
    if roc_auc is not None:
        print(f"{prefix}ROC-AUC: {roc_auc:.4f}")
        
    # Optional per-class F1 print
    per_class_f1 = {k: v for k, v in metrics.items() if k.startswith("f1_class_")}
    if per_class_f1:
        class_f1s = []
        for k, v in sorted(per_class_f1.items(), key=lambda x: int(x[0].split("_")[-1])):
            cls_id = int(k.split("_")[-1])
            class_f1s.append(f"Class {cls_id}: {v:.4f}")
        print(f"Per-Class F1: {', '.join(class_f1s)}")
    print()


def print_checkpoint_info(checkpoint_path: str, loaded: bool = True, details: Optional[Dict[str, Any]] = None) -> None:
    """Print structured checkpoint loading or saving updates."""
    print("## CHECKPOINT INFO")
    print()
    action = "Loaded checkpoint:" if loaded else "Saved checkpoint to:"
    print(action)
    print(checkpoint_path)
    if details:
        for k, v in details.items():
            print(f"  {k}: {v}")
    print()


def print_output_paths(output_paths: Dict[str, str]) -> None:
    """Print structured paths to generated results and visualization assets."""
    print("## OUTPUT FILES")
    print()
    for name, path in output_paths.items():
        print(f"{name}:")
        print(path)
    print()


def print_evaluation_summary(
    stage: str,
    model_name: str,
    model_cfg: Optional[Dict[str, Any]],
    label_space: str,
    clinical_sparse_ids: Optional[list],
    checkpoint_path: Optional[str],
    split_metrics: Dict[str, Dict[str, Any]],
    output_files: Optional[Dict[str, str]] = None
) -> None:
    """Print the complete, research-grade evaluation report for a given stage."""
    print_stage_header(stage)
    print_model_summary(model_name, model_cfg)
    print_label_space_info(label_space, clinical_sparse_ids)
    
    # Print metrics for each split
    for split_name, split_data in split_metrics.items():
        print("---")
        print(f"\nSPLIT: {split_name.upper()}")
        print()
        
        metrics = split_data.get("metrics", {})
        group_metrics = split_data.get("group_metrics", {})
        
        # Clarify metric semantic space
        if stage == "pretrain_30class":
            spec_title = "ISOLATE-SPACE METRICS"
            group_title = "GROUPED ISOLATE-LEVEL METRICS"
        elif stage == "pretrain_treatment_8class":
            spec_title = "TREATMENT-SPACE METRICS"
            group_title = "GROUPED TREATMENT-SPACE METRICS"
        elif stage == "transfer_5class":
            spec_title = "CLINICAL-SPACE METRICS"
            # For clinical OOD, grouped metrics are patient-level, else isolate-level
            if split_name in ("2018clinical", "2019clinical", "clinical_val"):
                group_title = "GROUPED PATIENT-LEVEL CLINICAL METRICS"
            else:
                group_title = "GROUPED ISOLATE-LEVEL CLINICAL METRICS"
        else:
            spec_title = f"{split_name.upper()} SPECTRUM METRICS"
            group_title = f"{split_name.upper()} GROUPED METRICS"
            
        print_metric_block(spec_title, metrics)
        
        if group_metrics:
            print("---")
            print()
            print_metric_block(group_title, group_metrics, is_grouped=True)
            
    print("---")
    print()
    
    if checkpoint_path:
        print_checkpoint_info(checkpoint_path, loaded=True)
        print("---")
        print()
        
    if output_files:
        print_output_paths(output_files)
        print("---")
        print()


class ExperimentLogger:
    """
    Logs metrics to:
      - Console (formatted table with stage-aware semantic columns)
      - <exp_dir>/metrics.json  (full history, machine-readable)
      - <exp_dir>/summary.json  (best metrics per split, for tables)
    """

    def __init__(self, exp_dir: str, model_name: str, config: Dict) -> None:
        self.exp_dir    = Path(exp_dir)
        self.model_name = model_name
        self.config     = config
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self._history: list = []
        self._best: Dict    = {}
        self._start_time    = time.time()

        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        stage = self.config.get("task", {}).get("stage", "")
        task_name = self.config.get("task", {}).get("name", "")
        
        # Centralized Stage Header and Model Summary
        print_stage_header(stage, task_name)
        print_model_summary(model_name, config.get("model", {}))
        print_feature_summary(config)
        print(f"  Directory:  {exp_dir}")
        print(f"{'='*60}\n")

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
        stage = self.config.get("task", {}).get("stage", "")
        
        # Determine semantic block title
        if stage == "pretrain_30class":
            title = f"FINAL ISOLATE-SPACE METRICS ({split.upper()})"
        elif stage == "pretrain_treatment_8class":
            title = f"FINAL TREATMENT-SPACE METRICS ({split.upper()})"
        else:
            title = f"FINAL CLINICAL-TRANSFER METRICS ({split.upper()})"
            
        print_metric_block(title, metrics)
        
        self._best[f"final_{split}"] = metrics
        self._flush()

    def _print_row(self, epoch: int, split: str, metrics: Dict) -> None:
        stage = self.config.get("task", {}).get("stage", "")
        acc  = metrics.get("accuracy", float("nan"))
        loss = metrics.get("loss",     float("nan"))
        f1   = metrics.get("f1_macro", float("nan"))
        
        # Set stage-aware column headers to avoid ambiguous metric names
        if stage == "pretrain_30class":
            acc_name = "isolate_acc"
            f1_name = "isolate_f1"
        elif stage == "pretrain_treatment_8class":
            acc_name = "treatment_acc"
            f1_name = "treatment_f1"
        elif stage == "transfer_5class":
            acc_name = "clinical_acc"
            f1_name = "clinical_f1"
        else:
            acc_name = "acc"
            f1_name = "f1"
            
        # Determine logging configurations
        verbose_losses = self.config.get("logging", {}).get("verbose_losses", False)
        contrastive_enabled = self.config.get("model", {}).get("contrastive", False)

        # 1. Print Metric Row: Primary Performance Metrics Visually Dominate
        print(f"  Ep {epoch:>3d} | {split:<12} | {acc_name}={acc:.4f}  {f1_name}={f1:.4f}")

        # 2. Print Loss Row: Secondary, Compact, and Visually Grouped
        if verbose_losses:
            # Verbose Mode: print decomposition of losses
            parts = [f"total={loss:.4f}"]
            if contrastive_enabled:
                contrastive_loss = metrics.get("contrastive_loss", 0.0)
                classification_loss = metrics.get("classification_loss", 0.0)
                if classification_loss > 0:
                    parts.append(f"ce={classification_loss:.4f}")
                if contrastive_loss > 0:
                    parts.append(f"supcon={contrastive_loss:.4f}")
            else:
                main_loss = metrics.get("main_loss", 0.0)
                source_loss = metrics.get("source_loss", 0.0)
                if main_loss > 0 and abs(main_loss - loss) > 1e-4:
                    parts.append(f"main={main_loss:.4f}")
                if source_loss > 0 and abs(source_loss - loss) > 1e-4:
                    parts.append(f"source={source_loss:.4f}")

            # Include any domain adaptation or aux losses if active
            coral_loss = metrics.get("coral_loss", 0.0)
            domain_loss = metrics.get("domain_loss", 0.0)
            aux_loss = metrics.get("aux_loss", 0.0)
            consistency_loss = metrics.get("consistency_loss", 0.0)

            if coral_loss > 0:
                parts.append(f"coral={coral_loss:.4f}")
            if domain_loss > 0:
                parts.append(f"domain={domain_loss:.4f}")
            if aux_loss > 0:
                parts.append(f"aux={aux_loss:.4f}")
            if consistency_loss > 0:
                parts.append(f"consistency={consistency_loss:.4f}")

            loss_str = " | ".join(parts)
            print(f"           [loss: {loss_str}]")
        else:
            # Compact Mode: Only total loss is shown
            print(f"           [loss: total={loss:.4f}]")

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
