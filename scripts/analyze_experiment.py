"""
scripts/analyze_experiment.py

Unified experiment-analysis runner.

One command:
  python scripts/analyze_experiment.py --exp_dir /path/to/experiment

This orchestrates evaluation, artifact staging, plot generation, tables,
reports, comparison summaries, and cleanup.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate import evaluate_one, _load_config_any
from scripts.generate_research_plots import run_all_plots


@dataclass
class DiscoveryResult:
    exp_dir: Path
    analysis_dir: Path
    model_name: str
    stage: str
    semantic_space: str
    config: Dict
    eval_results: List[Path]
    checkpoints: List[Path]
    has_predictions: bool
    has_embeddings: bool
    available_splits: List[str]


def _stage_from_eval_file(path: Path) -> str:
    return path.stem.replace("_eval_results", "")


class ConfigLoader:
    @staticmethod
    def load_optional(exp_dir: Path) -> Optional[Dict]:
        cfg_yaml = exp_dir / "config.yaml"
        cfg_json = exp_dir / "config.json"
        if cfg_yaml.exists() or cfg_json.exists():
            return _load_config_any(str(exp_dir))
        return None


class CheckpointResolver:
    @staticmethod
    def discover(exp_dir: Path) -> List[Path]:
        candidates = []
        for rel in [
            Path("checkpoints") / "best_model.pt",
            Path("checkpoints") / "best.pt",
            Path("best_model.pt"),
            Path("best.pt"),
        ]:
            candidate = exp_dir / rel
            if candidate.exists():
                candidates.append(candidate)
        return candidates


class ArtifactExporter:
    @staticmethod
    def copy_inputs(
        exp_dir: Path,
        analysis_dir: Path,
        eval_results: List[Path],
        synthetic_config: Optional[Dict] = None,
    ) -> None:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["predictions", "embeddings"]:
            source = exp_dir / subdir
            if source.exists():
                shutil.copytree(source, analysis_dir / subdir, dirs_exist_ok=True)

        for path in eval_results:
            shutil.copy2(path, analysis_dir / path.name)

        for cfg_name in ["config.yaml", "config.json", "metrics.json", "summary.json"]:
            source = exp_dir / cfg_name
            if source.exists():
                shutil.copy2(source, analysis_dir / source.name)

        if synthetic_config is not None and not (analysis_dir / "config.json").exists():
            (analysis_dir / "config.json").write_text(json.dumps(synthetic_config, indent=2), encoding="utf-8")

        # Keep the canonical evaluation JSON in the analysis root for the
        # plotting pipeline to discover automatically.
        for cm_dir in ["confusion_matrices"]:
            source = exp_dir / cm_dir
            if source.exists():
                shutil.copytree(source, analysis_dir / cm_dir, dirs_exist_ok=True)

    @staticmethod
    def trim_temporary_inputs(analysis_dir: Path) -> None:
        # Prediction arrays are only inputs for plot generation; the final
        # analysis package keeps embeddings but not raw prediction staging.
        pred_dir = analysis_dir / "predictions"
        if pred_dir.exists():
            shutil.rmtree(pred_dir, ignore_errors=True)

        cm_dir = analysis_dir / "confusion_matrices"
        if cm_dir.exists():
            shutil.rmtree(cm_dir, ignore_errors=True)


class ReportGenerator:
    @staticmethod
    def write_root_summary(exp_dir: Path, discovery: DiscoveryResult, results: Dict, analysis_dir: Path) -> None:
        summary_path = exp_dir / "evaluation_summary.md"
        lines = []
        lines.append("# Experiment Analysis Summary")
        lines.append("")
        lines.append(f"Experiment: {exp_dir.name}")
        lines.append(f"Model: {discovery.model_name}")
        lines.append(f"Stage: {discovery.stage}")
        lines.append(f"Semantic space: {discovery.semantic_space}")
        lines.append("")
        lines.append("## Available artifacts")
        lines.append(f"- Evaluation JSONs: {len(discovery.eval_results)}")
        lines.append(f"- Checkpoints: {len(discovery.checkpoints)}")
        lines.append(f"- Predictions present: {discovery.has_predictions}")
        lines.append(f"- Embeddings present: {discovery.has_embeddings}")
        lines.append(f"- Splits: {', '.join(discovery.available_splits) if discovery.available_splits else 'none'}")
        lines.append("")
        lines.append("## Key metrics")
        for split_name, split_data in results.get("splits", {}).items():
            metrics = split_data.get("metrics", {})
            group_metrics = split_data.get("group_metrics", {})
            lines.append(f"### {split_name}")
            lines.append(f"- Accuracy: {metrics.get('accuracy', 'n/a')}")
            lines.append(f"- Macro F1: {metrics.get('f1_macro', 'n/a')}")
            lines.append(f"- MCC: {metrics.get('mcc', 'n/a')}")
            if group_metrics:
                lines.append(f"- Grouped Accuracy: {group_metrics.get('accuracy', 'n/a')}")
                lines.append(f"- Grouped Macro F1: {group_metrics.get('f1_macro', 'n/a')}")
        lines.append("")
        lines.append("## Output package")
        lines.append(f"- Analysis directory: {analysis_dir}")
        lines.append("- Plots: analysis/plots/")
        lines.append("- Tables: analysis/tables/")
        lines.append("- Reports: analysis/reports/")
        lines.append("- Embeddings: analysis/embeddings/")
        lines.append("- Comparisons: analysis/comparisons/")
        lines.append("")
        lines.append("## Notes")
        lines.append("This run used the audited single-pass evaluation pipeline and staged exports.")
        summary_path.write_text("\n".join(lines), encoding="utf-8")


class CleanupManager:
    @staticmethod
    def cleanup() -> None:
        gc.collect()


class ExperimentAnalysisRunner:
    def __init__(self, exp_dir: str, seed: int, dpi: int, no_embeddings: bool) -> None:
        self.exp_dir = Path(exp_dir).resolve()
        self.analysis_dir = self.exp_dir / "analysis"
        self.seed = seed
        self.dpi = dpi
        self.no_embeddings = no_embeddings
        self.discovery: Optional[DiscoveryResult] = None
        self.results: Optional[Dict] = None
        self.synthetic_config: Optional[Dict] = None

    def discover(self) -> DiscoveryResult:
        config = ConfigLoader.load_optional(self.exp_dir)
        eval_results = sorted(self.exp_dir.glob("*_eval_results.json"))
        checkpoints = CheckpointResolver.discover(self.exp_dir)
        has_predictions = (self.exp_dir / "predictions").exists()
        has_embeddings = (self.exp_dir / "embeddings").exists()
        available_splits = []

        model_name = self.exp_dir.name
        stage = "unknown"
        semantic_space = "unknown"
        if config is not None:
            model_name = config.get("model", {}).get("name", model_name)
            stage = config.get("task", {}).get("stage", stage)
            semantic_space = config.get("model", {}).get("semantic_space", semantic_space)

        if eval_results:
            stage = _stage_from_eval_file(eval_results[0])
            with open(eval_results[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            model_name = data.get("model", model_name)
            first_split = next(iter(data.get("splits", {})), None)
            if first_split is not None:
                semantic_space = data.get("splits", {}).get(first_split, {}).get("semantic_space", semantic_space)
            available_splits = list(data.get("splits", {}).keys())

            # Synthesize a minimal config for legacy experiments without a saved
            # config file so the plotting pipeline can still infer labels.
            if config is None:
                inferred_n_classes = data.get("splits", {}).get(first_split, {}).get("n_classes", 0) if first_split else 0
                clinical_ids = []
                if stage == "transfer_5class":
                    clinical_ids = data.get("splits", {}).get(first_split, {}).get("original_present_classes", []) if first_split else []
                elif inferred_n_classes:
                    clinical_ids = list(range(int(inferred_n_classes)))
                self.synthetic_config = {
                    "model": {
                        "name": model_name,
                        "semantic_space": semantic_space,
                    },
                    "task": {
                        "stage": stage if stage != "unknown" else "transfer_5class",
                        "clinical_sparse_global_ids": clinical_ids,
                    },
                }
                config = self.synthetic_config

        if config is None:
            config = {}

        self.discovery = DiscoveryResult(
            exp_dir=self.exp_dir,
            analysis_dir=self.analysis_dir,
            model_name=model_name,
            stage=stage,
            semantic_space=semantic_space,
            config=config,
            eval_results=eval_results,
            checkpoints=checkpoints,
            has_predictions=has_predictions,
            has_embeddings=has_embeddings,
            available_splits=available_splits,
        )
        return self.discovery

    def evaluate(self) -> Dict:
        if ConfigLoader.load_optional(self.exp_dir) is None:
            print("[AnalysisRunner] No config file found; packaging existing artifacts without re-evaluation.")
            if self.discovery is not None and self.discovery.eval_results:
                with open(self.discovery.eval_results[0], "r", encoding="utf-8") as f:
                    self.results = json.load(f)
                return self.results
            raise FileNotFoundError(
                "No config file and no existing evaluation results found; cannot evaluate or package the experiment."
            )

        print("[AnalysisRunner] Running evaluation...")
        self.results = evaluate_one(
            str(self.exp_dir),
            self.seed,
            save_outputs=True,
            include_predictions=True,
            use_staging=True,
        )
        return self.results

    def prepare_analysis_inputs(self) -> None:
        assert self.discovery is not None
        print("[AnalysisRunner] Staging analysis inputs...")
        ArtifactExporter.copy_inputs(
            self.exp_dir,
            self.analysis_dir,
            self.discovery.eval_results,
            synthetic_config=self.synthetic_config,
        )

    def generate_analysis_package(self) -> None:
        assert self.discovery is not None
        print("[AnalysisRunner] Generating plots, tables, and reports...")
        run_all_plots(
            str(self.analysis_dir),
            dpi=self.dpi,
            no_staging=False,
            no_embeddings=self.no_embeddings,
        )
        # Keep the package clean: raw prediction arrays are only needed as
        # intermediate inputs for plotting.
        ArtifactExporter.trim_temporary_inputs(self.analysis_dir)

        comparisons_src = self.analysis_dir / "research_plots" / "model_comparisons"
        comparisons_dst = self.analysis_dir / "comparisons"
        comparisons_dst.mkdir(parents=True, exist_ok=True)
        if comparisons_src.exists():
            for item in comparisons_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, comparisons_dst / item.name)

    def write_summary(self) -> None:
        assert self.discovery is not None
        assert self.results is not None
        ReportGenerator.write_root_summary(self.exp_dir, self.discovery, self.results, self.analysis_dir)

    def run(self) -> Dict:
        self.discover()
        results = self.evaluate()
        self.discover()
        self.prepare_analysis_inputs()
        self.generate_analysis_package()
        self.write_summary()
        CleanupManager.cleanup()
        print(f"[AnalysisRunner] Completed analysis package in {self.analysis_dir}")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment analysis runner")
    parser.add_argument("--exp_dir", required=True, help="Path to experiment directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentAnalysisRunner(
        exp_dir=args.exp_dir,
        seed=args.seed,
        dpi=args.dpi,
        no_embeddings=args.no_embeddings,
    )
    runner.run()


if __name__ == "__main__":
    main()
