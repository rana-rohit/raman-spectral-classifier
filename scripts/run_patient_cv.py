"""
scripts/run_patient_cv.py

Orchestration script to run 5-fold patient-aware cross-validation sequentially.
Runs scripts/train.py with --split-mode patient_cv and --fold 0..4, and then
calls scripts/aggregate_folds.py on the resulting experiment folder.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate 5-fold Patient-level Cross-Validation")
    p.add_argument("--model", required=True, choices=["cnn", "resnet1d", "seresnet1d", "tcn", "transformer", "inception1d", "cnn_transformer"])
    p.add_argument("--stage", default="s3_transfer", choices=["s1_isolate", "s2_treatment", "s3_transfer"])
    p.add_argument("--exp-name", default=None, help="Base experiment name prefix")
    p.add_argument("--exp-dir", default="experiments", help="Parent experiment output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--run-finetune", action="store_true", help="Run explicit post-training clinical finetuning")
    p.add_argument("--two-stage", action="store_true", help="Enable decoupled two-stage representation and linear classifier training")
    p.add_argument("--override", nargs="*", default=[], help="Dotlist overrides for config parameters")
    args, dotlist_overrides = p.parse_known_args()
    args.override = list(args.override) + dotlist_overrides
    return args


def main():
    args = parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = args.exp_name or f"{args.model}_s3_patient_cv_{timestamp}"
    
    # We want to keep the folds organized in a single parent directory
    parent_exp_dir = Path(args.exp_dir) / base_name
    parent_exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n============================================================")
    print(f"STARTING 5-FOLD PATIENT-AWARE CROSS-VALIDATION")
    print(f"  Model:      {args.model}")
    print(f"  Stage:      {args.stage}")
    print(f"  Parent Dir: {parent_exp_dir.resolve()}")
    print(f"============================================================\n")
    
    fold_start_time = time.time()
    
    for fold in range(5):
        print(f"\n------------------------------------------------------------")
        print(f"RUNNING FOLD {fold}/4")
        print(f"------------------------------------------------------------\n")
        
        # Build command for train.py
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--model", args.model,
            "--stage", args.stage,
            "--split-mode", "patient_cv",
            "--fold", str(fold),
            "--exp-dir", str(parent_exp_dir),
            "--exp-name", f"fold_{fold}",
            "--seed", str(args.seed),
        ]
        
        if args.run_finetune:
            cmd.append("--run-finetune")
        if args.two_stage:
            cmd.append("--two-stage")
            
        if args.override:
            cmd.extend(args.override)
            
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute training
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"\nError: Fold {fold} training failed with exit code {res.returncode}. Aborting CV run.")
            sys.exit(res.returncode)
            
    print(f"\n============================================================")
    print(f"ALL 5 FOLDS COMPLETED IN {time.time() - fold_start_time:.1f}s.")
    print(f"RUNNING RESULTS AGGREGATION...")
    print(f"============================================================\n")
    
    # Run aggregation
    agg_cmd = [
        sys.executable,
        "scripts/aggregate_folds.py",
        "--run-dir", str(parent_exp_dir),
    ]
    
    print(f"Running command: {' '.join(agg_cmd)}")
    res = subprocess.run(agg_cmd)
    
    if res.returncode == 0:
        print(f"\n============================================================")
        print(f"PATIENT-AWARE 5-FOLD CV RUN SUCCESSFUL!")
        print(f"Results and confusion matrices saved in {parent_exp_dir.resolve()}/")
        print(f"============================================================\n")
    else:
        print(f"\nError: Aggregation failed with exit code {res.returncode}.")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
