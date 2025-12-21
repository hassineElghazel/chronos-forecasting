#!/usr/bin/env python3
"""
Focused ablation: baseline vs cross_learning vs top-k=2 vs thresh=0.7
on ALL datasets to validate selective cross-group attention.

Key question: Does selective attention reduce regressions on 
problematic datasets (dominick, tourism) while preserving gains elsewhere?
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVAL_SCRIPT = SCRIPT_DIR / "evaluate_cross_group.py"

# Configs to compare (reduced set for speed)
CONFIGS = [
    ("baseline", []),
    ("cross_learning", ["--compare-cross-learning"]),
    ("cga_topk2", ["--cross-group-top-k", "2"]),
    ("cga_thresh0.7", ["--cross-group-sim-threshold", "0.7"]),
]

# Use smaller datasets for faster iteration
FAST_CONFIG = SCRIPT_DIR / "configs" / "fast_extended.yaml"

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    batch_size = sys.argv[2] if len(sys.argv) > 2 else "8"
    
    print(f"Running focused ablation on device={device}, batch_size={batch_size}")
    
    # Run each config
    for name, extra_args in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, str(EVAL_SCRIPT), "evaluate",
            "--config-path", str(FAST_CONFIG),
            "--output-path", f"/tmp/focused_{name}.csv",
            "--device", device,
            "--batch-size", batch_size,
            "--no-compare-cross-learning",  # We handle this manually
        ] + extra_args
        
        subprocess.run(cmd, check=True)
    
    print("\n" + "="*60)
    print("Focused ablation complete! Results in /tmp/focused_*.csv")
    print("="*60)

if __name__ == "__main__":
    main()
