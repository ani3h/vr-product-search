import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study Orchestrator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset directory (containing list_eval_partition.txt)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="List of random seeds")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Config B and C")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for fine-tuning in Config C")
    parser.add_argument("--output_dir", type=str, default="results/ablation", help="Where to save results")
    return parser.parse_args()

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    return result

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    py_exec = sys.executable

    for seed in args.seeds:
        print(f"\n{'='*20}")
        print(f"Starting Seed: {seed}")
        print(f"{'='*20}")

        # --- Configuration A: Vision-only CLIP (Alpha=1, Frozen) ---
        print("\n[Config A] Baseline: Vision-only (Alpha=1, Frozen)")
        config_a_path = os.path.join(args.output_dir, f"idx_A_seed_{seed}")
        run_command([
            py_exec, "src/indexing.py",
            "--data_dir", args.data_dir,
            "--index_path", config_a_path,
            "--alpha", "1.0",
            "--seed", str(seed)
        ])
        eval_a_csv = os.path.join(args.output_dir, f"eval_A_seed_{seed}.csv")
        run_command([
            py_exec, "evaluate.py",
            "--data_root", args.data_dir,
            "--index_path", config_a_path,
            "--alpha", "1.0",
            "--output", eval_a_csv
        ])
        res_a = pd.read_csv(eval_a_csv).mean().to_dict()
        res_a.update({"Config": "A", "Seed": seed, "Alpha": 1.0, "Finetuned": False})
        all_results.append(res_a)

        # --- Configuration B: Frozen CLIP + Frozen BLIP-2 ---
        print("\n[Config B] Frozen CLIP + Frozen BLIP-2")
        config_b_path = os.path.join(args.output_dir, f"idx_B_seed_{seed}")
        run_command([
            py_exec, "src/indexing.py",
            "--data_dir", args.data_dir,
            "--index_path", config_b_path,
            "--alpha", str(args.alpha),
            "--seed", str(seed)
        ])
        eval_b_csv = os.path.join(args.output_dir, f"eval_B_seed_{seed}.csv")
        run_command([
            py_exec, "evaluate.py",
            "--data_root", args.data_dir,
            "--index_path", config_b_path,
            "--alpha", str(args.alpha),
            "--output", eval_b_csv
        ])
        res_b = pd.read_csv(eval_b_csv).mean().to_dict()
        res_b.update({"Config": "B", "Seed": seed, "Alpha": args.alpha, "Finetuned": False})
        all_results.append(res_b)

        # --- Configuration C: Fine-tuned CLIP + Frozen BLIP-2 ---
        print("\n[Config C] Fine-tuned CLIP + Frozen BLIP-2")
        ft_model_dir = os.path.join(args.output_dir, f"ft_CLIP_seed_{seed}")
        run_command([
            py_exec, "src/finetune.py",
            "--data_dir", args.data_dir,
            "--output_dir", ft_model_dir,
            "--epochs", str(args.epochs),
            "--seed", str(seed)
        ])
        ft_model_path = os.path.join(ft_model_dir, f"clip_finetuned_seed_{seed}.pt")
        
        config_c_path = os.path.join(args.output_dir, f"idx_C_seed_{seed}")
        run_command([
            py_exec, "src/indexing.py",
            "--data_dir", args.data_dir,
            "--index_path", config_c_path,
            "--alpha", str(args.alpha),
            "--seed", str(seed),
            "--clip_model_path", ft_model_path
        ])
        eval_c_csv = os.path.join(args.output_dir, f"eval_C_seed_{seed}.csv")
        run_command([
            py_exec, "evaluate.py",
            "--data_root", args.data_dir,
            "--index_path", config_c_path,
            "--alpha", str(args.alpha),
            "--model_path", ft_model_path,
            "--output", eval_c_csv
        ])
        res_c = pd.read_csv(eval_c_csv).mean().to_dict()
        res_c.update({"Config": "C", "Seed": seed, "Alpha": args.alpha, "Finetuned": True})
        all_results.append(res_c)

    # Final Aggregation
    summary_df = pd.DataFrame(all_results)
    
    # Calculate Mean and Std grouped by Config
    metrics = [c for c in summary_df.columns if "Recall" in c or "NDCG" in c or "mAP" in c]
    summary = summary_df.groupby("Config")[metrics].agg(['mean', 'std']).reset_index()
    
    # Flatten columns and format
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    
    output_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary.to_csv(output_path, index=False)
    
    print("\n--- Final Ablation Summary ---")
    print(summary)
    print(f"\nSummary saved to {output_path}")

if __name__ == "__main__":
    main()
