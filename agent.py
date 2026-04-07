"""
AlphaRed Agent -- Run this separately after train.py completes.

Usage:
    python train.py          # Step 1: train
    python agent.py          # Step 2: agent improves overnight
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphared.diagnostics.ratchet_loop import RatchetLoop, AgentBudget

OPENROUTER_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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"
)

# Baseline from last training run -- update these after train.py
BASELINE = {
    "val_accuracy": 0.9803,
    "val_loss":     0.0825,
}

def main():
    print("=" * 60)
    print("[AlphaRed Agent] Starting autonomous improvement loop")
    print(f"[AlphaRed Agent] Baseline: {BASELINE}")
    print(f"[AlphaRed Agent] LLM: {'OpenRouter' if OPENROUTER_KEY else 'static fallback'}")
    print("=" * 60)

    output_dir = f"./alphared_agent/agent-{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)

    def train_fn(config):
        import subprocess
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True, text=True,
            timeout=config.time_budget_sec * 2,
            env={**os.environ, "ALPHARED_AGENT_RUN": "1"},
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        output = result.stdout + result.stderr
        metrics = {}
        for line in output.split("\n"):
            for key in ["val_accuracy", "val_loss"]:
                if key + "=" in line.lower() or key + ":" in line.lower():
                    try:
                        sep = "=" if key + "=" in line.lower() else ":"
                        val = float(line.lower().split(key + sep)[1].strip().split()[0])
                        metrics[key] = val
                    except Exception:
                        pass
        if not metrics:
            raise RuntimeError(
                f"Could not parse metrics from output.\n"
                f"Last output:\n{output[-500:]}"
            )
        return metrics

    loop = RatchetLoop(
        run_id             = f"agent-{int(time.time())}",
        train_fn           = train_fn,
        train_file_path    = os.path.abspath("train.py"),
        model_type         = "deep_learning",
        budget             = AgentBudget(
            max_experiments     = 20,
            time_per_experiment = 180,
        ),
        output_dir         = output_dir,
        repo_dir           = "./",
        openrouter_api_key = OPENROUTER_KEY,
        git_enabled        = os.path.exists(".git"),
        status_callback    = print,
    )

    summary = loop.run(BASELINE)

    print("\n" + "=" * 60)
    print("[AlphaRed Agent] Complete!")
    print(f"   Experiments: {summary['experiments_done']}")
    print(f"   Committed:   {summary['committed']}")
    print(f"   Best:        {summary['best_metrics']}")
    print(f"   Changes:     {summary['changes_md_path']}")
    print(f"   Results:     {summary['results_tsv_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
