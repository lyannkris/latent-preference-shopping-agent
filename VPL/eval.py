from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load_episodes_jsonl, load_product_embeddings
from evaluate import evaluate_vpl_model
from model import PaperStyleVPLRanker


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"


def load_run_config(results_dir: Path) -> dict[str, object]:
    config_path = results_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-budget paper-style VPL-only.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--num-context-pairs", type=int, default=10)
    parser.add_argument("--scoring-mode", choices=["latent_utility", "reward_model"], default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or (DEFAULT_RESULTS_DIR / f"k{args.num_context_pairs}")
    dataset_dir = results_dir / "datasets"
    model_path = results_dir / "best_model.pt"
    test_results_path = results_dir / "test_results.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    test_path = dataset_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}. Run train.py first.")

    product_embeddings = load_product_embeddings()
    test_episodes = load_episodes_jsonl(test_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_config = load_run_config(results_dir)
    hidden_dim = int(run_config.get("hidden_dim", 128))
    latent_dim = int(run_config.get("latent_dim", 8))
    scoring_mode = args.scoring_mode or str(run_config.get("eval_scoring_mode", "latent_utility"))

    model = PaperStyleVPLRanker(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = evaluate_vpl_model(
        model=model,
        episodes=test_episodes,
        product_embeddings=product_embeddings,
        device=device,
        scoring_mode=scoring_mode,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(test_results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Test metrics:")
    for key, value in results["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved test results to {test_results_path}")


if __name__ == "__main__":
    main()
