from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import LATENT_KEYS, PaperVPLEpisode, build_dataset_splits, dict_to_vec, load_product_embeddings, save_episodes_jsonl
from evaluate import evaluate_vpl_model
from model import PaperStyleVPLRanker


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"


class EpisodeDataset(Dataset):
    def __init__(self, episodes: List[PaperVPLEpisode], product_embeddings: Dict[str, Dict[str, float]]) -> None:
        self.episodes = episodes
        self.product_embeddings = product_embeddings

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        episode = self.episodes[idx]
        context_pair_features = []
        for pair in episode.context_pairs:
            a_vec = dict_to_vec(self.product_embeddings[pair.item_a])
            b_vec = dict_to_vec(self.product_embeddings[pair.item_b])
            context_pair_features.append(np.concatenate([a_vec, b_vec, np.array([pair.label], dtype=np.float32)]))

        query_a_matrix = []
        query_b_matrix = []
        query_labels = []
        for pair in episode.query_pairs:
            query_a_matrix.append(dict_to_vec(self.product_embeddings[pair.item_a]))
            query_b_matrix.append(dict_to_vec(self.product_embeddings[pair.item_b]))
            query_labels.append(float(pair.label))

        candidate_matrix = [
            dict_to_vec(self.product_embeddings[item_id])
            for item_id in episode.candidate_item_ids
        ]
        candidate_array = np.stack(candidate_matrix)
        return {
            "episode_id": episode.episode_id,
            "context_pair_features": torch.tensor(np.stack(context_pair_features), dtype=torch.float32),
            "query_item_a_embeddings": torch.tensor(np.stack(query_a_matrix), dtype=torch.float32),
            "query_item_b_embeddings": torch.tensor(np.stack(query_b_matrix), dtype=torch.float32),
            "query_labels": torch.tensor(np.array(query_labels, dtype=np.float32), dtype=torch.float32),
            "user_latent": torch.tensor(dict_to_vec(episode.user_latent), dtype=torch.float32),
            "candidate_embeddings": torch.tensor(candidate_array, dtype=torch.float32),
        }


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    hidden_dim: int = 128
    batch_size: int = 32
    beta_kl: float = 0.01
    latent_alignment_weight: float = 1.0
    device: str = "cpu"
    seed: int = 42
    num_context_pairs: int = 10
    num_query_pairs: int = 10
    context_sampling: str = "random"
    eval_scoring_mode: str = "latent_utility"
    heldout_path: str | None = None
    test_source: str = "heldout"
    pickle_split_dir: str | None = None


def config_to_dict(config: TrainConfig) -> Dict[str, object]:
    return {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "hidden_dim": config.hidden_dim,
        "batch_size": config.batch_size,
        "beta_kl": config.beta_kl,
        "latent_alignment_weight": config.latent_alignment_weight,
        "device": config.device,
        "seed": config.seed,
        "num_context_pairs": config.num_context_pairs,
        "num_query_pairs": config.num_query_pairs,
        "context_sampling": config.context_sampling,
        "eval_scoring_mode": config.eval_scoring_mode,
        "heldout_path": config.heldout_path,
        "test_source": config.test_source,
        "pickle_split_dir": config.pickle_split_dir,
        "latent_dim": len(LATENT_KEYS),
    }


def ensure_dataset_files(
    product_embeddings: Dict[str, Dict[str, float]],
    config: TrainConfig,
    dataset_dir: Path,
) -> Dict[str, List[PaperVPLEpisode]]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": dataset_dir / "train.jsonl",
        "eval": dataset_dir / "eval.jsonl",
        "test": dataset_dir / "test.jsonl",
    }
    if all(path.exists() for path in split_paths.values()):
        from data import load_episodes_jsonl
        try:
            return {name: load_episodes_jsonl(path) for name, path in split_paths.items()}
        except KeyError:
            pass

    splits = build_dataset_splits(
        product_embeddings=product_embeddings,
        train_size=2000,
        eval_size=200,
        test_size=200,
        num_context_pairs=config.num_context_pairs,
        num_query_pairs=config.num_query_pairs,
        context_sampling=config.context_sampling,
        seed=config.seed,
        heldout_path=config.heldout_path,
        test_source=config.test_source,
        pickle_split_dir=config.pickle_split_dir,
    )
    for split_name, episodes in splits.items():
        save_episodes_jsonl(episodes, split_paths[split_name])
    return splits


def train_model(
    train_episodes: List[PaperVPLEpisode],
    eval_episodes: List[PaperVPLEpisode],
    product_embeddings: Dict[str, Dict[str, float]],
    config: TrainConfig,
    model_path: Path,
) -> tuple[PaperStyleVPLRanker, Dict[str, object], Dict[str, object]]:
    torch.manual_seed(config.seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = PaperStyleVPLRanker(latent_dim=len(LATENT_KEYS), hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dataset = EpisodeDataset(train_episodes, product_embeddings)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    history: Dict[str, object] = {
        "train_loss": [],
        "train_preference_loss": [],
        "train_kl_loss": [],
        "train_latent_mse": [],
        "train_latent_alignment_loss": [],
        "eval_avg_regret": [],
        "eval_avg_normalized_regret": [],
    }
    best_eval_regret = float("inf")
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())
    best_eval_results: Dict[str, object] = {}

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_preference_loss = 0.0
        total_kl_loss = 0.0
        total_latent_mse = 0.0
        total_latent_alignment_loss = 0.0
        num_samples = 0

        for sample in loader:
            context_pair_features = sample["context_pair_features"].to(config.device)
            query_item_a_embeddings = sample["query_item_a_embeddings"].to(config.device)
            query_item_b_embeddings = sample["query_item_b_embeddings"].to(config.device)
            query_labels = sample["query_labels"].to(config.device)
            user_latent = sample["user_latent"].to(config.device)

            posterior_mu, posterior_logvar = model.posterior_stats(context_pair_features)
            latent_sample = model.sample_latent(posterior_mu, posterior_logvar)

            logits = model.preference_logits(
                latent_sample.unsqueeze(1).expand(-1, query_item_a_embeddings.size(1), -1).reshape(-1, model.latent_dim),
                query_item_a_embeddings.reshape(-1, model.latent_dim),
                query_item_b_embeddings.reshape(-1, model.latent_dim),
            ).reshape(query_labels.shape)

            preference_loss = F.binary_cross_entropy_with_logits(logits, query_labels)
            kl_loss = model.kl_divergence(posterior_mu, posterior_logvar)
            semantic_mu = model.semantic_latent(posterior_mu)
            latent_mse = F.mse_loss(semantic_mu, user_latent)
            latent_alignment_loss = config.latent_alignment_weight * latent_mse
            loss = preference_loss + config.beta_kl * kl_loss + latent_alignment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = context_pair_features.size(0)
            total_loss += float(loss.item()) * batch_size
            total_preference_loss += float(preference_loss.item()) * batch_size
            total_kl_loss += float(kl_loss.item()) * batch_size
            total_latent_mse += float(latent_mse.item()) * batch_size
            total_latent_alignment_loss += float(latent_alignment_loss.item()) * batch_size
            num_samples += batch_size

        history["train_loss"].append(total_loss / max(num_samples, 1))
        history["train_preference_loss"].append(total_preference_loss / max(num_samples, 1))
        history["train_kl_loss"].append(total_kl_loss / max(num_samples, 1))
        history["train_latent_mse"].append(total_latent_mse / max(num_samples, 1))
        history["train_latent_alignment_loss"].append(total_latent_alignment_loss / max(num_samples, 1))

        eval_results = evaluate_vpl_model(
            model=model,
            episodes=eval_episodes,
            product_embeddings=product_embeddings,
            device=config.device,
            scoring_mode=config.eval_scoring_mode,
        )
        eval_avg_regret = float(eval_results["metrics"]["avg_regret"])
        eval_avg_normalized_regret = float(eval_results["metrics"]["avg_normalized_regret"])
        history["eval_avg_regret"].append(eval_avg_regret)
        history["eval_avg_normalized_regret"].append(eval_avg_normalized_regret)

        print(
            f"epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={history['train_loss'][-1]:.4f} | "
            f"eval_regret={eval_avg_regret:.4f}"
        )

        if eval_avg_regret < best_eval_regret:
            best_eval_regret = eval_avg_regret
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_eval_results = eval_results
            torch.save(best_state_dict, model_path)

    model.load_state_dict(best_state_dict)
    history["best_eval_regret"] = best_eval_regret
    history["best_epoch"] = best_epoch
    return model, history, best_eval_results


def build_results_paths(results_dir: Path) -> dict[str, Path]:
    return {
        "results_dir": results_dir,
        "dataset_dir": results_dir / "datasets",
        "model_path": results_dir / "best_model.pt",
        "config_path": results_dir / "run_config.json",
        "train_history_path": results_dir / "train_history.json",
        "train_results_path": results_dir / "eval_results.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-budget paper-style VPL-only.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--latent-alignment-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-context-pairs", type=int, default=10)
    parser.add_argument("--num-query-pairs", type=int, default=10)
    parser.add_argument(
        "--context-sampling",
        choices=["random", "dimension_aware"],
        default="random",
    )
    parser.add_argument(
        "--eval-scoring-mode",
        choices=["latent_utility", "reward_model"],
        default="latent_utility",
    )
    parser.add_argument("--heldout-path", type=str, default=None)
    parser.add_argument("--test-source", choices=["heldout", "generated"], default="heldout")
    parser.add_argument("--pickle-split-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or (DEFAULT_RESULTS_DIR / f"k{args.num_context_pairs}")
    paths = build_results_paths(results_dir)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    product_embeddings = load_product_embeddings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        beta_kl=args.beta_kl,
        latent_alignment_weight=args.latent_alignment_weight,
        device=device,
        seed=args.seed,
        num_context_pairs=args.num_context_pairs,
        num_query_pairs=args.num_query_pairs,
        context_sampling=args.context_sampling,
        eval_scoring_mode=args.eval_scoring_mode,
        heldout_path=args.heldout_path,
        test_source=args.test_source,
        pickle_split_dir=args.pickle_split_dir,
    )

    with open(paths["config_path"], "w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2)

    splits = ensure_dataset_files(product_embeddings, config, paths["dataset_dir"])
    _, history, best_eval_results = train_model(
        train_episodes=splits["train"],
        eval_episodes=splits["eval"],
        product_embeddings=product_embeddings,
        config=config,
        model_path=paths["model_path"],
    )

    with open(paths["train_history_path"], "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(paths["train_results_path"], "w", encoding="utf-8") as handle:
        json.dump(best_eval_results, handle, indent=2)

    print(f"Saved best model to {paths['model_path']}")


if __name__ == "__main__":
    main()
