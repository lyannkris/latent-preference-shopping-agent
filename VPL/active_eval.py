from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from data import (
    LATENT_KEYS,
    PairwisePreference,
    compute_utility,
    dict_to_vec,
    label_item_pair,
    load_episodes_jsonl,
    load_product_embeddings,
)
from eval import load_run_config
from model import PaperStyleVPLRanker


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "active_test_results.json"


def pair_feature(
    product_embeddings: Dict[str, Dict[str, float]],
    pair: PairwisePreference,
) -> np.ndarray:
    return np.concatenate(
        [
            dict_to_vec(product_embeddings[pair.item_a]),
            dict_to_vec(product_embeddings[pair.item_b]),
            np.array([pair.label], dtype=np.float32),
        ]
    )


def posterior_entropy_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
    latent_dim = logvar.size(-1)
    constant = latent_dim * math.log(2.0 * math.pi * math.e)
    return 0.5 * (constant + torch.sum(logvar, dim=-1))


def build_candidate_pairs(
    product_embeddings: Dict[str, Dict[str, float]],
    max_candidate_pairs: int,
    seed: int,
) -> List[Tuple[str, str]]:
    product_ids = list(product_embeddings.keys())
    all_pairs = [
        (product_ids[i], product_ids[j])
        for i in range(len(product_ids))
        for j in range(i + 1, len(product_ids))
    ]
    if max_candidate_pairs <= 0 or max_candidate_pairs >= len(all_pairs):
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, max_candidate_pairs)


def select_active_pair(
    model: PaperStyleVPLRanker,
    product_embeddings: Dict[str, Dict[str, float]],
    selected_pairs: List[PairwisePreference],
    candidate_pairs: List[Tuple[str, str]],
    used_pair_keys: set[Tuple[str, str]],
    device: str,
) -> Tuple[str, str]:
    base_features = [pair_feature(product_embeddings, pair) for pair in selected_pairs]
    candidate_features = []
    candidate_pairs_in_order = []
    candidate_keys = []

    for item_a, item_b in candidate_pairs:
        pair_key = tuple(sorted((item_a, item_b)))
        if pair_key in used_pair_keys:
            continue
        feature_label_0 = pair_feature(product_embeddings, PairwisePreference(item_a=item_a, item_b=item_b, label=0))
        feature_label_1 = pair_feature(product_embeddings, PairwisePreference(item_a=item_a, item_b=item_b, label=1))
        if base_features:
            context_0 = np.stack(base_features + [feature_label_0])
            context_1 = np.stack(base_features + [feature_label_1])
        else:
            context_0 = np.stack([feature_label_0])
            context_1 = np.stack([feature_label_1])
        candidate_features.append(context_0)
        candidate_features.append(context_1)
        candidate_pairs_in_order.append((item_a, item_b))
        candidate_keys.append(pair_key)

    if not candidate_keys:
        raise RuntimeError("No unused candidate pairs remain for active selection.")

    context_tensor = torch.tensor(np.stack(candidate_features), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, logvar = model.posterior_stats(context_tensor)
        entropies = posterior_entropy_from_logvar(logvar).reshape(-1, 2).mean(dim=1)
    best_idx = int(torch.argmin(entropies).item())
    return candidate_pairs_in_order[best_idx]


def posterior_entropy_for_pairs(
    model: PaperStyleVPLRanker,
    product_embeddings: Dict[str, Dict[str, float]],
    selected_pairs: List[PairwisePreference],
    device: str,
) -> float:
    pair_features = np.stack([pair_feature(product_embeddings, pair) for pair in selected_pairs])
    pair_tensor = torch.tensor(pair_features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, logvar = model.posterior_stats(pair_tensor)
    return float(posterior_entropy_from_logvar(logvar).item())


def active_query_context(
    model: PaperStyleVPLRanker,
    user_latent: Dict[str, float],
    product_embeddings: Dict[str, Dict[str, float]],
    candidate_pairs: List[Tuple[str, str]],
    num_context_pairs: int,
    device: str,
    selection_strategy: str,
    rng: random.Random,
) -> Tuple[List[PairwisePreference], List[float], List[float]]:
    selected_pairs: List[PairwisePreference] = []
    used_pair_keys: set[Tuple[str, str]] = set()
    prior_entropy = 0.5 * len(LATENT_KEYS) * math.log(2.0 * math.pi * math.e)
    previous_entropy = prior_entropy
    entropy_values: List[float] = []
    entropy_reductions: List[float] = []
    for _ in range(num_context_pairs):
        if selection_strategy == "active":
            item_a, item_b = select_active_pair(
                model=model,
                product_embeddings=product_embeddings,
                selected_pairs=selected_pairs,
                candidate_pairs=candidate_pairs,
                used_pair_keys=used_pair_keys,
                device=device,
            )
        elif selection_strategy == "random":
            unused_pairs = [
                (item_a, item_b)
                for item_a, item_b in candidate_pairs
                if tuple(sorted((item_a, item_b))) not in used_pair_keys
            ]
            if not unused_pairs:
                raise RuntimeError("No unused candidate pairs remain for random selection.")
            item_a, item_b = rng.choice(unused_pairs)
        else:
            raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
        labeled_pair = label_item_pair(
            user_latent=user_latent,
            product_embeddings=product_embeddings,
            item_a=item_a,
            item_b=item_b,
        )
        selected_pairs.append(labeled_pair)
        used_pair_keys.add(tuple(sorted((item_a, item_b))))
        current_entropy = posterior_entropy_for_pairs(
            model=model,
            product_embeddings=product_embeddings,
            selected_pairs=selected_pairs,
            device=device,
        )
        entropy_values.append(current_entropy)
        entropy_reductions.append(previous_entropy - current_entropy)
        previous_entropy = current_entropy
    return selected_pairs, entropy_values, entropy_reductions


def evaluate_active_queries(
    model: PaperStyleVPLRanker,
    episodes,
    product_embeddings: Dict[str, Dict[str, float]],
    candidate_pairs: List[Tuple[str, str]],
    num_context_pairs: int,
    threshold_ratio: float,
    device: str,
    scoring_mode: str,
    selection_strategy: str,
    seed: int,
) -> Dict[str, object]:
    model.eval()
    utilities = []
    regrets = []
    normalized_regrets = []
    threshold_successes = []
    success_at_1 = []
    success_at_3 = []
    success_at_5 = []
    success_at_10 = []
    success_at_50 = []
    success_at_100 = []
    success_at_200 = []
    success_at_500 = []
    success_at_1000 = []
    entropy_after = []
    uncertainty_reductions = []
    per_episode = []
    rng = random.Random(seed)

    for episode in episodes:
        context_pairs, entropy_values, entropy_reductions = active_query_context(
            model=model,
            user_latent=episode.user_latent,
            product_embeddings=product_embeddings,
            candidate_pairs=candidate_pairs,
            num_context_pairs=num_context_pairs,
            device=device,
            selection_strategy=selection_strategy,
            rng=rng,
        )
        pair_features = np.stack([pair_feature(product_embeddings, pair) for pair in context_pairs])
        pair_tensor = torch.tensor(pair_features, dtype=torch.float32, device=device).unsqueeze(0)
        candidate_matrix = np.array(
            [dict_to_vec(product_embeddings[item_id]) for item_id in episode.candidate_item_ids],
            dtype=np.float32,
        )
        candidate_tensor = torch.tensor(candidate_matrix, dtype=torch.float32, device=device)

        with torch.no_grad():
            posterior_mu, posterior_logvar = model.posterior_stats(pair_tensor)
            if scoring_mode == "reward_model":
                scores = model.score_candidates(posterior_mu.squeeze(0), candidate_tensor)
            elif scoring_mode == "latent_utility":
                scores = model.score_candidates_by_latent_utility(posterior_mu, candidate_tensor)
            else:
                raise ValueError(f"Unknown scoring_mode: {scoring_mode}")
            inferred_latent = model.semantic_latent(posterior_mu).squeeze(0).cpu().numpy()
            entropy = float(posterior_entropy_from_logvar(posterior_logvar).item())

        scores_np = scores.cpu().numpy()
        pred_idx = int(np.argmax(scores_np))
        pred_item_id = episode.candidate_item_ids[pred_idx]

        true_latent_vec = dict_to_vec(episode.user_latent)
        all_utilities = np.array(
            [compute_utility(true_latent_vec, dict_to_vec(product_embeddings[item_id])) for item_id in episode.candidate_item_ids],
            dtype=np.float32,
        )
        chosen_utility = compute_utility(true_latent_vec, dict_to_vec(product_embeddings[pred_item_id]))
        oracle_utility = float(np.max(all_utilities))
        worst_utility = float(np.min(all_utilities))
        regret = float(oracle_utility - chosen_utility)
        max_possible_regret = max(oracle_utility - worst_utility, 1e-8)
        normalized_regret = regret / max_possible_regret
        utility_threshold = float(np.percentile(all_utilities, 70))
        threshold_success = int(chosen_utility >= utility_threshold)
        top1 = np.argsort(all_utilities)[-1:]
        top3 = np.argsort(all_utilities)[-3:]
        top5 = np.argsort(all_utilities)[-5:]
        top10 = np.argsort(all_utilities)[-10:]
        top50 = np.argsort(all_utilities)[-50:]
        top100 = np.argsort(all_utilities)[-100:]
        top200 = np.argsort(all_utilities)[-200:]
        top500 = np.argsort(all_utilities)[-500:]
        top1000 = np.argsort(all_utilities)[-1000:]

        utilities.append(chosen_utility)
        regrets.append(regret)
        normalized_regrets.append(normalized_regret)
        threshold_successes.append(threshold_success)
        success_at_1.append(int(pred_idx in top1))
        success_at_3.append(int(pred_idx in top3))
        success_at_5.append(int(pred_idx in top5))
        success_at_10.append(int(pred_idx in top10))
        success_at_50.append(int(pred_idx in top50))
        success_at_100.append(int(pred_idx in top100))
        success_at_200.append(int(pred_idx in top200))
        success_at_500.append(int(pred_idx in top500))
        success_at_1000.append(int(pred_idx in top1000))
        entropy_after.append(entropy)
        uncertainty_reductions.extend(entropy_reductions)

        per_episode.append(
            {
                "episode_id": episode.episode_id,
                "user_id": episode.user_id,
                "active_context_pairs": [pair.to_dict() for pair in context_pairs],
                "recommended_item_id": pred_item_id,
                "oracle_best_item_id": episode.oracle_best_item_id,
                "chosen_utility": chosen_utility,
                "oracle_utility": oracle_utility,
                "worst_utility": worst_utility,
                "regret": regret,
                "max_possible_regret": max_possible_regret,
                "normalized_regret": normalized_regret,
                "utility_70th_percentile": utility_threshold,
                "threshold_success": threshold_success,
                "success_at_1": int(pred_idx in top1),
                "success_at_3": int(pred_idx in top3),
                "success_at_5": int(pred_idx in top5),
                "success_at_10": int(pred_idx in top10),
                "success_at_50": int(pred_idx in top50),
                "success_at_100": int(pred_idx in top100),
                "success_at_200": int(pred_idx in top200),
                "success_at_500": int(pred_idx in top500),
                "success_at_1000": int(pred_idx in top1000),
                "posterior_entropy_after": entropy,
                "posterior_entropy_by_turn": entropy_values,
                "uncertainty_reduction_by_turn": entropy_reductions,
                "inferred_latent": inferred_latent.tolist(),
            }
        )

    return {
        "metrics": {
            "avg_utility": float(np.mean(utilities)),
            "avg_regret": float(np.mean(regrets)),
            "avg_normalized_regret": float(np.mean(normalized_regrets)),
            "threshold_success_rate": float(np.mean(threshold_successes)),
            "success_at_1": float(np.mean(success_at_1)),
            "success_at_3": float(np.mean(success_at_3)),
            "success_at_5": float(np.mean(success_at_5)),
            "success_at_10": float(np.mean(success_at_10)),
            "success_at_50": float(np.mean(success_at_50)),
            "success_at_100": float(np.mean(success_at_100)),
            "success_at_200": float(np.mean(success_at_200)),
            "success_at_500": float(np.mean(success_at_500)),
            "success_at_1000": float(np.mean(success_at_1000)),
            "avg_context_pairs": float(num_context_pairs),
            "avg_posterior_entropy_after": float(np.mean(entropy_after)),
            "avg_uncertainty_reduction": float(np.mean(uncertainty_reductions)),
            "threshold_definition": "recommended utility >= 70th percentile of episode utilities",
            "normalized_regret_definition": "regret / (best utility - worst utility)",
        },
        "per_episode": per_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VPL active pairwise query selection.")
    parser.add_argument("--base-results-dir", type=Path, default=DEFAULT_BASE_RESULTS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-context-pairs", type=int, default=8)
    parser.add_argument("--max-candidate-pairs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scoring-mode", choices=["reward_model", "latent_utility"], default="reward_model")
    parser.add_argument("--selection-strategy", choices=["active", "random"], default="active")
    parser.add_argument("--threshold-ratio", type=float, default=0.95)
    args = parser.parse_args()

    product_embeddings = load_product_embeddings()
    test_episodes = load_episodes_jsonl(args.base_results_dir / "datasets" / "test.jsonl")
    run_config = load_run_config(args.base_results_dir)
    hidden_dim = int(run_config.get("hidden_dim", 128))
    latent_dim = int(run_config.get("latent_dim", len(LATENT_KEYS)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PaperStyleVPLRanker(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(args.base_results_dir / "best_model.pt", map_location=device))
    model.eval()

    candidate_pairs = build_candidate_pairs(
        product_embeddings=product_embeddings,
        max_candidate_pairs=args.max_candidate_pairs,
        seed=args.seed,
    )
    results = evaluate_active_queries(
        model=model,
        episodes=test_episodes,
        product_embeddings=product_embeddings,
        candidate_pairs=candidate_pairs,
        num_context_pairs=args.num_context_pairs,
        threshold_ratio=args.threshold_ratio,
        device=device,
        scoring_mode=args.scoring_mode,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )
    results["config"] = {
        "base_results_dir": str(args.base_results_dir),
        "num_context_pairs": args.num_context_pairs,
        "max_candidate_pairs": args.max_candidate_pairs,
        "seed": args.seed,
        "scoring_mode": args.scoring_mode,
        "selection_strategy": args.selection_strategy,
        "active_objective": "greedy expected posterior entropy minimization over labels y in {0,1}",
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Active-query test metrics:")
    for key, value in results["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved active-query results to {args.output_path}")


if __name__ == "__main__":
    main()
