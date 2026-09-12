from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from data import PaperVPLEpisode, compute_utility, dict_to_vec
from model import PaperStyleVPLRanker


def evaluate_vpl_model(
    model: PaperStyleVPLRanker,
    episodes: List[PaperVPLEpisode],
    product_embeddings: Dict[str, Dict[str, float]],
    device: str = "cpu",
    threshold_ratio: float = 0.95,
    scoring_mode: str = "latent_utility",
) -> Dict[str, object]:
    model.eval()

    utilities = []
    regrets = []
    normalized_regrets = []
    threshold_success_list = []
    success_at_1_list = []
    success_at_3_list = []
    success_at_5_list = []
    success_at_10_list = []
    success_at_50_list = []
    success_at_100_list = []
    success_at_200_list = []
    success_at_500_list = []
    success_at_1000_list = []
    context_pair_counts = []
    per_episode = []

    for episode in episodes:
        pair_features = []
        for pair in episode.context_pairs:
            a_vec = dict_to_vec(product_embeddings[pair.item_a])
            b_vec = dict_to_vec(product_embeddings[pair.item_b])
            pair_features.append(np.concatenate([a_vec, b_vec, np.array([pair.label], dtype=np.float32)]))
        pair_tensor = torch.tensor(np.stack(pair_features), dtype=torch.float32, device=device).unsqueeze(0)

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
        inferred_latent_np = model.semantic_latent(posterior_mu).squeeze(0).detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

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

        success_at_1 = int(pred_idx in top1)
        success_at_3 = int(pred_idx in top3)
        success_at_5 = int(pred_idx in top5)
        success_at_10 = int(pred_idx in top10)
        success_at_50 = int(pred_idx in top50)
        success_at_100 = int(pred_idx in top100)
        success_at_200 = int(pred_idx in top200)
        success_at_500 = int(pred_idx in top500)
        success_at_1000 = int(pred_idx in top1000)

        utilities.append(chosen_utility)
        regrets.append(regret)
        normalized_regrets.append(normalized_regret)
        threshold_success_list.append(threshold_success)
        success_at_1_list.append(success_at_1)
        success_at_3_list.append(success_at_3)
        success_at_5_list.append(success_at_5)
        success_at_10_list.append(success_at_10)
        success_at_50_list.append(success_at_50)
        success_at_100_list.append(success_at_100)
        success_at_200_list.append(success_at_200)
        success_at_500_list.append(success_at_500)
        success_at_1000_list.append(success_at_1000)
        context_pair_counts.append(len(episode.context_pairs))

        per_episode.append(
            {
                "episode_id": episode.episode_id,
                "user_id": episode.user_id,
                "num_context_pairs": len(episode.context_pairs),
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
                "success_at_1": success_at_1,
                "success_at_3": success_at_3,
                "success_at_5": success_at_5,
                "success_at_10": success_at_10,
                "success_at_50": success_at_50,
                "success_at_100": success_at_100,
                "success_at_200": success_at_200,
                "success_at_500": success_at_500,
                "success_at_1000": success_at_1000,
                "inferred_latent": inferred_latent_np.tolist(),
            }
        )

    return {
        "metrics": {
            "avg_utility": float(np.mean(utilities)),
            "avg_regret": float(np.mean(regrets)),
            "avg_normalized_regret": float(np.mean(normalized_regrets)),
            "threshold_success_rate": float(np.mean(threshold_success_list)),
            "success_at_1": float(np.mean(success_at_1_list)),
            "success_at_3": float(np.mean(success_at_3_list)),
            "success_at_5": float(np.mean(success_at_5_list)),
            "success_at_10": float(np.mean(success_at_10_list)),
            "success_at_50": float(np.mean(success_at_50_list)),
            "success_at_100": float(np.mean(success_at_100_list)),
            "success_at_200": float(np.mean(success_at_200_list)),
            "success_at_500": float(np.mean(success_at_500_list)),
            "success_at_1000": float(np.mean(success_at_1000_list)),
            "avg_context_pairs": float(np.mean(context_pair_counts)),
            "threshold_definition": "recommended utility >= 70th percentile of episode utilities",
            "normalized_regret_definition": "regret / (best utility - worst utility)",
        },
        "per_episode": per_episode,
    }
