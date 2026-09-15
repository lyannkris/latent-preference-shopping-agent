import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer

from curio_belief_model import CurioBeliefModel
from curio_environment import CurioEnvironment
from evaluation_functions import (
    compute_all_utilities,
    compute_regret,
    compute_utility,
    get_best_feasible_product,
    get_oracle_best_product,
    get_oracle_worst_product,
)
from generate_users import Simulated_User
from ppo_training import CategoricalActor, ValueNetwork
from product_scoring import load_product_embeddings, load_user_types


LATENT_KEYS = [
    "comfort", "functionality", "maintenance", "status_symbol",
    "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"
]


def dict_to_vec(d):
    return np.array([d[k] for k in LATENT_KEYS], dtype=float)

def compute_entropy(belief):
    return -np.sum(belief * np.log(belief + 1e-12))

def load_fixed_users(path):
    if not os.path.exists(path):
        print(f"User file not found at {path}; using random users.")
        return None
    with open(path, "rb") as f:
        users = pickle.load(f)
    print(f"Loaded {len(users)} test users from {path}")
    return users

def get_true_type(user, user_types):
    user_vec = dict_to_vec(user.latent_variables)
    type_dist = []
    for _, traits in user_types.items():
        type_vec = dict_to_vec(traits)
        type_dist.append(np.linalg.norm(user_vec - type_vec))
    return int(np.argmin(type_dist))

def load_models(
    num_types,
    belief_model_path,
    value_net_path,
    actor_path,
):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    belief_model = CurioBeliefModel(
        num_types=num_types,
        model_name=model_name,
    )
    belief_model.load_state_dict(torch.load(belief_model_path), strict=False)
    belief_model.to(device).eval()

    value_net = ValueNetwork()
    value_net.load_state_dict(torch.load(value_net_path))
    value_net.to(device).eval()

    actor = CategoricalActor(input_dim=8)
    actor.load_state_dict(torch.load(actor_path))
    actor.to(device).eval()

    return tokenizer, belief_model, value_net, actor, device

def test_curio(
    num_episodes=200,
    max_turns=8,
    seed=42,
    test_users_path="data/user_test_set.pkl",
    belief_model_path="checkpoints/belief_model_ep800.pth",
    value_net_path="checkpoints/value_net_ep800.pth",
    actor_path="checkpoints/actor_ep800.pth",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setup data
    user_types = load_user_types("data/user_types.json")
    prod_embs = load_product_embeddings("data/product_embeddings.json")
    prod_ids = list(prod_embs.keys())
    prod_matrix = np.array([dict_to_vec(prod_embs[p]) for p in prod_ids])

    max_turns = min(max_turns, len(LATENT_KEYS))

    tokenizer, belief_model, value_net, actor, device = load_models(
        num_types=len(user_types),
        belief_model_path=belief_model_path,
        value_net_path=value_net_path,
        actor_path=actor_path,
    )

    fixed_test_users = load_fixed_users(test_users_path)
    total_eval_episodes = min(num_episodes, len(fixed_test_users)) if fixed_test_users else num_episodes

    # metric storage
    stats = {
        "rewards": [], "accuracy": [], "extrinsic": [], "intrinsic": [],  "regret": [], "norm_regret": [],
        "oracle_util": [], "agent_util": [], "ig_per_turn": [],
        "turns": []
    }

    k_vals = [1, 3, 5, 10, 50, 100, 200, 500, 1000]
    success_at_k = {k: [] for k in k_vals}

    threshold_success_list = []
    uncertainty_reduction_per_turn = []

    for ep in range(total_eval_episodes):
        if fixed_test_users:
            user = fixed_test_users[ep]
            user.reset_convo()
        else:
            user = Simulated_User("test")

        true_type_idx = get_true_type(user, user_types)
        env = CurioEnvironment(
            user=user,
            user_types=user_types,
            product_map=prod_embs,
            belief_model=belief_model,
            tokenizer=tokenizer,
            max_turns=max_turns,
            alpha=1.0,
            gamma=1.0,
        )

        obs = env.reset(new_user=False)
        done, ep_intrinsic, ep_extrinsic, first_success_turn = False, 0, 0, None
        action_mask = torch.ones(len(LATENT_KEYS), device=device)

        curr_belief = obs["belief"]
        init_entropy = compute_entropy(curr_belief)
        h_prev = init_entropy

        episode_reductions = []
        turn_counter = 0
        

        while not done:
            ids, mask = env._encode_history()

            with torch.no_grad():
                _, type_probs, _ = belief_model(ids.to(device), mask.to(device))

            agent_message, action_idx, _ = actor.generate_question(type_probs.float(), mask=action_mask)
            action_mask[action_idx] = 0.0

            obs, reward, done, info = env.step(agent_message)
            turn_counter += 1

            # belief stats
            curr_belief = obs["belief"]
            h_curr = compute_entropy(curr_belief)
            stats["ig_per_turn"].append(h_prev - h_curr)
            h_prev = h_curr

            if int(np.argmax(curr_belief)) == true_type_idx and first_success_turn is None:
                if np.max(curr_belief) > 0.5:
                    first_success_turn = turn_counter

            if "intrinsic" in info:
                ep_intrinsic += info["intrinsic"]
            if "extrinsic" in info:
                ep_extrinsic += info["extrinsic"]

        # post-episode metrics
        uncertainty_reduction_per_turn.extend(episode_reductions)
        stats["turns"].append(first_success_turn if first_success_turn is not None else max_turns)

        asin = obs.get("recommendation", None)
        if asin is None:
            continue
        agent_choice = prod_ids.index(asin)
        
        user_latent_vec = dict_to_vec(user.latent_variables)
        best_idx, oracle_best_util = get_oracle_best_product(
            user_latent_vec,
            prod_matrix,
        )
        _, oracle_worst_util = get_oracle_worst_product(
            user_latent_vec,
            prod_matrix,
        )

        feasible_indices = info.get("feasible_indices", [])
        if len(feasible_indices) > 0:
            get_best_feasible_product(user_latent_vec, prod_matrix, feasible_indices)

        agent_util = compute_utility(prod_matrix[agent_choice], user_latent_vec)
        stats["agent_util"].append(agent_util)
        regret = compute_regret(oracle_best_util, agent_util)
        norm_regret = regret / (oracle_best_util - oracle_worst_util + 1e-8)

        stats["rewards"].append(ep_intrinsic + ep_extrinsic)
        stats["intrinsic"].append(ep_intrinsic)
        stats["extrinsic"].append(ep_extrinsic)
        stats["accuracy"].append(int(np.argmax(curr_belief)) == true_type_idx)
        stats["oracle_util"].append(oracle_best_util)
        stats["regret"].append(regret)
        stats["norm_regret"].append(norm_regret)

        # success@K
        all_utils = compute_all_utilities(user_latent_vec, prod_matrix)
        ranks = np.argsort(all_utils)[::-1]
        for k in k_vals:
            success_at_k[k].append(int(agent_choice in ranks[:k]))

        u_threshold = np.percentile(all_utils, 70)
        threshold_success_list.append(int(agent_util >= u_threshold))
        

        if ep % 10 == 0:
            print(f"[Test] Episode {ep}: total={stats['rewards'][-1]:.3f}")

    # summary and plots
    results = {
        "test_users_path": test_users_path,
        "belief_model_path": belief_model_path,
        "value_net_path": value_net_path,
        "actor_path": actor_path,
        "avg_total_reward": float(np.mean(stats["rewards"])),
        "avg_intrinsic_reward": float(np.mean(stats["intrinsic"])),
        "avg_extrinsic_reward": float(np.mean(stats["extrinsic"])),
        "avg_accuracy": float(np.mean(stats["accuracy"])),
        "avg_regret": float(np.mean(stats["regret"])),
        "avg_norm_regret": float(np.mean(stats["norm_regret"])),
        "threshold_success_rate": float(np.mean(threshold_success_list)),
        "avg_turns_to_success": float(np.mean(stats["turns"])),
        "avg_uncertainty_reduction_per_turn": float(np.mean(uncertainty_reduction_per_turn)),
        "success_rate_at_turn_5": float(np.mean([1 if t <= 5 else 0 for t in stats["turns"]])),
    }
    
    for k in k_vals:
        results[f"success@{k}"] = float(np.mean(success_at_k[k]))

    os.makedirs("results", exist_ok=True)
    with open("results/curio_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 5))
    sns.histplot(stats["regret"], kde=True, bins=20, color="royalblue")
    plt.title("Regret Distribution Across Test Users")
    plt.xlabel("Regret")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/curio_regret_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[stats["regret"], stats["norm_regret"]], palette="Set2")
    plt.xticks([0, 1], ["Regret", "Normalized Regret"])
    plt.title("Regret and Normalized Regret")
    plt.tight_layout()
    plt.savefig("results/curio_regret_boxplots.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=[f"@{k}" for k in k_vals], y=[results[f"success@{k}"] for k in k_vals],
        palette="viridis",
    )
    plt.ylim(0, 1)
    plt.title("Success@K")
    plt.tight_layout()
    plt.savefig("results/curio_success_at_k.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats["accuracy"], marker="o", alpha=0.6)
    plt.title("Belief Accuracy Across Test Users")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("results/curio_belief_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats["oracle_util"], label="Oracle Utility", linestyle="--")
    plt.plot(stats["agent_util"], label="Agent Utility", alpha=0.7)
    plt.title("Oracle vs Agent Utility")
    plt.xlabel("Episode")
    plt.ylabel("Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/curio_utility_comparison.png")
    plt.close()

    print("Saved test plots to results/")
    print("\n=== Testing Complete ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import time
    from datetime import timedelta

    print("=== Testing Start ===")
    start_eval = time.time()
    test_curio(num_episodes=200)
    end_eval = time.time()
    print(f"Total time taken for testing: {timedelta(seconds=int(end_eval - start_eval))}")
