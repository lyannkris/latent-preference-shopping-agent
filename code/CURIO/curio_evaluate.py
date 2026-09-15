import torch
import numpy as np
import json
import os
import random
import pickle

from curio_environment import CurioEnvironment
from curio_belief_model import CurioBeliefModel
from product_scoring import load_product_embeddings, load_user_types
from ppo_training import CategoricalActor

from transformers import AutoTokenizer
from ppo_training import ValueNetwork   
from generate_users import Simulated_User

from evaluation_functions import (
    compute_all_utilities,
    compute_utility,
    get_oracle_best_product,
    get_best_feasible_product,
    compute_regret,
    get_oracle_worst_product
)


LATENT_KEYS = [
    "comfort", "functionality", "maintenance", "status_symbol",
    "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"
]

def dict_to_vec(d):
    return np.array([d[k] for k in LATENT_KEYS], dtype=float)

def compute_entropy(belief):
    return -np.sum(belief * np.log(belief + 1e-12))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def load_fixed_users(path: str):
    if not os.path.exists(path):
        print(f"User file not found at {path}; using random users.")
        return None
    with open(path, "rb") as f:
        users = pickle.load(f)
    print(f"Loaded {len(users)} users from {path}")
    return users

def evaluate_curio(
    num_episodes=200,
    max_turns=8,
    seed=42,
    belief_model_path="final_model/belief_model_final.pth",
    value_net_path="final_model/value_net_final.pth",
    actor_path="final_model/actor_final.pth",
    eval_users_path="data/user_eval_set.pkl",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # setup data and weight
    user_types = load_user_types("data/user_types.json")
    prod_emb = load_product_embeddings("data/product_embeddings.json")
    prod_ids = list(prod_emb.keys())
    prod_matrix = np.array([dict_to_vec(prod_emb[p]) for p in prod_emb])
    
    max_turns = min(max_turns, len(LATENT_KEYS))

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # initialize models
    belief_model = CurioBeliefModel(
        num_types=len(user_types),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ).to(device)
    belief_model.load_state_dict(torch.load(belief_model_path), strict=False)
    belief_model.eval()

    value_net = ValueNetwork()
    value_net.load_state_dict(torch.load(value_net_path))
    value_net.eval()

    # actor conf
    with open("final_model/config.json") as f:
        actor_cfg = json.load(f)
    
    actor = CategoricalActor(input_dim=8).to(device)
    actor.load_state_dict(torch.load(actor_path))
    actor.eval()

    
    # metric accumulators
    stats = {
        "rewards": [], "intrinsic": [], "extrinsic": [],
        "accuracy": [], "regret": [], "norm_regret": [], 
        "success_turns": [], "ig_per_turn": []
    }


    # Success@K tracking
    k_vals = [1, 3, 5, 10, 50, 100, 200, 500, 1000]
    success_at_k = {k: [] for k in k_vals}

    threshold_success_list = []
    fixed_eval_users = load_fixed_users(eval_users_path)

    # loop
    for ep in range(num_episodes):
        if fixed_eval_users:
            user = fixed_eval_users[ep % len(fixed_eval_users)]
            user.reset_convo()
        else:
            user = Simulated_User()

        # determine target type for this user
        user_vec = dict_to_vec(user.latent_variables) 
        dists = []

        # 2. Compare against each of the 8 standard types
        for t_id, t_traits in user_types.items():
            t_vec = dict_to_vec(t_traits)
            dist = np.linalg.norm(user_vec - t_vec)
            dists.append(dist)

        target_idx = np.argmin(dists)

        env = CurioEnvironment(
            user=user,
            user_types=user_types,
            product_map=prod_emb,
            belief_model=belief_model,
            tokenizer=tokenizer,
            max_turns=max_turns,
            alpha=1.0, 
            gamma=1.0,
        )

        obs = env.reset(new_user=False)
        done = False
        curr_belief = obs["belief"]

        ep_intrinsic = 0
        ep_extrinsic = 0

        # h_prev is initialised from the uniform reset belief 
        initial_entropy = compute_entropy(curr_belief)
        h_prev = initial_entropy

        first_success_turn = None
        turn_counter = 0
        action_mask = torch.ones(len(LATENT_KEYS), device=device)

        while not done:
            ids, mask = env._encode_history()

            # forward pass
            with torch.no_grad(): 
                _, type_probs, _ = belief_model(ids.to(device), mask.to(device))
            
            # pass action_mask so actor doesn't ask the same question twice per episode
            agent_message, action_idx, _ = actor.generate_question(type_probs.float(), mask=action_mask)
            action_mask[action_idx] = 0.0  # mark this attribute as used

            obs, reward, done, info = env.step(agent_message)
            turn_counter += 1

            # belief stats
            curr_belief = obs["belief"]
            h_curr = compute_entropy(curr_belief)
            stats["ig_per_turn"].append(h_prev - h_curr)
            h_prev = h_curr

            # success tracking
            if np.argmax(curr_belief) == target_idx and first_success_turn is None:
                # add confidence threshold to avoid lucky guesses
                if np.max(curr_belief) > 0.5:
                    first_success_turn = turn_counter

            # Track intrinsic reward
            if "intrinsic" in info:
                ep_intrinsic += info["intrinsic"]

            # Track extrinsic reward (only at final step)
            if "extrinsic" in info:
                ep_extrinsic += info["extrinsic"]

        # final post-episode metrics
        stats["success_turns"].append(first_success_turn if first_success_turn else max_turns)
        stats["intrinsic"].append(ep_intrinsic)
        stats["extrinsic"].append(ep_extrinsic)
        stats["rewards"].append(ep_intrinsic + ep_extrinsic)    
        stats["accuracy"].append(int(np.argmax(curr_belief) == target_idx))

        user_vec = dict_to_vec(user.latent_variables)
        best_idx, oracle_best_util = get_oracle_best_product(user_vec, prod_matrix)
        _, oracle_worst_util = get_oracle_worst_product(user_vec, prod_matrix)

        asin = obs.get("recommendation", None)
        if asin is None:
            continue
        agent_choice = prod_ids.index(asin)
        agent_util = compute_utility(prod_matrix[agent_choice], user_vec)

        regret = compute_regret(oracle_best_util, agent_util)
        norm_regret = regret / (oracle_best_util - oracle_worst_util + 1e-8)
        stats["regret"].append(regret)
        stats["norm_regret"].append(norm_regret)

        feasible_indices = info.get("feasible_indices", [])
        if len(feasible_indices) > 0:
            get_best_feasible_product(user_vec, prod_matrix, feasible_indices)

        # multi-k success
        all_utils = compute_all_utilities(user_vec, prod_matrix)
        ranks = np.argsort(all_utils)[::-1] # High to low
        for k in k_vals:
            success_at_k[k].append(int(agent_choice in ranks[:k]))

        # threshold success (utility ≥ 0.7)
        # calculate what the 70th percentile "Utility Score" is for this episode
        u_threshold = np.percentile(all_utils, 70)
        threshold_success = int(agent_util >= u_threshold)
        threshold_success_list.append(threshold_success)

        if ep % 10 == 0:
            print(f"[Eval] Episode {ep}: total={stats['rewards'][-1]:.3f}")

    # results
    results = {
        "belief_model_path": belief_model_path,
        "value_net_path": value_net_path,
        "actor_path": actor_path,
        "eval_users_path": eval_users_path,
        "avg_total_reward": float(np.mean(stats["rewards"])),
        "avg_intrinsic_reward": float(np.mean(stats["intrinsic"])),
        "avg_extrinsic_reward": float(np.mean(stats["extrinsic"])),
        "avg_accuracy": float(np.mean(stats["accuracy"])),
        "avg_regret": float(np.mean(stats["regret"])),
        "avg_norm_regret": float(np.mean(stats["norm_regret"])),
        "threshold_success_rate": float(np.mean(threshold_success_list)),
        "avg_turns_to_success": float(np.mean(stats["success_turns"])),
        "avg_ig_per_turn": float(np.mean(stats["ig_per_turn"])),
        "success_rate_at_turn_5": float(np.mean([1 if t <= 5 else 0 for t in stats["success_turns"]]))
    }

    for k, vals in success_at_k.items():
        results[f"success@{k}"] = float(np.mean(vals))

    if not os.path.exists("results"):   
        os.makedirs("results")

    with open("results/curio_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import time
    from datetime import timedelta

    print("=== Evaluation Start ===")
    start_eval = time.time() # Start clock

    evaluate_curio(num_episodes=200)

    end_eval = time.time()
    total_time = str(timedelta(seconds=int(end_eval - start_eval)))
    
    print(f"Total time taken for evaluating: {total_time}")
