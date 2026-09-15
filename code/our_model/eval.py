import torch
import numpy as np
import pandas as pd
import json
import math
from tqdm import tqdm

from transformers import AutoTokenizer

# Import your custom modules
from model_belief import BeliefEncoder
from reward_module import total_reward, extrinsic_reward, gaussian_entropy
from ppo_training import LLMActor, ValueNetwork, answerable_attributes, attr_to_latent_map, LATENT_KEYS
from model_environment import ModelEnvironment


# ==========================================
# NEW VECTORIZED UTILITY & REGRET FUNCTIONS
# ==========================================

def compute_all_utilities(user_vec, product_mat):
    """
    Computes -(distance^2) for all products simultaneously using matrix math.
    """
    # a^2
    prod_norms = np.sum(product_mat**2, axis=1)
    # b^2
    user_norm = np.sum(user_vec**2)
    # -2ab (The dot product!)
    dot_part = -2 * (product_mat @ user_vec)
    
    return -1.0 * (prod_norms + user_norm + dot_part)

def get_oracle_best_product(user_latent_vector, product_latent_matrix):
    utilities = compute_all_utilities(user_latent_vector, product_latent_matrix)
    best_index = int(np.argmax(utilities))
    best_utility = float(utilities[best_index])
    return best_index, best_utility

def get_oracle_worst_product(user_latent_vector, product_latent_matrix):
    utilities = compute_all_utilities(user_latent_vector, product_latent_matrix)
    worst_index = int(np.argmin(utilities))
    worst_utility = float(utilities[worst_index])
    return worst_index, worst_utility

def compute_regret(oracle_utility, agent_utility):
    """ Regret = U(oracle_best) - U(agent_choice) """
    return float(oracle_utility - agent_utility)


# ==========================================
# PERCENTILE EVALUATION METRICS
# ==========================================

def evaluate_recommendations(mu_predicted, user_pref, product_mat):
    """
    Evaluates Regret and Top-K Success across a deep list of thresholds.
    """
    user_vec = np.array([user_pref[k] for k in LATENT_KEYS], dtype=np.float32)
    mu_vec = mu_predicted.squeeze().detach().cpu().numpy()
    
    # 1. Oracle truth
    true_utilities = compute_all_utilities(user_vec, product_mat)
    oracle_best_idx = int(np.argmax(true_utilities))
    oracle_best_utility = float(true_utilities[oracle_best_idx])
    
    # 2. Agent prediction
    agent_predictions = compute_all_utilities(mu_vec, product_mat)
    agent_best_idx = int(np.argmax(agent_predictions))
    agent_achieved_true_utility = float(true_utilities[agent_best_idx])
    
    # 3. Regret
    regret = compute_regret(oracle_best_utility, agent_achieved_true_utility)
    _, oracle_worst_utility = get_oracle_worst_product(user_vec, product_mat)
    utility_range = max(1e-5, oracle_best_utility - oracle_worst_utility)
    normalized_regret = regret / utility_range
    
    # 4. Success @ K for the exact requested depths
    k_list = [1, 3, 5, 10, 50, 100, 200, 500, 1000]
    success_dict = {}
    
    predicted_ranking_indices = np.argsort(agent_predictions)[::-1]
    
    for k in k_list:
        success_dict[f"success_at_{k}"] = 1 if oracle_best_idx in predicted_ranking_indices[:k] else 0
        
    return success_dict, regret, normalized_regret


# ==========================================
# EVALUATION LOOP
# ==========================================

def evaluate_model(test_path, belief_path, value_path, actor_path, product_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # 1. Load Data
    test_users = pd.read_pickle(test_path)
    with open(product_path, "r") as f:
        product_embeddings = json.load(f)

    # Convert product_embeddings to a NumPy matrix for fast math
    asins = list(product_embeddings.keys())
    product_mat = np.array(
        [[product_embeddings[asin][k] for k in LATENT_KEYS] for asin in asins], 
        dtype=np.float32
    )

    # 2. Initialize Models
    LATENT_DIM = 8
    tokenizer = AutoTokenizer.from_pretrained("./local_minilm")
    
    belief_model = BeliefEncoder().to(device)
    actor = LLMActor(latent_keys=LATENT_KEYS).to(device)
    value_net = ValueNetwork(latent_dim=LATENT_DIM, questions_dim=len(answerable_attributes)).to(device)

    # Load Checkpoints
    belief_model.load_state_dict(torch.load(belief_path, map_location=device))
    actor.policy_net.load_state_dict(torch.load(actor_path, map_location=device))
    value_net.load_state_dict(torch.load(value_path, map_location=device))

    # Set to Eval mode
    belief_model.eval()
    actor.eval()
    value_net.eval()

    env = ModelEnvironment(
        belief_model=belief_model,
        vocab_size=tokenizer.vocab_size,
        product_embeddings=product_embeddings,
        tokenizer=tokenizer,
        device=device,
        latent_dim=LATENT_DIM,
        max_turns=10
    )

    # 3. Metrics Trackers
    metrics = {
        "avg_total_reward": [],
        "avg_intrinsic_reward": [],
        "avg_extrinsic_reward": [],
        "avg_entropy_reduction": [],
        "avg_belief_accuracy": [],
        "avg_regret": [],
        "avg_normalized_regret": [],
        "success_at_1": [],
        "success_at_3": [],
        "success_at_5": [],
        "success_at_10": [],
        "success_at_50": [],
        "success_at_100": [],
        "success_at_200": [],
        "success_at_500": [],
        "success_at_1000": [],
        "threshold_success_rate": [],
        "avg_dialogue_turns_to_success": [],
        "avg_uncertainty_reduction_per_turn": [],
        "success_rate_at_turn_5": []
    }

    MSE_SUCCESS_THRESHOLD = 0.3

    # 4. Evaluation Loop
    with torch.no_grad():
        for user in tqdm(test_users, desc="Evaluating Users"):
            user.reset_convo()
            env.user = user
            env.user_pref = user.latent_variables
            
            obs = env.reset()
            current_mask = torch.zeros(1, actor.num_attrs, device=device)
            
            ep_intrinsic = 0.0
            turn0_entropy = float(gaussian_entropy(obs["std"]).sum())
            
            done = False
            turn = 0
            
            turn_5_mse = None
            turn_success = 10 # Default to max turns
            
            while not done:
                mu, std = obs["mu"], obs["std"]
                
                # Actor picks question (greedy for eval)
                logits = actor(mu, std, action_mask=current_mask)
                action = torch.argmax(logits) 
                
                target_attr = actor.answerable_attributes[action.item()]
                question = f"Could you tell me more about your {target_attr}?"
                
                current_mask[0, action.item()] = 1
                
                obs, reward, done, info = env.step(question, alpha=0.5, gamma=0.99)
                
                if "intrinsic" in info:
                    ep_intrinsic += float(info["intrinsic"])
                    
                turn += 1
                
                # Check MSE halfway through
                true_latent = torch.tensor([user.latent_variables[k] for k in LATENT_KEYS], device=device)
                current_mse = torch.mean((obs["mu"].squeeze() - true_latent) ** 2).item()
                
                if turn == 5:
                    turn_5_mse = current_mse
                    
                if current_mse < MSE_SUCCESS_THRESHOLD and turn_success == 10:
                    turn_success = turn 
                    
            # Episode finished
            final_entropy = float(gaussian_entropy(obs["std"]).sum())
            entropy_reduction = turn0_entropy - final_entropy
            
            # --- NEW PERCENTILE RECOMMENDATION LOGIC ---
            success_dict, regret, norm_regret = evaluate_recommendations(
                obs["mu"], user.latent_variables, product_mat
            )
            
            final_mse = torch.mean((obs["mu"].squeeze() - true_latent) ** 2).item()
            
            # Extract final step rewards from env
            if isinstance(info, dict) and "total" in info:
                metrics["avg_total_reward"].append(float(info["total"]))
                metrics["avg_extrinsic_reward"].append(float(info["extrinsic"]))
            else:
                metrics["avg_total_reward"].append(float(reward))
            
            metrics["avg_intrinsic_reward"].append(ep_intrinsic)
            metrics["avg_entropy_reduction"].append(entropy_reduction)
            metrics["avg_belief_accuracy"].append(1 if final_mse < MSE_SUCCESS_THRESHOLD else 0)
            metrics["avg_regret"].append(regret)
            metrics["avg_normalized_regret"].append(norm_regret)
            metrics["avg_dialogue_turns_to_success"].append(turn_success)
            metrics["avg_uncertainty_reduction_per_turn"].append(entropy_reduction / turn)
            metrics["threshold_success_rate"].append(1 if final_mse < MSE_SUCCESS_THRESHOLD else 0)
            
            if turn_5_mse is not None:
                metrics["success_rate_at_turn_5"].append(1 if turn_5_mse < MSE_SUCCESS_THRESHOLD else 0)
                
            for k, v in success_dict.items():
                metrics[k].append(v)

    # 5. Aggregate Results
    final_results = {
        "test_users_path": test_path,
        "belief_model_path": belief_path,
        "value_net_path": value_path,
        "actor_path": actor_path,
    }
    
    for key, val_list in metrics.items():
        if len(val_list) > 0:
            final_results[key] = float(np.mean(val_list))
        else:
            final_results[key] = 0.0
            
    print("\n--- Evaluation Results ---")
    print(json.dumps(final_results, indent=2))
    
    with open("eval_results.json", "w") as f:
        json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    evaluate_model(
        test_path="data/user_test_set.pkl",
        belief_path="checkpoints/belief_model_checkpoint.pth",
        value_path="checkpoints/value_net_checkpoint.pth",
        actor_path="checkpoints/actor_policy_checkpoint.pth",
        product_path="data/product_embeddings.json"
    )