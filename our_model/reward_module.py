import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


LATENT_KEYS = ["comfort", "functionality", "maintenance", "status_symbol",
                "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"]

# -----------------------------
# UTILITY HELPERS
# -----------------------------

def dict_to_vec(d: Dict[str, float]) -> np.ndarray:
    return np.array([d[k] for k in LATENT_KEYS], dtype=np.float32)


def utility(product_vec, user_vec) -> float:
    # 1. Convert to PyTorch Tensors if they aren't already
    if not isinstance(product_vec, torch.Tensor):
        product_vec = torch.as_tensor(product_vec, dtype=torch.float32)
    if not isinstance(user_vec, torch.Tensor):
        user_vec = torch.as_tensor(user_vec, dtype=torch.float32)

    # 2. Move the product vector to the SAME device as the user vector (mu)
    product_vec = product_vec.to(user_vec.device)

    # 3. Calculate Squared Euclidean Distance
    sq_dist = torch.sum(torch.pow(product_vec - user_vec, 2))

    # 4. The Hyper-Tuned Exponential Spike
    # Dist 0.0 -> Reward = 100.0
    # Dist 0.2 -> Reward =  45.0
    # Dist 0.5 -> Reward =   6.0
    # Dist 1.0 -> Reward = -14.0
    shaped_reward = (120.0 * torch.exp(-3.0 * sq_dist)) - 20.0
    
    return float(torch.clamp(shaped_reward, min=-50.0, max=100.0))


# -----------------------------
# REWARD MODEL (model predicted reward)
# -----------------------------

class RewardModel(nn.Module):
    def __init__(self, item_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(item_dim + latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, item_features: torch.Tensor, z: torch.Tensor):
        x = torch.cat([item_features, z], dim=-1)
        h = F.relu(self.fc(x))
        return self.out(h).squeeze(-1)
    

# Gaussian entropy for intrinsic reward
def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log(2 * np.pi * np.e * std**2)

# --- Intrinsic_reward_gaussian ---
def intrinsic_reward_gaussian(std_before: torch.Tensor, std_after: torch.Tensor) -> torch.Tensor:
    H_before = gaussian_entropy(std_before)
    H_after = gaussian_entropy(std_after)
    
    # Calculate info gain across ALL 8 dimensions
    info_gain = H_before - H_after 
    
    # Sum the true per-dimension info gain
    raw_reward = info_gain.sum()

    # Scale and clip to encourage curiosity
    scaled_reward = raw_reward * 5.0 
    clipped_reward = torch.clamp(scaled_reward, min=-1.0, max=20.0)
    
    return clipped_reward


def find_max_utility_in_catalog(item_features: dict, user_pref: dict) -> float:
    """
    Finds the theoretical maximum utility possible given the current catalog.
    """
    u_vec = dict_to_vec(user_pref)
    u_vec_tensor = torch.as_tensor(u_vec, dtype=torch.float32)
    
    max_u = -50.0 # Absolute minimum clamp from utility()
    
    # Iterate over the catalog to find the best matching item
    for asin, features in item_features.items():
        p_vec = dict_to_vec(features)
        current_u = utility(p_vec, u_vec_tensor)
        if current_u > max_u:
            max_u = current_u
            
    return max_u


# --- Extrinsic_reward ---
def extrinsic_reward(chosen_asin, item_features, user_pref):
    """
    Calculates the normalized reward based on the best available item in the catalog.
    """
    # 1. Calculate how good the agent's choice was
    p_vec = dict_to_vec(item_features[chosen_asin])
    u_vec = dict_to_vec(user_pref)
    agent_utility = utility(p_vec, u_vec)
    
    # 2. Calculate the best possible choice available in the inventory
    max_possible_utility = find_max_utility_in_catalog(item_features, user_pref)
    
    # 3. Normalize (Shift by +50 to safely handle the negative utility bound)
    # utility() ranges from -50 to 100. Shifting by +50 maps it to 0 to 150.
    shifted_agent = agent_utility + 50.0
    shifted_max = max_possible_utility + 50.0
    
    if shifted_max > 0:
        # Ratio of how close the agent got to perfection (0.0 to 1.0)
        ratio = shifted_agent / shifted_max
        
        # Scale back down to the -50 to 100 range.
        # If they pick the best available item -> ratio is 1.0 -> reward is 100.0
        normalized_reward = (ratio * 150.0) - 50.0
    else:
        # Fallback if all items are perfectly terrible
        normalized_reward = -50.0
        
    return normalized_reward

# --- Total_reward ---
def total_reward(chosen_asin,
                 item_features: dict, # Updated type hint from torch.Tensor to dict
                 user_pref: torch.Tensor,
                 std_before: torch.Tensor,
                 std_after: torch.Tensor,
                 alpha: float = 1.0,
                 gamma: float = 1.0) -> dict:

    # Extrinsic reward now automatically normalizes against the catalog
    r_ext = extrinsic_reward(chosen_asin, item_features, user_pref)
    r_int = intrinsic_reward_gaussian(std_before, std_after) 

    # Calculate the raw total
    raw_total = r_ext + alpha * float(r_int)
    
    # Return raw_total directly, removing the np.clip limitation
    return {
        "extrinsic": r_ext, 
        "intrinsic_raw": r_int, 
        "intrinsic_scaled": alpha * float(r_int),
        "total": float(raw_total)
    }