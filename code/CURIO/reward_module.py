import numpy as np

LATENT_KEYS = [
    "comfort",
    "functionality",
    "maintenance",
    "status_symbol",
    "modesty",
    "sustainability",
    "risk_tolerance",
    "trend_sensitivity",
]


def dict_to_vec(d):
      result = []
      for k in LATENT_KEYS:
          result.append(float(d[k]))

      return np.array(result, dtype=np.float32)


def utility(product_vec: np.ndarray, user_vec: np.ndarray):
    """U(p, u) = Squared Euclidean distance in R^8 (smaller is better)."""
    
    # 1. If product_vec is a PyTorch tensor, convert it to a NumPy array
    if hasattr(product_vec, "detach"):
        product_vec = product_vec.detach().cpu().numpy()
        
    # 2. If user_vec (mu) is a PyTorch tensor, convert it to a NumPy array
    if hasattr(user_vec, "detach"):
        user_vec = user_vec.detach().cpu().numpy()
        
    # 3. Ensure they are the correct shape
    product_vec = np.squeeze(np.asarray(product_vec, dtype=np.float32))
    user_vec = np.squeeze(np.asarray(user_vec, dtype=np.float32))

    # 4. Calculate Euclidean distance
    return -1 * float(np.sum((product_vec - user_vec)**2))


def extrinsic_reward( chosen_a, product_embeddings, true_user_latent):

    p = dict_to_vec(product_embeddings[chosen_a])
    u = dict_to_vec(true_user_latent)
    ut = utility(p, u)
    return ut




def entropy(prob):
    p = np.clip(prob, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def intrinsic_reward(belief_before, belief_after, gamma: float = 1.0):
    H_before = entropy(belief_before)
    H_after = entropy(belief_after)
    return H_before - gamma * H_after


def total_reward(chosen_a, product_emb,true_user_latent, belief_before,
    belief_after, alpha: float = 1.0, gamma: float = 1.0):

    ext = extrinsic_reward(chosen_a, product_emb, true_user_latent)
    int = intrinsic_reward(belief_before, belief_after, gamma)
    total = ext + alpha * int

    return {
        "extrinsic": ext,
        "intrinsic": int,
        "total": total,
    }

