import json
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

def load_product_embeddings(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_user_types(path: str):
    with open(path, "r") as f:
        return json.load(f)

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



def dict_to_vec(d):
    result = []
    for k in LATENT_KEYS:
        result.append(float(d[k]))
    return np.array(result, dtype=np.float32)

def score_products(belief_dist, user_types, product_embeddings):
    # calculate type vectors
    type_ids = sorted(user_types.keys())
    type_vectors = [dict_to_vec(user_types[tid]) for tid in type_ids]
    
    scores = {}
    for asin, emb in product_embeddings.items():
        p_vec = dict_to_vec(emb)
        
        expected_u = 0.0
        # Sum the utility for each archetype
        for i, t_vec in enumerate(type_vectors):
            u_for_this_type = -1.0 * np.sum((p_vec - t_vec)**2)
            expected_u += belief_dist[i] * u_for_this_type
            
        scores[asin] = expected_u
        
    return scores


def recommend_product( belief, user_types, product_embeddings):
    scores = score_products(belief, user_types, product_embeddings)

    b = max(scores, key=scores.get)
    return b, scores[b]

