import json
import numpy as np
from typing import Dict, Tuple, List
from reward_module import utility, dict_to_vec


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


# -----------------------------
# LOADING
# -----------------------------

def load_product_embeddings(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r") as f:
        return json.load(f)


def load_user_types(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# BELIEF-WEIGHTED SCORING
# -----------------------------

def score_products(
    mu: np.ndarray,                     # belief mean vector (8,)
    product_embeddings: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Score products using the estimated user preference vector.
    """

    scores: Dict[str, float] = {}

    for asin, emb in product_embeddings.items():
        p_vec = dict_to_vec(emb)

        score = utility(p_vec, mu)

        # small noise helps break ties
        score += np.random.normal(0, 0.01)

        scores[asin] = float(score)

    return scores


def recommend_product(
    mu: np.ndarray,
    product_embeddings: Dict[str, Dict[str, float]],
) -> Tuple[str, float]:

    scores = score_products(mu, product_embeddings)

    best_asin = max(scores, key=scores.get)

    return best_asin, scores[best_asin]


# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":
    # Example: random belief over 10 user types
    user_types = load_user_types("user_types.json")
    product_embeddings = load_product_embeddings("product_embeddings.json")

    K = len(user_types)
    raw = np.random.randn(K)
    belief = np.exp(raw) / np.exp(raw).sum()  # softmax

    best_asin, best_score = recommend_product(belief, user_types, product_embeddings)
    print("Best ASIN:", best_asin)
    print("Best score:", best_score)
