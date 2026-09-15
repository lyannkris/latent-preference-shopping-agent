import numpy as np

def compute_utility(product_vec, user_vec) -> float:
    """
    Negative Squared Euclidean Distance: -||p - u||^2
    """
        
    # cast to numpy and flatten to ensure 1D vectors
    prod = np.asarray(product_vec).flatten()
    user = np.asarray(user_vec).flatten()
    
    return -1.0 * float(np.sum((prod - user)**2))


def compute_all_utilities(user_vec, product_mat):
    """
    Vectorized utility for the entire catalog using the expansion:
    ||p - u||^2 = ||p||^2 + ||u||^2 - 2<p, u>
    """
    prod_norms = np.sum(product_mat**2, axis=1)
    user_norm = np.sum(user_vec**2)
    dot_product = product_mat @ user_vec
    
    return -1.0 * (prod_norms + user_norm - 2 * dot_product)


def get_oracle_best_product(user_vec, prod_mat):
    """
    Finds the globally optimal product and its utility
    """
    utils = compute_all_utilities(user_vec, prod_mat)
    best_idx = np.argmax(utils)
    return int(best_idx), float(utils[best_idx])

def get_oracle_worst_product(user_vec, prod_mat):
    """
    Finds the globally worst product and its utility
    """
    utils = compute_all_utilities(user_vec, prod_mat)
    worst_idx = np.argmin(utils)
    return int(worst_idx), float(utils[worst_idx])


def get_best_feasible_product(user_vec, prod_mat, feasible_idxs):
    """
    Finds the best product within a specific feasible subset
    """
    if not feasible_idxs:
        return None, -np.inf
    
    # Calculate utilities only for the feasible subset
    sub_utils = compute_all_utilities(user_vec, prod_mat[feasible_idxs])
    local_idx = np.argmax(sub_utils)

    return int(feasible_idxs[local_idx]), float(sub_utils[local_idx])


def compute_regret(oracle_util, agent_util):
    """
    Regret = Optimal Utility - Achieved Utility
    """
    return float(oracle_util - agent_util)
