import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List


# -----------------------------
# CONFIG
# -----------------------------

OUTPUT_FILE = "data/user_types.json"
NUM_USER_TYPES = 8  # you can change this (e.g., 8–12)


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
# DATA CLASS
# -----------------------------

@dataclass
class UserType:
    comfort: float
    functionality: float
    maintenance: float
    status_symbol: float
    modesty: float
    sustainability: float
    risk_tolerance: float
    trend_sensitivity: float


# -----------------------------
# GENERATION LOGIC
# -----------------------------

def sample_user_type(alpha: float = 2.0, beta: float = 2.0) -> UserType:
    """Sample one user type as an 8-dim vector in [0,1]^8 from Beta(alpha, beta)."""
    values = np.random.beta(alpha, beta, size=len(LATENT_KEYS))
    return UserType(**{k: float(v) for k, v in zip(LATENT_KEYS, values)})


def generate_user_types(n: int) -> List[UserType]:
    return [sample_user_type() for _ in range(n)]


# -----------------------------
# MAIN
# -----------------------------

def main():
    user_types = generate_user_types(NUM_USER_TYPES)

    # Save as a dict: { "type_0": {...}, "type_1": {...}, ... }
    output = {f"type_{i}": asdict(ut) for i, ut in enumerate(user_types)}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {NUM_USER_TYPES} user types to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
