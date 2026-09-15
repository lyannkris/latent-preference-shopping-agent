import json

USER_TYPES_FILE = "curio_output/user_types.json"


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


def load_user_types(path: str):
    with open(path, "r") as f:
        return json.load(f)
