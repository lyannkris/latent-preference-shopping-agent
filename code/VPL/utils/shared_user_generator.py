from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
VPL_ROOT = ROOT.parent
PROJECT_ROOT = VPL_ROOT.parents[1]
GENERATE_USERS_PATH = ROOT / "generate_users.py"
DEFAULT_PICKLE_SPLIT_DIR = ROOT
DEFAULT_HELDOUT_CANDIDATES = [
    PROJECT_ROOT / "20260417" / "heldout_test_set.pkl",
    PROJECT_ROOT / "20260417" / "CURIO" / "data" / "heldout_test_set.pkl",
]


def load_generate_users_module():
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class OpenAI:  # pragma: no cover - import shim only
            pass

        openai_stub.OpenAI = OpenAI
        sys.modules["openai"] = openai_stub

    spec = importlib.util.spec_from_file_location("generate_users_runtime", GENERATE_USERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {GENERATE_USERS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_heldout_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    for candidate in DEFAULT_HELDOUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate heldout_test_set.pkl in {DEFAULT_HELDOUT_CANDIDATES}")


def sample_attributes(mod) -> Dict[str, object]:
    return {
        "age": random.randint(mod.AGE_RANGE[0], mod.AGE_RANGE[1]),
        "gender": random.choice(mod.GENDER_VALS),
        "event": random.choice(mod.EVENT_VALS),
        "size": random.choice(mod.SIZE_VALS),
        "income": random.choice(mod.INCOME_VALS),
        "location": random.choice(mod.LOCATION_VALS),
        "climate": random.choice(mod.CLIMATE_VALS),
        "activity_level": random.choice(mod.ACTIVITY_LEVEL_VALS),
        "social_frequency": random.choice(mod.SOCIAL_FREQUENCY_VALS),
        "world_news_awareness": random.choice(mod.WORLD_NEWS_AWARENESS_VALS),
        "brand_awareness": random.choice(mod.BRAND_AWARENESS_VALS),
        "highest_education_level": random.choice(mod.HIGHEST_EDUCATION_LEVEL_VALS),
        "personality": random.choice(mod.PERSONALITY_VALS),
        "name": random.choice(mod.NAME_VALS),
        "occupation": random.choice(mod.OCCUPATION_VALS),
        "relationship_status": random.choice(mod.RELATIONSHIP_STATUS_VALS),
        "languages_spoken": random.choice(mod.LANGUAGES_SPOKEN_VALS),
        "places_traveled": random.choice(mod.PLACES_TRAVELED_VALS),
        "pet_ownership": random.choice(mod.PET_OWNERSHIP_VALS),
        "sibling_count": random.randint(mod.SIBLING_COUNT_RANGE[0], mod.SIBLING_COUNT_RANGE[1]),
        "life_goals": random.choice(mod.LIFE_GOALS_VALS),
    }


def generate_attributes_only_users(num_users: int, seed: int, prefix: str) -> List[Dict[str, object]]:
    random.seed(seed)
    mod = load_generate_users_module()
    users = []
    for idx in range(num_users):
        users.append(
            {
                "user_id": f"{prefix}_{idx:05d}",
                "attributes": sample_attributes(mod),
            }
        )
    return users


def load_heldout_attributes_only_users(path: str | Path | None = None) -> List[Dict[str, object]]:
    heldout_path = resolve_heldout_path(path)
    mod = load_generate_users_module()
    sys.modules.setdefault("generate_users", mod)
    with open(heldout_path, "rb") as handle:
        users = pickle.load(handle)

    normalized = []
    for idx, user in enumerate(users):
        normalized.append(
            {
                "user_id": f"heldout_{idx:04d}",
                "attributes": dict(user.attributes),
            }
        )
    return normalized


def load_pickle_user_split(path: str | Path, prefix: str) -> List[Dict[str, object]]:
    mod = load_generate_users_module()
    sys.modules.setdefault("generate_users", mod)
    with open(path, "rb") as handle:
        users = pickle.load(handle)

    normalized = []
    for idx, user in enumerate(users):
        normalized.append(
            {
                "user_id": f"{prefix}_{idx:05d}",
                "attributes": dict(user.attributes),
                "latent_variables": {
                    key: float(value)
                    for key, value in dict(user.latent_variables).items()
                },
            }
        )
    return normalized


def load_pickle_user_splits(split_dir: str | Path = DEFAULT_PICKLE_SPLIT_DIR) -> Dict[str, List[Dict[str, object]]]:
    split_root = Path(split_dir)
    return {
        "train": load_pickle_user_split(split_root / "user_train_set.pkl", "train"),
        "eval": load_pickle_user_split(split_root / "user_eval_set.pkl", "eval"),
        "test": load_pickle_user_split(split_root / "user_test_set.pkl", "test"),
    }


def infer_true_latent(attributes: Dict[str, Any]) -> Dict[str, float]:
    mod = load_generate_users_module()
    latent = mod.attribute_latent_variables_mapping(attributes)
    return {key: float(value) for key, value in latent.items()}


def build_shared_user_splits(
    train_size: int = 2000,
    eval_size: int = 200,
    test_size: int = 200,
    seed: int = 42,
    heldout_path: str | Path | None = None,
    test_source: str = "heldout",
    pickle_split_dir: str | Path | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    if pickle_split_dir is not None:
        return load_pickle_user_splits(pickle_split_dir)

    train_users = generate_attributes_only_users(train_size, seed=seed, prefix="train")
    eval_users = generate_attributes_only_users(eval_size, seed=seed + 1, prefix="eval")
    if test_source == "heldout":
        heldout_users = load_heldout_attributes_only_users(heldout_path)
        test_users = heldout_users[:test_size]
    elif test_source == "generated":
        test_users = generate_attributes_only_users(test_size, seed=seed + 2, prefix="test")
    else:
        raise ValueError(f"Unknown test_source: {test_source}")
    return {
        "train": train_users,
        "eval": eval_users,
        "test": test_users,
    }


def save_shared_users_jsonl(users: List[Dict[str, object]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for user in users:
            handle.write(json.dumps(user, ensure_ascii=False) + "\n")


def load_shared_users_jsonl(path: str | Path) -> List[Dict[str, object]]:
    users: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                users.append(json.loads(stripped))
    return users


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attributes-only shared users for paper-style VPL.")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-source", choices=["heldout", "generated"], default="heldout")
    parser.add_argument("--pickle-split-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VPL_ROOT / "results" / "shared_users",
    )
    parser.add_argument("--heldout-path", type=str, default=None)
    args = parser.parse_args()

    splits = build_shared_user_splits(
        train_size=args.train_size,
        eval_size=args.eval_size,
        test_size=args.test_size,
        seed=args.seed,
        heldout_path=args.heldout_path,
        test_source=args.test_source,
        pickle_split_dir=args.pickle_split_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, users in splits.items():
        save_shared_users_jsonl(users, args.output_dir / f"{split_name}.jsonl")
    manifest = {
        "train_size": args.train_size,
        "eval_size": args.eval_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "test_source": args.test_source,
            "pickle_split_dir": str(args.pickle_split_dir) if args.pickle_split_dir else None,
            "output_dir": str(args.output_dir),
        }
    with open(args.output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved shared user splits to {args.output_dir}")


if __name__ == "__main__":
    main()
