from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from utils.shared_user_generator import build_shared_user_splits, infer_true_latent


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_DATA_DIR_CANDIDATES = [
    ROOT / "utils",
    PROJECT_ROOT / "20260417" / "CURIO" / "data",
    PROJECT_ROOT / "VPL_old" / "data",
]

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


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_default_data_dir() -> Path:
    for candidate in DEFAULT_DATA_DIR_CANDIDATES:
        if (candidate / "product_embeddings.json").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate product_embeddings.json in any default data directory: "
        f"{DEFAULT_DATA_DIR_CANDIDATES}"
    )


def load_product_embeddings(path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
    resolved_path = Path(path) if path is not None else (resolve_default_data_dir() / "product_embeddings.json")
    raw = load_json(resolved_path)
    return {
        str(item_id): {key: float(value) for key, value in latent.items()}
        for item_id, latent in raw.items()
    }


def dict_to_vec(d: Dict[str, float]) -> np.ndarray:
    return np.array([d[key] for key in LATENT_KEYS], dtype=np.float32)


def compute_utility(user_latent_vector: np.ndarray, product_latent_vector: np.ndarray) -> float:
    diff = user_latent_vector - product_latent_vector
    mse = float(np.mean(diff ** 2))
    return 1.0 - mse


@dataclass
class PairwisePreference:
    item_a: str
    item_b: str
    label: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PaperVPLEpisode:
    episode_id: str
    user_id: str
    user_attributes: Dict[str, object]
    user_latent: Dict[str, float]
    context_pairs: List[PairwisePreference]
    query_pairs: List[PairwisePreference]
    candidate_item_ids: List[str]
    oracle_best_item_id: str
    oracle_best_utility: float

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["context_pairs"] = [pair.to_dict() for pair in self.context_pairs]
        return payload


def get_oracle_best_product(
    user_latent: Dict[str, float],
    candidate_item_ids: List[str],
    product_embeddings: Dict[str, Dict[str, float]],
) -> tuple[str, float]:
    user_vec = dict_to_vec(user_latent)
    scores = []
    for item_id in candidate_item_ids:
        product_vec = dict_to_vec(product_embeddings[item_id])
        scores.append(compute_utility(user_vec, product_vec))
    best_idx = int(np.argmax(scores))
    return candidate_item_ids[best_idx], float(scores[best_idx])


def sample_pairwise_context(
    user_latent: Dict[str, float],
    product_embeddings: Dict[str, Dict[str, float]],
    rng: random.Random,
    num_pairs: int,
) -> List[PairwisePreference]:
    product_ids = list(product_embeddings.keys())
    user_vec = dict_to_vec(user_latent)
    pairs: List[PairwisePreference] = []

    for _ in range(num_pairs):
        item_a, item_b = rng.sample(product_ids, 2)
        score_a = compute_utility(user_vec, dict_to_vec(product_embeddings[item_a]))
        score_b = compute_utility(user_vec, dict_to_vec(product_embeddings[item_b]))
        label = int(score_a >= score_b)
        pairs.append(PairwisePreference(item_a=item_a, item_b=item_b, label=label))
    return pairs


def build_dimension_probe_pairs(
    product_embeddings: Dict[str, Dict[str, float]],
    max_pairs: int | None = None,
) -> List[tuple[str, str]]:
    """
    Build fixed item pairs that probe each latent dimension.

    For every latent dimension, choose one pair with a large contrast on that
    target dimension and a small contrast on the remaining dimensions. This
    makes K=8 interpretable as one preference observation per latent axis.
    """
    product_ids = list(product_embeddings.keys())
    product_matrix = np.stack([dict_to_vec(product_embeddings[item_id]) for item_id in product_ids])
    selected_pairs: List[tuple[str, str]] = []
    used_pair_keys = set()

    for dim_idx in range(len(LATENT_KEYS)):
        best_pair: tuple[str, str] | None = None
        best_score = -float("inf")
        for i in range(len(product_ids)):
            for j in range(i + 1, len(product_ids)):
                diff = np.abs(product_matrix[i] - product_matrix[j])
                target_gap = float(diff[dim_idx])
                off_target_gap = float((np.sum(diff) - target_gap) / max(len(LATENT_KEYS) - 1, 1))
                score = target_gap - 0.35 * off_target_gap
                pair_key = tuple(sorted((product_ids[i], product_ids[j])))
                if pair_key in used_pair_keys:
                    continue
                if score > best_score:
                    best_score = score
                    best_pair = (product_ids[i], product_ids[j])
        if best_pair is None:
            best_pair = tuple(product_ids[:2])  # defensive fallback for tiny product pools
        used_pair_keys.add(tuple(sorted(best_pair)))
        selected_pairs.append(best_pair)

    if max_pairs is not None:
        return selected_pairs[:max_pairs]
    return selected_pairs


def label_item_pair(
    user_latent: Dict[str, float],
    product_embeddings: Dict[str, Dict[str, float]],
    item_a: str,
    item_b: str,
) -> PairwisePreference:
    user_vec = dict_to_vec(user_latent)
    score_a = compute_utility(user_vec, dict_to_vec(product_embeddings[item_a]))
    score_b = compute_utility(user_vec, dict_to_vec(product_embeddings[item_b]))
    label = int(score_a >= score_b)
    return PairwisePreference(item_a=item_a, item_b=item_b, label=label)


def sample_dimension_aware_context(
    user_latent: Dict[str, float],
    product_embeddings: Dict[str, Dict[str, float]],
    rng: random.Random,
    num_pairs: int,
    probe_pairs: List[tuple[str, str]] | None = None,
) -> List[PairwisePreference]:
    if probe_pairs is None:
        probe_pairs = build_dimension_probe_pairs(product_embeddings, max_pairs=min(num_pairs, len(LATENT_KEYS)))
    else:
        probe_pairs = probe_pairs[: min(num_pairs, len(probe_pairs))]
    pairs = [
        label_item_pair(
            user_latent=user_latent,
            product_embeddings=product_embeddings,
            item_a=item_a,
            item_b=item_b,
        )
        for item_a, item_b in probe_pairs
    ]
    if num_pairs > len(pairs):
        pairs.extend(
            sample_pairwise_context(
                user_latent=user_latent,
                product_embeddings=product_embeddings,
                rng=rng,
                num_pairs=num_pairs - len(pairs),
            )
        )
    return pairs


def build_episode(
    user: Dict[str, object],
    episode_id: str,
    product_embeddings: Dict[str, Dict[str, float]],
    rng: random.Random,
    num_context_pairs: int = 10,
    num_query_pairs: int = 10,
    candidate_set_size: int = 0,
    context_sampling: str = "random",
    dimension_probe_pairs: List[tuple[str, str]] | None = None,
) -> PaperVPLEpisode:
    user_latent = dict(user["latent_variables"]) if "latent_variables" in user else infer_true_latent(user["attributes"])
    product_ids = list(product_embeddings.keys())
    if candidate_set_size and candidate_set_size > 0:
        candidate_item_ids = rng.sample(product_ids, min(candidate_set_size, len(product_ids)))
    else:
        candidate_item_ids = list(product_ids)

    oracle_best_item_id, oracle_best_utility = get_oracle_best_product(
        user_latent=user_latent,
        candidate_item_ids=candidate_item_ids,
        product_embeddings=product_embeddings,
    )

    return PaperVPLEpisode(
        episode_id=episode_id,
        user_id=str(user["user_id"]),
        user_attributes=dict(user["attributes"]),
        user_latent=dict(user_latent),
        context_pairs=(
            sample_dimension_aware_context(
                user_latent=user_latent,
                product_embeddings=product_embeddings,
                rng=rng,
                num_pairs=num_context_pairs,
                probe_pairs=dimension_probe_pairs,
            )
            if context_sampling == "dimension_aware"
            else sample_pairwise_context(
                user_latent=user_latent,
                product_embeddings=product_embeddings,
                rng=rng,
                num_pairs=num_context_pairs,
            )
        ),
        query_pairs=sample_pairwise_context(
            user_latent=user_latent,
            product_embeddings=product_embeddings,
            rng=rng,
            num_pairs=num_query_pairs,
        ),
        candidate_item_ids=candidate_item_ids,
        oracle_best_item_id=oracle_best_item_id,
        oracle_best_utility=oracle_best_utility,
    )


def build_dataset_splits(
    product_embeddings: Dict[str, Dict[str, float]],
    train_size: int = 2000,
    eval_size: int = 200,
    test_size: int = 200,
    num_context_pairs: int = 10,
    num_query_pairs: int = 10,
    candidate_set_size: int = 0,
    seed: int = 42,
    heldout_path: str | Path | None = None,
    context_sampling: str = "random",
    test_source: str = "heldout",
    pickle_split_dir: str | Path | None = None,
) -> Dict[str, List[PaperVPLEpisode]]:
    rng = random.Random(seed)
    dimension_probe_pairs = (
        build_dimension_probe_pairs(product_embeddings, max_pairs=min(num_context_pairs, len(LATENT_KEYS)))
        if context_sampling == "dimension_aware"
        else None
    )
    shared_user_splits = build_shared_user_splits(
        train_size=train_size,
        eval_size=eval_size,
        test_size=test_size,
        seed=seed,
        heldout_path=heldout_path,
        test_source=test_source,
        pickle_split_dir=pickle_split_dir,
    )

    def build_split(users: List[Dict[str, object]], prefix: str) -> List[PaperVPLEpisode]:
        episodes = []
        for idx, user in enumerate(users):
            episodes.append(
                build_episode(
                    user=user,
                    episode_id=f"{prefix}_{idx:06d}",
                    product_embeddings=product_embeddings,
                    rng=rng,
                    num_context_pairs=num_context_pairs,
                    num_query_pairs=num_query_pairs,
                    candidate_set_size=candidate_set_size,
                    context_sampling=context_sampling,
                    dimension_probe_pairs=dimension_probe_pairs,
                )
            )
        return episodes

    return {
        "train": build_split(shared_user_splits["train"], "train"),
        "eval": build_split(shared_user_splits["eval"], "eval"),
        "test": build_split(shared_user_splits["test"], "test"),
    }


def save_episodes_jsonl(episodes: Iterable[PaperVPLEpisode], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")


def load_episodes_jsonl(path: str | Path) -> List[PaperVPLEpisode]:
    episodes: List[PaperVPLEpisode] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            episodes.append(
                PaperVPLEpisode(
                    episode_id=raw["episode_id"],
                    user_id=raw["user_id"],
                    user_attributes=dict(raw["user_attributes"]),
                    user_latent={key: float(value) for key, value in raw["user_latent"].items()},
                    context_pairs=[PairwisePreference(**pair) for pair in raw["context_pairs"]],
                    query_pairs=[PairwisePreference(**pair) for pair in raw["query_pairs"]],
                    candidate_item_ids=list(raw["candidate_item_ids"]),
                    oracle_best_item_id=raw["oracle_best_item_id"],
                    oracle_best_utility=float(raw["oracle_best_utility"]),
                )
            )
    return episodes
