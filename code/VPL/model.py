from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseContextEncoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.posterior_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.posterior_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, pair_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pair_features: [batch, num_pairs, 2 * latent_dim + 1]
        returns posterior stats: [batch, latent_dim], [batch, latent_dim]
        """
        encoded_pairs = self.pair_encoder(pair_features)
        pooled = encoded_pairs.mean(dim=1)
        return self.posterior_mu(pooled), self.posterior_logvar(pooled)


class RewardModel(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, item_embeddings: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if item_embeddings.dim() == 2 and latent.size(0) == 1:
            latent = latent.expand(item_embeddings.size(0), -1)
        features = torch.cat([item_embeddings, latent], dim=-1)
        return self.net(features).squeeze(-1)


class PaperStyleVPLRanker(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = PairwiseContextEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.reward_model = RewardModel(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def posterior_stats(self, pair_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(pair_features)

    def semantic_latent(self, posterior_mu: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(posterior_mu)

    def sample_latent(self, posterior_mu: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * posterior_logvar)
        eps = torch.randn_like(std)
        return posterior_mu + eps * std

    def score_candidates_by_latent_utility(
        self,
        posterior_mu: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        inferred_latent = self.semantic_latent(posterior_mu)
        if inferred_latent.dim() == 2 and inferred_latent.size(0) == 1:
            inferred_latent = inferred_latent.squeeze(0)
        diff = candidate_embeddings - inferred_latent
        return 1.0 - torch.mean(diff ** 2, dim=-1)

    def infer_latent(self, pair_features: torch.Tensor) -> torch.Tensor:
        posterior_mu, posterior_logvar = self.posterior_stats(pair_features)
        return self.sample_latent(posterior_mu, posterior_logvar)

    def preference_logits(
        self,
        latent: torch.Tensor,
        item_a_embeddings: torch.Tensor,
        item_b_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        reward_a = self.reward_model(item_a_embeddings, latent)
        reward_b = self.reward_model(item_b_embeddings, latent)
        return reward_a - reward_b

    def kl_divergence(self, posterior_mu: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(
            torch.exp(posterior_logvar) + posterior_mu ** 2 - 1.0 - posterior_logvar,
            dim=-1,
        ).mean()

    def score_candidates(self, inferred_latent: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        return self.reward_model(candidate_embeddings, inferred_latent)

    def forward(
        self,
        pair_features: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mu, posterior_logvar = self.posterior_stats(pair_features)
        latent_sample = self.sample_latent(posterior_mu, posterior_logvar)
        scores = self.score_candidates(latent_sample.squeeze(0), candidate_embeddings)
        return posterior_mu, posterior_logvar, latent_sample, scores
