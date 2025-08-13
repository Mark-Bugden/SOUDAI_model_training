from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from training.types import Batch


class DecisionRegressor(nn.Module):
    """Tabular + multi-label embedding regressor (raw output)."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim_single: int = 32,
        embed_dim_multi: int = 32,
        hidden: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Single-label embeddings
        self.soud_emb = nn.Embedding(
            num_embeddings=vocab_sizes["soud"],
            embedding_dim=embed_dim_single,
        )
        self.autor_emb = nn.Embedding(
            num_embeddings=vocab_sizes["autor"],
            embedding_dim=embed_dim_single,
        )

        # Multi-label embeddings (pad index = 0)
        self.kw_emb = nn.Embedding(
            num_embeddings=vocab_sizes["klicovaSlova"],
            embedding_dim=embed_dim_multi,
            padding_idx=0,
        )
        self.ust_emb = nn.Embedding(
            num_embeddings=vocab_sizes["zminenaUstanoveni"],
            embedding_dim=embed_dim_multi,
            padding_idx=0,
        )

        in_dim = 2 * embed_dim_single + 2 * embed_dim_multi

        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]

        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emb: [B, L, D], mask: [B, L] -> [B, D].

        Args:
          emb: torch.Tensor:
          mask: torch.Tensor:

        Returns:

        """
        mask = mask.unsqueeze(-1)
        emb = emb * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return emb.sum(dim=1) / denom

    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
          batch:

        Returns:

        """
        # Single-label
        soud_vec = self.soud_emb(batch["soud_id"])
        autor_vec = self.autor_emb(batch["autor_id"])

        # Multi-label
        kw_vecs = self.kw_emb(batch["kw_ids"])
        kw_mean = self.masked_mean(kw_vecs, batch["kw_mask"])

        ust_vecs = self.ust_emb(batch["ust_ids"])
        ust_mean = self.masked_mean(ust_vecs, batch["ust_mask"])

        # Concat and regress
        feats = torch.cat([soud_vec, autor_vec, kw_mean, ust_mean], dim=1)
        y_hat = self.mlp(feats).squeeze(-1)
        return y_hat
