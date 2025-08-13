from typing import List, TypedDict

import torch


class Sample(TypedDict):
    """One row from the dataset, pre-collation."""

    soud_id: int
    autor_id: int
    kw_ids: List[int]
    ust_ids: List[int]
    target: float


class Batch(TypedDict):
    """The collated batch format consumed by the model."""

    soud_id: torch.Tensor  # (B,)
    autor_id: torch.Tensor  # (B,)
    kw_ids: torch.Tensor  # (B, L_kw)
    kw_mask: torch.Tensor  # (B, L_kw)
    ust_ids: torch.Tensor  # (B, L_ust)
    ust_mask: torch.Tensor  # (B, L_ust)
    target: torch.Tensor  # (B,)
