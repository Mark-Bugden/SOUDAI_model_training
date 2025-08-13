from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from training.types import Batch, Sample


@dataclass
class FeatureCols:
    """Column names used by the dataset."""

    soud_id: str = "soud_id"
    autor_id: str = "autor_id"
    kw_ids: str = "klicovaSlova_ids"
    ust_ids: str = "zminenaUstanoveni_ids"
    target: str = "days_to_decision"


class DecisionDataset(Dataset):
    """Custom Dataset class to be used for Torch."""

    def __init__(self, df: pd.DataFrame, cols: FeatureCols | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.cols = cols or FeatureCols()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        soud = int(row[self.cols.soud_id])
        autor = int(row[self.cols.autor_id])
        kw = row[self.cols.kw_ids]  # already a list
        ust = row[self.cols.ust_ids]  # already a list
        target = float(row[self.cols.target])

        return {
            "soud_id": soud,
            "autor_id": autor,
            "kw_ids": kw,
            "ust_ids": ust,
            "target": target,
        }


def _pad_1d_list_batch(
    lists: List[List[int]],
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of 1D integer lists to a uniform length tensor and create a mask.

    This function takes a list of integer lists which may have varying lengths and pads
    them so that they all have the same length (the length of the longest list). It also
    generates a mask showing which values were not padded (1.0) and which were
    padded (0.0).

    If the input list is empty, returns empty (0, 0) tensors for both the padded
    values and the mask.

    Return

    Args:
      lists (List[List[int]]): Batch of integer lists to pad.
      pad_value (int, optional): Value to be used to pad lists. Default 0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - **padded**: torch.Tensor: Tensor of shape (B, Lmax) with B being the batch
                size (number of lists) and Lmax being the length of the longest list.
            - **mask**: torch.Tensor: Tensor of shape (B, Lmax) with 1.0 for valid
                elements and 0.0 for padding.

    """
    if not lists:
        z = torch.zeros(0, 0, dtype=torch.long)
        m = torch.zeros(0, 0, dtype=torch.float32)
        return z, m

    max_len = max(len(x) for x in lists)
    padded = []
    mask = []
    for x in lists:
        n = len(x)
        if n < max_len:
            pad = [pad_value] * (max_len - n)
            padded.append(x + pad)
            mask.append([1] * n + [0] * (max_len - n))
        else:
            padded.append(x[:max_len])
            mask.append([1] * max_len)
    pt = torch.tensor(padded, dtype=torch.long)
    mk = torch.tensor(mask, dtype=torch.float32)

    return pt, mk


def _collate(batch: List[Sample]) -> Batch:
    """Collate dict items into tensors. Needed because some tensors needs padding.

    This function takes a batch (list of dictionaries, with each item in the list
    representing one sample) and stacks and pads them into tensors suitable for model
    input.The result is the batch in a format in which it can be ingested by the model.

    Args:
        batch (List[Dict]): A list of dictionaries, where each dictionary must
            contain the following keys:
            - "soud_id" (int): Court ID.
            - "autor_id" (int): Judge ID.
            - "kw_ids" (List[int]): Sequence of keyword IDs.
            - "ust_ids" (List[int]): Sequence of statute IDs.
            - "target" (float): Target value.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing batched features.

    """
    soud = torch.tensor([b["soud_id"] for b in batch], dtype=torch.long)
    autor = torch.tensor([b["autor_id"] for b in batch], dtype=torch.long)
    kw_padded, kw_mask = _pad_1d_list_batch([b["kw_ids"] for b in batch])
    ust_padded, ust_mask = _pad_1d_list_batch([b["ust_ids"] for b in batch])
    target = torch.tensor([b["target"] for b in batch], dtype=torch.float32)

    return {
        "soud_id": soud,
        "autor_id": autor,
        "kw_ids": kw_padded,
        "kw_mask": kw_mask,
        "ust_ids": ust_padded,
        "ust_mask": ust_mask,
        "target": target,
    }


def make_dataloaders(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    df_test: pd.DataFrame,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    cols: FeatureCols | None = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    The only non-standard part of this function is the use of our custom `_collate` as a
    `collate_function`. This is required because some of our features consists of ragged
    lists (lists which have varying length) and we need to pad them.

    Args:
        df_train (pd.DataFrame): Train DataFrame.
        df_eval (pd.DataFrame): Evaluation DataFrame.
        df_test (pd.DataFrame): Test DataFrame.
        batch_size (int, optional): Size of each batch. Defaults to 512.
        num_workers (int, optional): Number of subprocesses for loading. Defaults to 0.
        pin_memory (bool): If True, enables faster transfer to GPU. Defaults to False.
        cols (FeatureCols, optional): Column selector for the dataset. If None, uses
            FeatureCols().
        logger (Optional[logging.Logger]): Logger for status messages. If None, logging
            is skipped.


    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, evaluation, and test
            DataLoaders whose batches are training.types.Batch
    """
    if logger:
        logger.info(
            "Building dataloaders (bs=%d, workers=%d)",
            batch_size,
            num_workers,
        )

    cols = cols or FeatureCols()
    train_ds = DecisionDataset(df_train, cols=cols)
    eval_ds = DecisionDataset(df_eval, cols=cols)
    test_ds = DecisionDataset(df_test, cols=cols)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )

    return train_dl, eval_dl, test_dl
