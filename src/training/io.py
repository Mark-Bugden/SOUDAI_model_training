from __future__ import annotations

import ast
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, PROCESSED_CSV_PATH
from training.encoder import MultiLabelEncoder, SingleLabelEncoder


def _ensure_list_raw(x: object) -> List:
    """Convert stringified lists to Python lists.

    If the input is a string representation of a list (e.g., `"[1, 2, 3]"`), it is
    parsed into a Python list. If parsing fails or the value is not a string, the value
    is returned unchanged unless it is invalid, in which case an empty list is returned.

    Args:
        x (Any): The input value to process.

    Returns:
        list: A Python list if parsing or conversion succeeds, otherwise an empty list.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x


def _normalize_list_column(df: pd.DataFrame, col: str) -> None:
    """Ensure all values in a DataFrame column are Python lists.

    This function replaces None, NaN, tuples, or numpy arrays with lists. Empty or
    invalid values are replaced with empty lists.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to normalize.
        col (str): The name of the column to normalize.

    Returns:
        None
    """

    def to_list(v: object) -> List:
        if v is None:
            return []
        if isinstance(v, float) and np.isnan(v):
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        if hasattr(v, "tolist"):  # numpy array / pandas Series
            return list(v.tolist())
        return []

    df[col] = df[col].apply(to_list)


def load_or_prepare_encoded(
    force_rebuild: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load or build encoded datasets for model training.

    Loads pre-encoded train, eval, and test splits from cached Parquet files if
    available, unless `force_rebuild` is set. Otherwise, it reads the processed CSV,
    fits encoders on the training set, transforms all splits, and caches them along with
    vocabulary sizes.

    Args:
        force_rebuild (bool, optional): If True, rebuilds the encoded data even if
            cached files exist. Defaults to False.
        logger (logging.Logger | None, optional): Logger instance for status messages.
            If None, logging is skipped.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
            - Train DataFrame with encoded features.
            - Evaluation DataFrame with encoded features.
            - Test DataFrame with encoded features.
            - Dictionary mapping feature names to vocabulary sizes.

    """
    train_path = DATA_DIR / "encoded/train.parquet"
    eval_path = DATA_DIR / "encoded/eval.parquet"
    test_path = DATA_DIR / "encoded/test.parquet"
    vocab_sizes_path = DATA_DIR / "encoded/vocab_sizes.json"

    have_all = (
        os.path.exists(train_path)
        and os.path.exists(eval_path)
        and os.path.exists(test_path)
        and os.path.exists(vocab_sizes_path)
    )

    if have_all and not force_rebuild:
        if logger:
            logger.info("Loading cached encoded Parquet")
        df_train = pd.read_parquet(train_path)
        df_eval = pd.read_parquet(eval_path)
        df_test = pd.read_parquet(test_path)
        with open(vocab_sizes_path, "r") as f:
            vocab_sizes = json.load(f)
    else:
        if logger:
            logger.info("Building encodings from processed CSV")
        df = pd.read_csv(
            PROCESSED_CSV_PATH,
            parse_dates=["date_start", "date_decision", "date_end"],
        )
        df["klicovaSlova"] = df["klicovaSlova"].apply(_ensure_list_raw)
        df["zminenaUstanoveni"] = df["zminenaUstanoveni"].apply(_ensure_list_raw)

        df_train_raw, df_remaining = train_test_split(
            df, test_size=0.3, random_state=42
        )
        df_test_raw, df_eval_raw = train_test_split(
            df_remaining, test_size=0.5, random_state=42
        )

        # Fit encoders on train
        soud_encoder = SingleLabelEncoder()
        soud_encoder.fit(df_train_raw["soud"])

        autor_encoder = SingleLabelEncoder()
        autor_encoder.fit(df_train_raw["autor"])

        kw_encoder = MultiLabelEncoder(max_vocab_size=5000)
        kw_encoder.fit(df_train_raw["klicovaSlova"])

        ust_encoder = MultiLabelEncoder(max_vocab_size=5000)
        ust_encoder.fit(df_train_raw["zminenaUstanoveni"])

        def _transform(dfx: pd.DataFrame) -> pd.DataFrame:
            """

            Args:
              dfx: pd.DataFrame:

            Returns:

            """
            dfx = dfx.copy()
            dfx["soud_id"] = soud_encoder.transform(dfx["soud"])
            dfx["autor_id"] = autor_encoder.transform(dfx["autor"])
            dfx["klicovaSlova_ids"] = kw_encoder.transform(dfx["klicovaSlova"])
            dfx["zminenaUstanoveni_ids"] = ust_encoder.transform(
                dfx["zminenaUstanoveni"]
            )
            return dfx

        df_train = _transform(df_train_raw)
        df_eval = _transform(df_eval_raw)
        df_test = _transform(df_test_raw)

        # Ensure output dir exists
        (DATA_DIR / "encoded").mkdir(parents=True, exist_ok=True)

        # Save artifacts
        df_train.to_parquet(train_path)
        df_eval.to_parquet(eval_path)
        df_test.to_parquet(test_path)

        vocab_sizes = {
            "soud": soud_encoder.vocab_size(),
            "autor": autor_encoder.vocab_size(),
            "klicovaSlova": kw_encoder.vocab_size(),
            "zminenaUstanoveni": ust_encoder.vocab_size(),
        }
        with open(vocab_sizes_path, "w") as f:
            json.dump(vocab_sizes, f)

    # Normalize list columns just before returning
    for col in ["klicovaSlova_ids", "zminenaUstanoveni_ids"]:
        _normalize_list_column(df_train, col)
        _normalize_list_column(df_eval, col)
        _normalize_list_column(df_test, col)

    if logger:
        logger.info(
            "Loaded splits: train=%s eval=%s test=%s",
            df_train.shape,
            df_eval.shape,
            df_test.shape,
        )
    return df_train, df_eval, df_test, vocab_sizes
