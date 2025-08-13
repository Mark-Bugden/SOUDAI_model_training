from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Dict, Literal, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def _select_device(name: Literal["auto", "cpu", "cuda"] = "auto") -> torch.device:
    """Select a computation device for PyTorch.

    Chooses a CUDA GPU if available and `name` is "auto",
    otherwise returns the device specified by `name`.

    Args:
        name (Literal["auto", "cpu", "cuda"], optional): Name of the device,
            e.g., "cpu", "cuda", or "auto". If "auto", selects "cuda" if available,
            otherwise "cpu". Defaults to "auto".

    Returns:
        torch.device: Selected device object.
    """
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class Trainer:
    """Utility class for training and evaluating PyTorch regression models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        use_huber: bool = False,
        logger: logging.Logger = None,
        run_dir: str | Path | None = None,
        save_best: bool = True,
    ) -> None:
        self.model = model
        self.device = _select_device(device)
        self.model.to(self.device)

        self.loss_fn = nn.HuberLoss() if use_huber else nn.MSELoss()
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.logger = logger
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.save_best = save_best

        self._best_rmse = float("inf")
        self._best_state: Dict[str, torch.Tensor] | None = None

        self.history = []

    @torch.no_grad()
    def evaluate(self, dl: DataLoader) -> Tuple[float, float, float]:
        """Evaluate the model on a validation or test DataLoader.

        Args:
            dl (DataLoader): DataLoader containing evaluation batches.

        Returns:
            tuple[float, float, float]: Tuple containing:
                - avg_loss: Mean loss over the dataset.
                - rmse: Root Mean Squared Error.
                - mae: Mean Absolute Error.
        """
        self.model.eval()
        sse = 0.0
        sae = 0.0
        n = 0
        loss_sum = 0.0
        for batch in dl:
            y = batch["target"].to(self.device)
            x = {k: v.to(self.device) for k, v in batch.items() if k != "target"}
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss_sum += float(loss.item()) * y.shape[0]
            diff = y_hat - y
            sse += float(torch.sum(diff * diff).item())
            sae += float(torch.sum(torch.abs(diff)).item())
            n += int(y.shape[0])
        rmse = math.sqrt(sse / max(n, 1))
        mae = sae / max(n, 1)
        avg_loss = loss_sum / max(n, 1)
        return avg_loss, rmse, mae

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        *,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """Train the model for a specified number of epochs.

        Runs training and validation loops, updates metrics history,
        and optionally saves the best model checkpoint.

        Args:
            train_dl (DataLoader): DataLoader for the training set.
            val_dl (DataLoader): DataLoader for the validation set.
            epochs (int, optional): Number of training epochs. Defaults to 10.

        Returns:
            dict[str, float]: Dictionary containing the best validation RMSE.
        """
        v_loss, v_rmse, v_mae = self.evaluate(val_dl)
        self._log(
            "Epoch 0  val_loss=%.4f  val_rmse=%.4f  val_mae=%.4f",
            v_loss,
            v_rmse,
            v_mae,
        )
        self._maybe_update_best(v_rmse, save_now=True)

        outer = tqdm(
            total=epochs,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
            disable=self.logger is None,
        )

        inner = tqdm(
            total=len(train_dl),
            desc=f"Epoch 1/{epochs}",
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=self.logger is None,
        )

        for ep in range(1, epochs + 1):
            self.model.train()

            # reset inner bar for this epoch (same line, no new bar)
            inner.reset(total=len(train_dl))
            inner.set_description(f"Epoch {ep}/{epochs}")

            train_loss_sum = 0.0
            train_n = 0

            for batch in train_dl:
                y = batch["target"].to(self.device)
                x = {k: v.to(self.device) for k, v in batch.items() if k != "target"}
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                train_loss_sum += float(loss.item()) * int(y.shape[0])
                train_n += int(y.shape[0])

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                # advance inner bar by one batch
                inner.update(1)

            train_loss = train_loss_sum / max(train_n, 1)

            # validation and final postfix for this epoch
            v_loss, v_rmse, v_mae = self.evaluate(val_dl)
            inner.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": f"{v_loss:.4f}",
                    "rmse": f"{v_rmse:.4f}",
                    "mae": f"{v_mae:.4f}",
                }
            )

            self._maybe_update_best(v_rmse, save_now=True)

            # record metrics row
            self.history.append(
                {
                    "epoch": ep,
                    "train_loss": train_loss,
                    "val_loss": v_loss,
                    "val_rmse": v_rmse,
                    "val_mae": v_mae,
                }
            )

            # advance outer bar by one epoch
            outer.update(1)

        inner.close()
        outer.close()

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        # save history to CSV once per run
        if self.run_dir is not None and self.history:
            metrics_path = self.run_dir / "metrics.csv"
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_rmse",
                        "val_mae",
                    ],
                )
                writer.writeheader()
                writer.writerows(self.history)
            if self.logger:
                self.logger.info("Saved metrics: %s", str(metrics_path))

        return {"val_rmse": self._best_rmse}

    def _save_checkpoint(self, tag: str = "best") -> None:
        """Save the model's state dictionary to a checkpoint file.

        Args:
            tag (str, optional): File tag for the checkpoint. Defaults to "best".

        Returns:
            None
        """
        if self.run_dir is None:
            return
        path = self.run_dir / f"{tag}.ckpt"
        state = self.model.state_dict()
        torch.save(state, path)
        if self.logger:
            self.logger.info("Saved checkpoint: %s", str(path))

    def _maybe_update_best(self, val_rmse: float, save_now: bool) -> None:
        """Update the best model state if the given RMSE is lower.

        Args:
            val_rmse (float): Validation RMSE from the latest evaluation.
            save_now (bool): If True, immediately save a checkpoint when updated.

        Returns:
            None
        """
        if val_rmse < self._best_rmse:
            self._best_rmse = val_rmse
            self._best_state = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
            if save_now and self.save_best and self.run_dir is not None:
                self._save_checkpoint(tag="best")

    def _log(self, msg: str, *args: object) -> None:
        """Log a formatted message using the provided logger or print fallback.

        Args:
            msg (str): Log message format string.
            *args (object): Arguments for string formatting.

        Returns:
            None
        """
        if self.logger:
            self.logger.info(msg, *args)
        else:
            # Fallback to print with %-style formatting
            if args:
                print(msg % args)
            else:
                print(msg)
