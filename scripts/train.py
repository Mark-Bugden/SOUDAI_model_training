import training.training_config as cfg
from config import RUNS_PATH
from training.data import make_dataloaders
from training.io import load_or_prepare_encoded
from training.models import DecisionRegressor
from training.trainer import Trainer
from utils import get_logger, prepare_run_dir, save_json


def main():
    """ """
    logger = get_logger("train", cfg.LOG_LEVEL, cfg.USE_LOGGING)

    # Prepare a timestamped run directory under the global RUNS_DIR
    run_dir = prepare_run_dir(RUNS_PATH)
    if logger:
        logger.info("Run directory: %s", str(run_dir))

    # Data
    df_train, df_eval, df_test, vocab_sizes = load_or_prepare_encoded(
        force_rebuild=cfg.FORCE_REBUILD,
        logger=logger,
    )
    train_dl, eval_dl, _ = make_dataloaders(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        logger=logger,
    )

    # Model
    model = DecisionRegressor(
        vocab_sizes=vocab_sizes,
        embed_dim_single=cfg.EMBED_DIM_SINGLE,
        embed_dim_multi=cfg.EMBED_DIM_MULTI,
        hidden=cfg.HIDDEN,
        dropout=cfg.DROPOUT,
    )

    # Save a minimal config snapshot for reproducibility
    cfg_snapshot = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": cfg.PIN_MEMORY,
        "epochs": cfg.EPOCHS,
        "lr": cfg.LR,
        "weight_decay": cfg.WEIGHT_DECAY,
        "device": cfg.DEVICE,
        "use_huber": cfg.USE_HUBER,
        "embed_dim_single": cfg.EMBED_DIM_SINGLE,
        "embed_dim_multi": cfg.EMBED_DIM_MULTI,
        "hidden": list(cfg.HIDDEN),
        "dropout": cfg.DROPOUT,
        "vocab_sizes": vocab_sizes,
    }
    save_json(run_dir / "config_snapshot.json", cfg_snapshot)

    # Trainer
    trainer = Trainer(
        model,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        device=cfg.DEVICE,
        use_huber=cfg.USE_HUBER,
        logger=logger,
        run_dir=run_dir,
        save_best=cfg.SAVE_BEST,
    )

    # Train
    metrics = trainer.fit(
        train_dl=train_dl,
        val_dl=eval_dl,
        epochs=cfg.EPOCHS,
    )

    print("Finished training", metrics)


if __name__ == "__main__":
    main()
