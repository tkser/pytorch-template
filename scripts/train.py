import argparse
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch import set_float32_matmul_precision

from pytorch_template.config import Config
from pytorch_template.dataset import DataModule
from pytorch_template.model import LitModule


def train(
    cfg: Config,
) -> None:
    seed_everything(cfg.seed)
    set_float32_matmul_precision("high")

    model = LitModule(cfg)

    dm = DataModule(cfg.data)
    dm.setup()

    logger = WandbLogger(project=cfg.project)
    logger.watch(model)
    logger.log_hyperparams(cfg)

    early_stopping = EarlyStopping(**cfg.callbacks.EarlyStopping)
    model_checkpoint = ModelCheckpoint(**cfg.callbacks.ModelCheckpoint)

    trainer = Trainer(
        logger=logger,
        callbacks=[early_stopping, model_checkpoint],
        **cfg.trainer,
    )
    trainer.fit(model, dm.train_dataloader, dm.val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf/default_config.yaml")
    args = parser.parse_args()

    base_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    train(cfg)
