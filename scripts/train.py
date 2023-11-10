import hydra
import lightning as l
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_template.dataset import DataModule


@hydra.main(config_path="../config", config_name="default_config")
def train(
    cfg: DictConfig,
) -> None:
    l.seed_everything(cfg.seed)

    model = instantiate(cfg.model.instance)

    dm = DataModule(cfg.data)
    dm.setup()

    logger = instantiate(cfg.logger)
    logger.watch(model, log="gradients", log_freq=100)

    early_stopping = instantiate(cfg.callbacks.EarlyStopping)
    model_checkpoint = instantiate(cfg.callbacks.ModelCheckpoint)

    trainer = l.Trainer(
        logger=logger,
        callbacks=[early_stopping, model_checkpoint],
        **cfg.trainer.args,
    )
    trainer.fit(model, dm.train_loader, dm.val_loader)


if __name__ == "__main__":
    train()
