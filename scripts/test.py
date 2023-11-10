import argparse
from pathlib import Path

import hydra
import lightning as l
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_template.dataset import DataModule


@hydra.main(config_path="../config", config_name="default_config")
def test(
    cfg: DictConfig,
    ckpt_path: Path,
) -> None:
    l.seed_everything(cfg.seed)

    model = instantiate(cfg.model.instance)

    dm = DataModule(cfg.data)
    dm.setup()

    trainer = l.Trainer(
        **cfg.trainer.args,
    )
    trainer.test(model, dm.test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    test(ckpt_path=Path(args.ckpt_path))
