# credit: https://github.com/cliport/cliport

"""BC training script."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from askdagger_cliport import agents
from askdagger_cliport.dataset import RavensDataset, collate_fn

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=1.3, config_path="./cfg", config_name="train")
def train(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg["tag"]) if cfg["train"]["log"] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    if cfg["train"]["n_demos"] == 0:
        return
    checkpoint_path = os.path.join(cfg["train"]["train_dir"], "checkpoints")
    last_checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
    last_checkpoint = (
        last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg["train"]["load_from_last_ckpt"] else None
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg["wandb"]["saver"]["monitor"],
        filepath=os.path.join(checkpoint_path, "best"),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg["train"]["n_steps"] // cfg["train"]["n_demos"]
    batch_size = cfg["train"]["batch_size"]
    n_workers = cfg["train"]["n_workers"]
    trainer = Trainer(
        gpus=cfg["train"]["gpu"],
        fast_dev_run=cfg["debug"],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max(max_epochs // 50, 1),
        resume_from_checkpoint=last_checkpoint,
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt["epoch"]
        trainer.global_step = last_ckpt["global_step"]
        del last_ckpt

    # Config
    data_dir = cfg["train"]["data_dir"]
    task = cfg["train"]["task"]
    agent_type = cfg["train"]["agent"]
    n_demos = cfg["train"]["n_demos"]
    n_val = cfg["train"]["n_val"]
    name = "{}-{}-{}".format(task, agent_type, n_demos)

    # Datasets
    train_ds = RavensDataset(os.path.join(data_dir, "{}-train".format(task)), cfg, n_demos=n_demos, augment=True)
    val_ds = RavensDataset(os.path.join(data_dir, "{}-val".format(task)), cfg, n_demos=n_val, augment=False)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=n_workers, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, num_workers=n_workers, collate_fn=collate_fn)

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_dl, val_dl)
    agent.automatic_optimization = False

    # Main training loop
    trainer.fit(agent)


if __name__ == "__main__":
    train()
