"""Test agents"""

import os
import shutil
from hydra.experimental import compose, initialize
from askdagger_cliport.train_interactive import train_interactive
import torch


def test_train_interactive():
    askdagger_root = os.environ["ASKDAGGER_ROOT"]
    root_dir = f"{askdagger_root}/test"
    assets_root = f"{askdagger_root}/src/askdagger_cliport/environments/assets/"

    agent = "cliport"
    train_demos = 0
    interactive_demos = 1
    max_steps = 1
    gpu = [0] if torch.cuda.is_available() else 0
    log = False
    iteration = 0
    disp = False
    task = "put-block-in-bowl-seen-colors"
    batch_size = 1
    n_workers = 1

    with initialize(config_path="../src/askdagger_cliport/cfg"):
        icfg = compose(
            config_name="train_interactive",
            overrides=[
                f"root_dir={root_dir}",
                f"assets_root={assets_root}",
                f"max_steps={max_steps}",
                f"train_demos={train_demos}",
                f"train_steps={max_steps}",
                f"model_task={task}",
                f"iteration={iteration}",
                f"train_interactive_task={task}",
                f"interactive_demos={interactive_demos}",
                f"train_interactive.batch_size={batch_size}",
                f"train_interactive.n_workers={n_workers}",
                f"agent={agent}",
                f"disp={disp}",
                f"save_model=True",
                f"train_interactive.gpu={gpu}",
                f"train_interactive.log={log}",
            ],
        )
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    train_interactive(icfg=icfg)
    with initialize(config_path="../src/askdagger_cliport/cfg"):
        icfg = compose(
            config_name="train_interactive",
            overrides=[
                f"root_dir={root_dir}",
                f"assets_root={assets_root}",
                f"max_steps={max_steps}",
                f"train_demos={train_demos}",
                f"train_steps={max_steps}",
                f"model_task={task}",
                f"iteration={iteration}",
                f"train_interactive_task={task}",
                f"interactive_demos={interactive_demos+1}",
                f"train_interactive.batch_size={batch_size}",
                f"train_interactive.n_workers={n_workers}",
                f"agent={agent}",
                f"disp={disp}",
                f"save_model=True",
                f"train_interactive.gpu={gpu}",
                f"train_interactive.log={log}",
            ],
        )

    # Resume training
    train_interactive(icfg=icfg)

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
