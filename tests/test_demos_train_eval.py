"""Test collecting demos, training and evaluation"""

import os
import shutil
import torch
from hydra.experimental import compose, initialize
from askdagger_cliport.demos import demos
from askdagger_cliport.train import train
from askdagger_cliport.train_interactive import train_interactive
from askdagger_cliport.eval import eval
from omegaconf import OmegaConf


def test_bc_interactive_eval():
    askdagger_root = os.environ["ASKDAGGER_ROOT"]
    root_dir = f"{askdagger_root}/test"
    assets_root = f"{askdagger_root}/src/askdagger_cliport/environments/assets/"

    agent = "cliport"
    train_demos = 2
    interactive_demos = 2
    max_steps = 2
    gpu = [0] if torch.cuda.is_available() else 0
    log = False
    iteration = 0
    disp = False
    task = "put-block-in-bowl-seen-colors"
    shared_memory = False
    save_video = False
    batch_size = 1
    n_workers = 1
    n_rotations = 1

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    with initialize(config_path="../src/askdagger_cliport/cfg"):
        demo_cfg = compose(
            config_name="data",
            overrides=[
                f"root_dir={root_dir}",
                f"assets_root={assets_root}",
                "mode=train",
                f"iteration={iteration}",
                f"shared_memory={shared_memory}",
                f"save_data=True",
                f"n={train_demos}",
                f"task={task}",
                f"disp={disp}",
                f"record.save_video={save_video}",
            ],
        )
    demos(demo_cfg)

    with initialize(config_path="../src/askdagger_cliport/cfg"):
        demo_cfg = compose(
            config_name="data",
            overrides=[
                f"root_dir={root_dir}",
                f"assets_root={assets_root}",
                "mode=val",
                f"iteration={iteration}",
                f"shared_memory={shared_memory}",
                f"save_data=True",
                f"n=1",
                f"task={task}",
                f"disp={disp}",
                f"record.save_video={save_video}",
            ],
        )
    demos(demo_cfg)

    with initialize(config_path="../src/askdagger_cliport/cfg"):
        train_cfg = compose(
            config_name="train",
            overrides=[
                f"root_dir={root_dir}",
                f"train.iteration={iteration}",
                f"train.task={task}",
                f"train.save_steps.0=1",
                f"train.save_steps.1=2",
                f"train.n_demos={train_demos}",
                f"train.n_steps={max_steps}",
                f"train.n_val=1",
                f"train.agent={agent}",
                f"train.gpu={gpu}",
                f"train.log={log}",
                f"train.batch_size={batch_size}",
                f"train.n_workers={n_workers}",
                f"train.n_rotations={n_rotations}",
            ],
        )
    train(train_cfg)

    config_path = f"{root_dir}/exps/{task}-{agent}-n{train_demos}-train/0/.hydra/config.yaml"
    # create config file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        OmegaConf.save(train_cfg, f)

    with initialize(config_path="../src/askdagger_cliport/cfg"):
        interactive_cfg = compose(
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
                f"shared_memory={shared_memory}",
                "save_model=True",
                f"train_interactive.gpu={gpu}",
                f"train_interactive.log={log}",
            ],
        )

    train_interactive(interactive_cfg)

    with initialize(config_path="../src/askdagger_cliport/cfg"):
        eval_cfg = compose(
            config_name="eval",
            overrides=[
                f"root_dir={root_dir}",
                f"assets_root={assets_root}",
                f"agent={agent}",
                f"iteration={iteration}",
                f"eval_task={task}",
                f"model_task={task}",
                "n_demos=2",
                f"disp={disp}",
                f"train_demos={train_demos}",
                f"interactive_demos={interactive_demos}",
                f"shared_memory={shared_memory}",
                f"record.save_video={save_video}",
                f"eval.gpu={gpu}",
                f"eval.log={log}",
            ],
        )
    eval(eval_cfg)

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
