# credit: https://github.com/cliport/cliport

"""Ravens main training script."""

import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import hydra
from askdagger_cliport import agents
from askdagger_cliport import dataset
from askdagger_cliport import tasks
from askdagger_cliport.utils import utils
from askdagger_cliport.environments.environment import Environment
from pathlib import Path
from hydra import compose, initialize
from omegaconf import OmegaConf
from askdagger_cliport.dataset import collate_fn


@hydra.main(config_path="./cfg", config_name="eval")
def eval(vcfg):
    # Load train cfg
    interactive_demos = vcfg["interactive_demos"]

    train_dir = Path(str(vcfg["model_path"]).split("checkpoints")[0])

    if interactive_demos > 0:
        interactive_train_dir = utils.get_interactive_train_dir(train_dir, vcfg)
        save_path = results_path = model_path = interactive_train_dir / "checkpoints"
    else:
        save_path = results_path = model_path = train_dir / "checkpoints/"

    vcfg["train_config"] = str(train_dir / ".hydra" / "config.yaml")
    vcfg["save_path"] = str(save_path)
    vcfg["results_path"] = str(results_path)
    vcfg["model_path"] = str(model_path)

    if not os.path.exists(vcfg["train_config"]) and vcfg["train_demos"] == 0:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        os.chdir(os.environ["ASKDAGGER_ROOT"])
        with initialize(config_path="cfg"):
            tcfg = compose(
                config_name="train",
                overrides=[
                    f"train.iteration={vcfg['iteration']}",
                    f"train.gpu={vcfg['eval']['gpu']}",
                    f"train.agent={vcfg['agent']}",
                    f"train.task={vcfg['model_task']}",
                ],
            )
        OmegaConf.set_struct(tcfg, False)
    else:
        tcfg = utils.load_hydra_config(vcfg["train_config"])
    tcfg["eval"] = vcfg["eval"]
    iteration = vcfg["iteration"]

    # Initialize environment and task.
    env = Environment(
        vcfg["assets_root"], disp=vcfg["disp"], shared_memory=vcfg["shared_memory"], hz=480, record_cfg=vcfg["record"]
    )

    # Choose eval mode and task.
    mode = vcfg["mode"]
    eval_task = vcfg["eval_task"]
    n_workers = vcfg["n_workers"]
    if mode not in {"train", "val", "test"}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    ds = dataset.RavensDataset(os.path.join(vcfg["data_dir"], f"{eval_task}-{mode}"), tcfg, n_demos=0, augment=False)
    # Dataloaders
    dl = DataLoader(ds, num_workers=n_workers, collate_fn=collate_fn)

    all_results = {}
    name = "{}-{}-n{}".format(eval_task, vcfg["agent"], vcfg["n_demos"])

    # Save path for results.
    json_name = f"results-{mode}.json"
    save_path = vcfg["save_path"]
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f"{name}-{json_name}")

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, "r") as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = utils.list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg["model_path"], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg["update_results"] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Initialize agent.
        seed = -1 + 10000 + iteration * 500
        utils.set_seed(seed, torch=True)
        agent = agents.names[vcfg["agent"]](name, tcfg, None, dl)

        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")

        record = vcfg["record"]["save_video"]
        n_demos = vcfg["n_demos"]

        # Run testing and save total rewards with last transition info.
        for i in range(0, n_demos):
            print(f"Test: {i + 1}/{n_demos}")
            total_reward = 0
            seed += 2
            np.random.seed(seed)

            # set task
            task_name = vcfg["eval_task"]
            task = tasks.names[task_name]()
            task.mode = mode
            env.set_task(task)
            obs, info = env.reset(seed=seed)
            reward = 0

            rewards = []
            hl_rewards = []
            hl_lang_goals = []

            # Start recording video (NOTE: super slow)
            if record:
                video_name = f"{task_name}-{i+1:06d}"
                env.start_rec(video_name)

            for j in range(task.max_steps):
                hl_reward = 0
                hl_lang_goal = None

                with torch.no_grad():
                    act = agent.act(obs, info)
                lang_goal = info["lang_goal"]
                print(f"Lang Goal: {lang_goal}")

                matched_objects_prev = env.task.get_matched_objects()

                # Check uncertainty
                pick_doubt = info["pick_doubt"] if "pick_doubt" in info else None
                place_doubt = info["place_doubt"] if "place_doubt" in info else None
                if pick_doubt is not None and place_doubt is not None:
                    print(f"doubt: {pick_doubt:.3f} | {place_doubt:.3f}")
                obs, reward, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                rewards.append(reward)
                total_reward += reward

                if not reward > 0:
                    matched_objects = env.task.get_matched_objects()
                    hl_lang_goal = env.task.relabel(matched_objects, matched_objects_prev)
                    if hl_lang_goal is not None:
                        objs, _, _, _, _, _, _, max_reward = env.task.goals[0]
                        hl_reward = max_reward / len(objs)
                hl_rewards.append(hl_reward)
                hl_lang_goals.append(hl_lang_goal if hl_lang_goal is not None else "")

                print(f"Total Reward: {total_reward:.3f} | Done: {done}\n")
                if done:
                    break
            results.append((total_reward, rewards, hl_rewards, hl_lang_goal))
            mean_reward = np.mean([t for t, _, _, _ in results])
            print(f"Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}")

            # End recording video
            if record:
                env.end_rec()

        all_results[ckpt] = {
            "episodes": results,
            "mean_reward": mean_reward,
        }

        # Save results in a json file.
        if vcfg["save_results"]:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, "r") as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, "w") as f:
                json.dump(all_results, f, indent=4)
    env.shutdown()


if __name__ == "__main__":
    eval()
