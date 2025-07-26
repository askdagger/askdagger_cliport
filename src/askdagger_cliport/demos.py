# credit: https://github.com/cliport/cliport

"""Data collection script."""

import os
import hydra
import numpy as np
import random

from askdagger_cliport import tasks
from askdagger_cliport.dataset import RavensDataset
from askdagger_cliport.environments.environment import Environment


@hydra.main(version_base=1.3, config_path="./cfg", config_name="data")
def demos(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg["assets_root"], disp=cfg["disp"], shared_memory=cfg["shared_memory"], hz=480, record_cfg=cfg["record"]
    )
    task = tasks.names[cfg["task"]]()
    task.mode = cfg["mode"]
    record = cfg["record"]["save_video"]
    save_data = cfg["save_data"]
    iteration = cfg["iteration"]

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg["data_dir"], "{}-{}".format(cfg["task"], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == "train":
            seed = -2 + iteration * 10000
        elif task.mode == "val":  # NOTE: beware of increasing val set to >100
            seed = -1 + iteration * 500
        elif task.mode == "test":
            seed = -1 + 10000 + iteration * 500
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg["n"]:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print("Oracle demo: {}/{} | Seed: {}".format(dataset.n_episodes + 1, cfg["n"], seed))

        env.set_task(task)
        obs, info = env.reset(seed=seed)
        # info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == "val" and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f"{dataset.n_episodes+1:06d}")

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info["lang_goal"]
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            total_reward += reward
            print(f"Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}")
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)

    env.shutdown()


if __name__ == "__main__":
    demos()
