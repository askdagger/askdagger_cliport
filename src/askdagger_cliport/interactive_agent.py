import sys
import os
import json
import warnings
import time
import cv2
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, Container
import tempfile
import shutil

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import askdagger_cliport
from askdagger_cliport import agents
from askdagger_cliport.dataset import RavensDataset, collate_fn
from askdagger_cliport.sag import sag
from askdagger_cliport.utils import utils
from askdagger_cliport.pier import prioritize_sampling


class InteractiveAgent:
    def __init__(self, icfg, mode, tcfg=None):
        self._icfg = icfg
        self._tcfg = tcfg if tcfg is not None else utils.get_train_cfg(icfg)
        self._mode = mode

        # Interactive Config
        self._logging = icfg["train_interactive"]["log"]
        self._log_window = icfg["train_interactive"]["log_window"]
        self._random_rate = icfg["train_interactive"]["p_rand"]
        self._pier = icfg["train_interactive"]["pier"]
        self._alpha = icfg["train_interactive"]["alpha"]
        self._beta0 = icfg["train_interactive"]["beta0"]
        self._lambda = icfg["train_interactive"]["lambda"]
        self._b = icfg["train_interactive"]["b"]
        self._bias_compensation = icfg["train_interactive"]["bias_compensation"]
        self._train_interactive_task = icfg["train_interactive_task"]
        self._s_desired = icfg["train_interactive"]["s_desired"]
        self._n_min = icfg["train_interactive"]["n_min"]
        self._load_from_last_ckpt = icfg["train_interactive"]["load_from_last_ckpt"]
        self._train_interactive_task = icfg["train_interactive_task"]
        self._gpu = icfg["train_interactive"]["gpu"]
        self._batch_size = icfg["train_interactive"]["batch_size"]
        self._n_workers = icfg["train_interactive"]["n_workers"]
        self._validation_demos = icfg["validation_demos"]
        self._relabling_demos = icfg["relabeling_demos"]
        self._sag = icfg["sag"]
        self._iteration = icfg["iteration"]
        self._interactive_demos = icfg["interactive_demos"]
        self._save_every = icfg["save_every"]
        self._train_demos = icfg["train_demos"]
        self._save_path = icfg["save_path"]
        self._save_model = icfg["save_model"]
        self._save_results = icfg["save_results"]
        self._model_path = icfg["model_path"]
        self._train_steps = icfg["train_steps"]
        self._max_epochs = icfg["max_epochs"]
        self._max_steps = icfg["max_steps"]

        # Initialize data directory
        self.prepare_data_dir()

        # Train Config
        self._train_data_dir = self._tcfg["train"]["data_dir"]
        self._train_task = self._tcfg["train"]["task"]

        # Load and check existing results
        self._get_save_json()
        self._get_logger()
        self._load_stats()
        self._get_latest_checkpoint()
        self._initialize_datasets()
        self._check_seed()
        self._initialize_stats()
        self._filter_warnings()

        # Initialize agent.
        self._agent = agents.names[icfg["agent"]](self._name, self._tcfg, self._train_dl, None)

        # Load checkpoint
        if self._model_file is not None:
            self._agent.load(self._model_file)

        # Initialize trainer
        self._trainer = Trainer(
            checkpoint_callback=False,
            gpus=self._gpu,
            fast_dev_run=False,
            logger=self._wandb_logger,
            max_epochs=self._max_epochs,
            automatic_optimization=False,
            max_steps=self._max_steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=0,
            weights_summary=None,
            log_every_n_steps=1,
        )
        self._agent.save_steps = []

        # Episode stats
        self._novice_rewards = []
        self._system_rewards = []

    @property
    def seed(self):
        return self._stats["seed"]

    @seed.setter
    def seed(self, value):
        self._stats["seed"] = value

    @property
    def n_interactive(self):
        return self._stats["n_interactive"]

    @property
    def episode_count(self):
        return len(self._stats["episodes"])

    @property
    def bounds(self):
        return self._train_ds.bounds

    @bounds.setter
    def bounds(self, value):
        self._train_ds.bounds = value
        self._agent.bounds = value
        self._update_dataloader()

    @property
    def pix_size(self):
        return self._train_ds.pix_size

    @pix_size.setter
    def pix_size(self, value):
        self._train_ds.pix_size = value
        self._agent.pix_size = value
        self._update_dataloader()

    @property
    def in_shape(self):
        return self._train_ds.in_shape

    @in_shape.setter
    def in_shape(self, value):
        self._train_ds.in_shape = value
        self._agent.in_shape = value
        self._update_dataloader()

    @property
    def cam_config(self):
        return self._train_ds.cam_config

    @cam_config.setter
    def cam_config(self, value):
        self._train_ds.cam_config = value
        self._update_dataloader()

    def prepare_data_dir(self):
        data_dir = str(Path(self._icfg["save_path"]) / "data")
        # Create random tmp dir
        if self._icfg["tmp_dir"] is not None:
            tmp_dir = str(Path(self._icfg["tmp_dir"]) / "askdagger")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_data_dir = tempfile.mkdtemp(dir=tmp_dir)
            # Copy data from data_dir to tmp_data_dir if data_dir exists
            if Path(data_dir).exists():
                shutil.copytree(data_dir, tmp_data_dir, dirs_exist_ok=True)
            self._data_dir = tmp_data_dir
            self._data_dir_persistent = data_dir
        else:
            self._data_dir = data_dir
            self._data_dir_persistent = data_dir

    def remove_temp_dir(self):
        if self._data_dir != self._data_dir_persistent:
            shutil.rmtree(self._data_dir)

    def visualize_action_projected(self, act, obs, font_scale=0.5, flip_x=False, flip_y=False):
        projected_img = obs[:, :, :3]
        projected_img = projected_img.transpose((1, 0, 2)).astype(np.uint8).copy()
        if flip_x:
            projected_img = cv2.flip(projected_img, 0)
        if flip_y:
            projected_img = cv2.flip(projected_img, 1)
        for pose_type in ["pose0", "pose1"]:
            pose = act[pose_type][0]
            pix = utils.xyz_to_pix(pose[:3], self.bounds, self.pix_size)
            if flip_x:
                pix = [pix[0], projected_img.shape[0] - pix[1]]
            if flip_y:
                pix = [projected_img.shape[1] - pix[0], pix[1]]
            projected_point = pix
            cv2.circle(projected_img, tuple(projected_point), 2, (0, 0, 0), -1)
            cv2.circle(projected_img, tuple(projected_point), 1, (255, 255, 255), -1)
            text = "Pick" if pose_type == "pose0" else "Place"
            projected_loc = (projected_point[0] + 5, projected_point[1])
            cv2.putText(
                projected_img,
                text,
                projected_loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                int(font_scale * 4),
                cv2.LINE_AA,
            )
            cv2.putText(
                projected_img,
                text,
                projected_loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                int(font_scale * 2),
                cv2.LINE_AA,
            )
        return projected_img

    def get_image(self, obs):
        return self._train_ds.get_image(obs)

    def visualize_action(self, act, obs, info, rotate_180=False, scale=2.0):
        color_data = obs["color"]
        depth_data = obs["depth"][0]
        img = color_data[0].copy().astype(np.uint8)
        projected_img = self._train_ds.get_image(obs)[:, :, :3]
        projected_img = projected_img.transpose((1, 0, 2)).astype(np.uint8).copy()
        projected_img = cv2.resize(projected_img, (int(projected_img.shape[1] * scale), int(projected_img.shape[0] * scale)))
        dummy_color = np.arange(color_data[0].shape[0] * color_data[0].shape[1]).reshape(
            color_data[0].shape[0], color_data[0].shape[1], 1
        )
        dummy_obs = {"color": (dummy_color,), "depth": (depth_data,)}
        _, colormaps = utils.reconstruct_heightmaps_dummy(
            dummy_obs["color"], dummy_obs["depth"], self.cam_config, self.bounds, self.pix_size
        )
        colormaps = colormaps[0][:, :, 0]
        window = [0, 1, -1, 2, -2]
        for pose_type in ["pose0", "pose1"]:
            pose = act[pose_type][0]
            pix = utils.xyz_to_pix(pose[:3], self.bounds, self.pix_size)
            point_idx = colormaps[pix]
            if point_idx == 0:
                for i in window:
                    for j in window:
                        pixely = np.clip(pix[0] + i, 0, colormaps.shape[0] - 1)
                        pixelx = np.clip(pix[1] + j, 0, colormaps.shape[1] - 1)
                        point_idx = colormaps[pixely, pixelx]
                        if point_idx > 0:
                            break
            point = np.unravel_index(point_idx, dummy_color.shape)[:2]
            point = np.flip(point)
            projected_point = int(pix[0] * scale), int(pix[1] * scale)
            cv2.circle(img, tuple(point), 4, (0, 0, 0), -1)
            cv2.circle(img, tuple(point), 2, (255, 255, 255), -1)
            cv2.circle(projected_img, tuple(projected_point), 2, (0, 0, 0), -1)
            cv2.circle(projected_img, tuple(projected_point), 1, (255, 255, 255), -1)
            text = "Pick" if pose_type == "pose0" else "Place"
            loc = (int(point[0] + 5), int(point[1]))
            projected_loc = (projected_point[0] + 5, projected_point[1])
            if rotate_180:
                rotation_matrix = cv2.getRotationMatrix2D(loc, 180, 1)
                text_img = np.zeros_like(img)
                cv2.putText(
                    text_img,
                    text,
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    text_img,
                    text,
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                text_img = cv2.warpAffine(text_img, rotation_matrix, (text_img.shape[1], text_img.shape[0]))
                img = cv2.add(img, text_img)
            else:
                cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                projected_img,
                text,
                projected_loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                projected_img,
                text,
                projected_loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img, projected_img

    def load_ckpt(self, ckpt):
        self._agent.load(ckpt)

    def update_stats(self, demo):
        self._r[-1] = demo["r"]
        self._stats["online_idx"].append(True)
        self._stats["queries"][-1] = (
            demo["r"] == askdagger_cliport.KNOWN_SUCCESS or demo["r"] == askdagger_cliport.KNOWN_FAILURE
        )
        self._stats["demos"].append(demo["demo"] is not None)
        if demo["relabeling_demo"] is not None:
            self._u.append(np.NaN)
            self._r.append(askdagger_cliport.UNKNOWN_RELABELING)
            self._k.append(self._stats["n_updates"])
            self._stats["demos"].append(True)
            self._stats["online_idx"].append(False)
        if demo["oracle_demo"]:
            self._stats["n_annotation"] += 1

    def update_rewards(self, done, novice_reward=np.NaN, system_reward=np.NaN):
        self._novice_rewards.append(novice_reward)
        self._system_rewards.append(system_reward)
        query = self._stats["queries"][-1]
        print(f"Query: {bool(query)}")
        self._stats["novice_success"] = np.append(self._stats["novice_success"], novice_reward > 0)
        self._stats["system_success"] = np.append(self._stats["system_success"], system_reward > 0)
        self._stats["tp"] = np.append(self._stats["tp"], not self._stats["novice_success"][-1] and query)
        self._stats["fp"] = np.append(self._stats["fp"], self._stats["novice_success"][-1] and query)
        self._stats["tn"] = np.append(self._stats["tn"], self._stats["novice_success"][-1] and not query)
        self._stats["fn"] = np.append(self._stats["fn"], not self._stats["novice_success"][-1] and not query)

        print(f"Reward system: {system_reward:.3f} | Reward novice: {novice_reward:.3f}")
        print(f"Total reward system: {np.sum(self._system_rewards)} | Total reward novice: {np.sum(self._novice_rewards)}")

        self._log_stats()

        if done:
            self._stats["episodes"].append(
                (np.sum(self._system_rewards), self._system_rewards, np.sum(self._novice_rewards), self._novice_rewards)
            )
            self._stats["system_rewards"] = np.append(self._stats["system_rewards"], np.sum(self._system_rewards))
            self._stats["novice_rewards"] = np.append(self._stats["novice_rewards"], np.sum(self._novice_rewards))
            self._stats["mean_reward"] = np.mean(self._stats["system_rewards"][-self._log_window :])
            mean_reward_novice = np.mean(self._stats["novice_rewards"][-self._log_window :])
            episode_count = len(self._stats["episodes"])
            print(f"Mean system: {self._stats['mean_reward']:.3f} | Mean agent: {mean_reward_novice:.3f}")
            if self._wandb_logger is not None:
                self._wandb_logger.log_metrics(
                    {
                        "reward_system": self._stats["mean_reward"],
                        "reward_novice": mean_reward_novice,
                        "episode": episode_count,
                    }
                )
            self._novice_rewards = []
            self._system_rewards = []

    def end_episode(self):
        if len(self._system_rewards) > 0:
            self._stats["episodes"].append(
                (np.sum(self._system_rewards), self._system_rewards, np.sum(self._novice_rewards), self._novice_rewards)
            )
            self._stats["system_rewards"] = np.append(self._stats["system_rewards"], np.sum(self._system_rewards))
            self._stats["novice_rewards"] = np.append(self._stats["novice_rewards"], np.sum(self._novice_rewards))
            self._stats["mean_reward"] = np.mean(self._stats["system_rewards"][-self._log_window :])
            mean_reward_novice = np.mean(self._stats["novice_rewards"][-self._log_window :])
            episode_count = len(self._stats["episodes"])
            print(f"Mean system: {self._stats['mean_reward']:.3f} | Mean agent: {mean_reward_novice:.3f}")
            if self._wandb_logger is not None:
                self._wandb_logger.log_metrics(
                    {
                        "reward_system": self._stats["mean_reward"],
                        "reward_novice": mean_reward_novice,
                        "episode": episode_count,
                    }
                )
            self._novice_rewards = []
            self._system_rewards = []

    def add_demo(self, seed, demo):
        self._train_ds.add(seed, demo)
        self._update_dataloader()
        self._stats["n_interactive"] += 1

    def act(self, obs, info):
        if self._gpu != 0:
            self._agent.to("cuda")
            self._agent.eval()
        with torch.no_grad():
            agent_act = self._agent.act(obs, info)

        # Quantify uncertainty
        self._stats["u_max_pick"] = max(self._stats["u_max_pick"], info["pick_uncertainty"])
        self._stats["u_max_place"] = max(self._stats["u_max_place"], info["place_uncertainty"])
        pick_uncertainty = info["pick_uncertainty"] / self._stats["u_max_pick"]
        place_uncertainty = info["place_uncertainty"] / self._stats["u_max_place"]
        self._u.append(np.mean([pick_uncertainty, place_uncertainty]))
        self._r.append(askdagger_cliport.UNKNOWN_ONLINE)
        self._k.append(self._stats["n_updates"])
        self._stats["queries"] = np.append(self._stats["queries"], False)
        self._stats["active_queries"] = np.append(self._stats["active_queries"], False)
        self._gamma.append(
            sag(
                self._u,
                self._r,
                self._k,
                s_des=self._s_desired,
                n_min=self._n_min,
                p_rand=self._random_rate,
            )
        )
        query = self._u[-1] >= self._gamma[-1]
        self._stats["active_queries"][-1] = query
        print(f"Uncertainty: {self._u[-1]:.6f} | Threshold: {self._gamma[-1]:.6f}")
        return agent_act, query

    def undo_last_action(self):
        self._u = self._u[:-1]
        self._r = self._r[:-1]
        self._k = self._k[:-1]
        self._gamma = self._gamma[:-1]
        self._stats["queries"] = self._stats["queries"][:-1]

    def save_checkpoint(self):
        if self._agent.trainer is not None:
            self._agent.trainer.save_checkpoint(
                os.path.join(self._checkpoint_path, f"interactive={self._stats['n_interactive']}.ckpt")
            )

    def update_model(self):
        if hasattr(self._train_ds, "__len__") and len(self._train_ds) > 0:
            self._stats["n_updates"] += 1
            total_samples = self._batch_size * self._max_steps
            self._agent.train_ds = self._train_dl
            self._agent.train()
            self._trainer.max_epochs = total_samples // len(self._train_ds) + 1
            self._trainer.fit(self._agent)
            self._trainer.current_epoch = 0
            self._trainer.global_step = 0

    def save_model(self):
        if self._last_checkpoint_path is not None and self._agent.trainer is not None:
            self._agent.trainer.save_checkpoint(self._last_checkpoint_path)

    def prioritize_replay(self):
        if self._pier:
            p = prioritize_sampling(self._u, self._r, self._k, self._b, lam=self._lambda)
            p = np.delete(p, np.where(np.asarray(self._stats["demos"]) == 0)[0])
            self._train_ds.p = p**self._alpha / np.sum(p**self._alpha).reshape(-1)
            self._train_ds.beta = min(
                self._beta0 + self._stats["n_interactive"] * (1 - self._beta0) / 100, 1.0
            )  # Linear annealing till 100 demos are collected
            self._train_ds.wmax = (len(self._train_ds.p) * np.min(self._train_ds.p)) ** -self._train_ds.beta
            self._update_dataloader()

    def save_results(self):
        if "mean_reward" not in self._stats:
            self._stats["mean_reward"] = 0
        # Save data from tmp_data_dir to data_dir_persistent
        if self._data_dir != self._data_dir_persistent:
            # Do rsync from tmp_data_dir to data_dir_persistent
            os.system(f"rsync -av --no-perms {self._data_dir}/ {self._data_dir_persistent}")

        # Save results in a json file.
        self._results[self._name] = {
            "episodes": self._stats["episodes"],
            "mean_reward": self._stats["mean_reward"],
            "interactive_demos": self._stats["n_interactive"],
            "last_seed": self._stats["seed"],
            "u": self._u,
            "r": self._r,
            "k": self._k,
            "gamma": self._gamma,
            "demos": self._stats["demos"],
            "u_max_pick": self._stats["u_max_pick"],
            "u_max_place": self._stats["u_max_place"],
            "novice_success": self._stats["novice_success"],
            "system_success": self._stats["system_success"],
            "n_annotation": self._stats["n_annotation"],
        }

        if self._save_results:
            # Load existing results
            if os.path.exists(self._save_json):
                with open(self._save_json, "r") as f:
                    existing_results = json.load(f)
                existing_results.update(self._results)
                self._results = existing_results
            with open(self._save_json, "w") as f:
                json.dump(self._results, f, indent=4, cls=utils.NumpyEncoder)

    def _filter_warnings(self):
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="`LightningModule.configure_optimizers` returned `None`",
        )  # We use our own optimizers
        warnings.filterwarnings(
            "ignore", ".*does not have many workers.*"
        )  # Number of workers is not the bottleneck so can be ignored

    def _update_dataloader(self):
        self._train_dl = DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            collate_fn=collate_fn,
        )

    def _log_stats(self):
        tp = np.asarray(self._stats["tp"])
        tn = np.asarray(self._stats["tn"])
        novice_success = np.asarray(self._stats["novice_success"])
        system_success = np.asarray(self._stats["system_success"])
        r = np.asarray(self._r)
        demos = np.asarray(self._stats["demos"])
        n_annotation = self._stats["n_annotation"]
        n_updates = self._stats["n_updates"]

        step = np.sum(r >= -1)
        queries = self._stats["queries"]
        query_rate = queries[-self._log_window :].mean()
        n_validation = (
            np.sum(np.logical_and(r == askdagger_cliport.KNOWN_SUCCESS, np.asarray(demos))) if self._validation_demos else 0
        )
        n_relabeling = np.sum(r == askdagger_cliport.UNKNOWN_RELABELING)
        novice_success_rate = novice_success[-self._log_window :].mean()
        system_success_rate = system_success[-self._log_window :].mean()
        novice_failure = np.logical_not(novice_success)
        failure_window = np.where(novice_failure)[0][-self._log_window :]
        # we calculate the sensitivity over the last log_window failures
        sensitivity = (
            np.sum(tp[failure_window]) / np.sum(novice_failure[failure_window])
            if np.sum(novice_failure[failure_window]) > 0
            else np.NaN
        )
        success_window = np.where(novice_success)[0][-self._log_window :]
        specificity = (
            np.sum(tn[success_window]) / np.sum(novice_success[success_window])
            if np.sum(novice_success[success_window]) > 0
            else np.NaN
        )
        uncertainty = self._u[-2] if np.isnan(self._u[-1]) else self._u[-1]
        if self._wandb_logger is not None:
            self._wandb_logger.log_metrics(
                {
                    "query_rate": query_rate,
                    "n_validation": n_validation,
                    "n_relabeling": n_relabeling,
                    "n_annotation": n_annotation,
                    "s_desired": self._s_desired,
                    "system_success_rate": system_success_rate,
                    "novice_success_rate": novice_success_rate,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "step": step,
                    "threshold": self._gamma[-1],
                    "n_updates": n_updates,
                    "uncertainty": uncertainty,
                }
            )

    def _load_stats(self):
        # Load existing results.
        loaded_results = False
        self._stats = {}
        if os.path.exists(self._save_json):
            with open(self._save_json, "r") as f:
                self._results = json.load(f)
            if self._name in self._results:
                loaded_results = True
                self._stats["n_interactive"] = self._results[self._name]["interactive_demos"]
                self._stats["n_annotation"] = self._results[self._name]["n_annotation"]
                self._u = self._results[self._name]["u"]
                self._r = self._results[self._name]["r"]
                self._k = self._results[self._name]["k"]
                self._gamma = self._results[self._name]["gamma"]
                self._stats["demos"] = self._results[self._name]["demos"]
                self._stats["last_seed"] = self._results[self._name]["last_seed"]
                self._stats["episodes"] = self._results[self._name]["episodes"]
                self._stats["u_max_pick"] = self._results[self._name]["u_max_pick"]
                self._stats["u_max_place"] = self._results[self._name]["u_max_place"]
                self._stats["novice_success"] = self._results[self._name]["novice_success"]
                self._stats["system_success"] = self._results[self._name]["system_success"]
                self._stats["novice_rewards"] = np.asarray([episode[2] for episode in self._stats["episodes"]])
                self._stats["system_rewards"] = np.asarray([episode[0] for episode in self._stats["episodes"]])
                self._stats["online_idx"] = list(np.asarray(self._r) >= -1)
                self._stats["queries"] = np.logical_or(np.asarray(self._r) == -1, np.asarray(self._r) == 1)[
                    self._stats["online_idx"]
                ]
                self._stats["active_queries"] = np.asarray(self._u)[self._stats["online_idx"]] >= np.asarray(self._gamma)
                self._stats["tn"] = np.logical_and(
                    np.asarray(self._stats["novice_success"]), np.logical_not(self._stats["queries"])
                )
                self._stats["fp"] = np.logical_and(np.asarray(self._stats["novice_success"]), self._stats["queries"])
                self._stats["tp"] = np.logical_and(np.logical_not(self._stats["novice_success"]), self._stats["queries"])
                self._stats["fn"] = np.logical_and(
                    np.logical_not(self._stats["novice_success"]), np.logical_not(self._stats["queries"])
                )
        # Initialize results.
        if not loaded_results:
            self._results = {}
            self._stats["n_interactive"] = 0
            self._stats["n_annotation"] = 0
            self._stats["last_seed"] = None
            self._stats["episodes"] = []
            self._u = []
            self._r = []
            self._k = []
            self._gamma = []
            self._stats["demos"] = []
            self._stats["novice_success"] = []
            self._stats["system_success"] = []
            self._stats["novice_rewards"] = np.asarray([])
            self._stats["system_rewards"] = np.asarray([])
            self._stats["queries"] = np.asarray([])
            self._stats["active_queries"] = np.asarray([])
            self._stats["tn"] = np.asarray([])
            self._stats["fp"] = np.asarray([])
            self._stats["tp"] = np.asarray([])
            self._stats["fn"] = np.asarray([])
            self._stats["u_max_pick"] = 0
            self._stats["u_max_place"] = 0
            self._stats["mean_reward"] = 0
            self._stats["online_idx"] = []

    def _initialize_stats(self):
        if len(self._k) == 0 and self._train_ds.n_demos > 0 or not self._save_results:
            # Initialize stats in case of BC training
            n_pairs = int(np.sum(self._train_ds.eps_lens))
            self._k = [0] * n_pairs
            self._u = [np.NaN] * n_pairs
            self._r = [askdagger_cliport.UNKNOWN_OFFLINE] * n_pairs
            self._stats["demos"] = [True] * n_pairs
            self._stats["n_annotation"] = n_pairs
        else:
            assert np.sum(self._stats["demos"]) == np.sum(
                self._train_ds.eps_lens
            ), f"{np.sum(self._stats['demos'])} != {np.sum(self._train_ds.eps_lens)} | {len(self._stats['demos'])} != {len(self._train_ds.eps_lens)}"
        self._stats["n_updates"] = self._k[-1] if len(self._k) > 0 else 0

    def _get_latest_checkpoint(self):
        # Get checkpoint
        self._checkpoint_path = os.path.join(self._save_path, "checkpoints")
        self._last_checkpoint_path = os.path.join(self._checkpoint_path, "last.ckpt")
        last_checkpoint = (
            self._last_checkpoint_path if os.path.exists(self._last_checkpoint_path) and self._load_from_last_ckpt else None
        )
        if last_checkpoint is not None:
            print(f"Loading from last checkpoint: {self._last_checkpoint_path}")
            self._model_file = self._last_checkpoint_path
        elif self._train_demos == 0:
            print("Loading from scratch.")
            self._model_file = None
        else:
            checkpoints = sorted([c for c in os.listdir(self._model_path) if "steps=" in c])
            ckpts_steps = [int(c.split("=")[1].split(".")[0]) for c in checkpoints]
            ckpt = checkpoints[ckpts_steps.index(self._train_steps)]
            print(f"Loading: {str(ckpt)}")
            self._model_file = os.path.join(self._model_path, ckpt)
            # Check if checkpoint from offline training exists
        if self._train_demos > 0:
            assert os.path.exists(self._model_file) and os.path.isfile(
                self._model_file
            ), f"Checkpoint not found: {self._model_file}"

    def _get_logger(self):
        self._wandb_logger = None
        if self._logging:
            interactive_config = (
                OmegaConf.to_container(self._icfg, resolve=True) if isinstance(self._icfg, Container) else self._icfg
            )
            train_config = OmegaConf.to_container(self._tcfg, resolve=True) if isinstance(self._tcfg, Container) else self._tcfg
            config = {"train": {**train_config}, "interactive": {**interactive_config}}
            train_interactive_task_name = "".join([l[0] for l in self._train_interactive_task.split("-")])
            task_name = "".join([l[0] for l in self._train_task.split("-")])
            task_name = (
                train_interactive_task_name
                if train_interactive_task_name == task_name
                else f"{task_name}-{train_interactive_task_name}"
            )
            train_demos_name = f"-bc{self._train_demos}" if self._train_demos > 0 else ""
            log_name = f"{self._icfg['agent']}-{self._iteration}-{task_name}{train_demos_name}-fier={self._relabling_demos}-pier={self._pier}-sd={self._s_desired}"
            self._wandb_logger = WandbLogger(project="askdagger-cliport", name=log_name)
            # Using old versions of wandb and pytorch-lightning
            # There is a bug related to communication timeouts, see https://github.com/wandb/wandb/issues/1409
            # This is a workaround (to prevent problems when performing cluster runs)
            while True:
                try:
                    self._wandb_logger.log_hyperparams(config)
                    break
                except Exception as e:
                    print(e)
                    print("Retrying")
                    time.sleep(10)

    def _get_save_json(self):
        # Create data_dir if it doesn't exist, plus action, color, depth, info and reward dirs.
        os.makedirs(os.path.join(self._data_dir, "{}-train".format(self._train_task)), exist_ok=True)
        subdirs = ["action", "color", "depth", "info", "reward"]
        for subdir in subdirs:
            os.makedirs(os.path.join(os.path.join(self._data_dir, "{}-train".format(self._train_task)), subdir), exist_ok=True)

        # Create symbolic link to each demo data file in the subdirs.
        if self._train_demos > 0:
            for subdir in subdirs:
                # Create symbolic link to first [self._train_demos] files in each subdir if they don't exist.
                for file in os.listdir(
                    os.path.join(os.path.join(self._train_data_dir, "{}-train".format(self._train_task)), subdir)
                ):
                    if int(file.split("-")[0]) < self._train_demos:
                        if not os.path.exists(
                            os.path.join(os.path.join(self._data_dir, "{}-train".format(self._train_task)), subdir, file)
                        ):
                            os.symlink(
                                os.path.join(
                                    os.path.join(self._train_data_dir, "{}-train".format(self._train_task)), subdir, file
                                ),
                                os.path.join(os.path.join(self._data_dir, "{}-train".format(self._train_task)), subdir, file),
                            )
        self._name = "{}-{}-n{}".format(self._train_interactive_task, self._icfg["agent"], self._train_demos)
        # Save path for results.
        json_name = f"results-{self._mode}.json"
        print(f"Save path for results: {self._save_path}")
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        self._save_json = os.path.join(self._save_path, f"{self._name}-{json_name}")

    def _initialize_datasets(self):
        total_demos = self._train_demos + self._stats["n_interactive"]
        # Initialize datasets
        self._train_ds = RavensDataset(
            os.path.join(self._data_dir, "{}-train".format(self._train_task)),
            self._tcfg,
            n_demos=total_demos,
            augment=True,
            bias_compensation=self._bias_compensation,
        )
        self._update_dataloader()

    def _check_seed(self):
        if not self._save_results:
            self._stats["seed"] = self._train_ds.max_seed
            return
        if (
            self._name in self._results
            and self._train_ds.n_demos == self._train_ds.n_episodes
            and self._stats["n_interactive"] >= self._interactive_demos
        ):
            print(f"Skipping because of existing results for {self._name}.")
            sys.exit()
        # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
        self._stats["seed"] = self._train_ds.max_seed
        if self._stats["seed"] < 0:
            self._stats["seed"] = -2 + self._iteration * 10000
        # Check if seeds match, there could be a mismatch if training was interrupted
        if self._train_ds.n_episodes > self._train_ds.n_demos:
            print(f"Training interrupted. Returning to seed {self._stats['seed']}.")
            self._train_ds.n_demos += 1
            self._train_ds.remove(self._train_ds.max_seed, self._train_ds.n_episodes - 1)
            self._check_seed()
            return
        if self._stats["last_seed"] is not None:
            if self._train_ds.n_episodes == self._train_ds.n_demos and self._stats["seed"] == self._stats["last_seed"]:
                print(f"Continuing from seed {self._stats['seed']}.")
            elif self._train_ds.n_episodes == self._train_ds.n_demos and self._stats["last_seed"] > self._stats["seed"]:
                # In this case, there was no demo collected in the last episode
                print(f"Continuing from seed {self._stats['last_seed']}.")
            else:
                print("Seeds and demos do not match. Skipping.")
                raise ValueError
            self._stats["seed"] = self._stats["last_seed"]
        elif self._train_ds.n_episodes == self._train_ds.n_demos and self._train_ds.n_demos == self._train_demos:
            print(f"Start interactive training. Seed: {self._stats['seed']}")
        else:
            print("Seeds and demos do not match. Skipping.")
            raise ValueError
