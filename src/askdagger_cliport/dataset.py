# credit: https://github.com/cliport/cliport

"""Image dataset."""

import os
import pickle
import warnings
import torch

import numpy as np
from torch.utils.data import Dataset

from askdagger_cliport import tasks
from askdagger_cliport.tasks import cameras
from askdagger_cliport.utils import utils


# See transporter.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], list):
        return [collate_fn([d[i] for d in batch]) for i in range(len(batch[0]))]
    elif isinstance(batch[0], tuple):
        return tuple(collate_fn([d[i] for d in batch]) for i in range(len(batch[0])))
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)

    else:
        return batch


class RavensDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, n_demos=0, augment=False, bias_compensation="step"):
        """A simple RGB-D image dataset."""
        self._path = path

        self.cfg = cfg
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg["dataset"]["images"]
        self.cache = self.cfg["dataset"]["cache"]
        self.n_demos = n_demos
        self.augment = augment
        self._p = None
        self._beta = None
        self._wmax = None
        self._bias_compensation = bias_compensation

        self.aug_theta_sigma = (
            self.cfg["dataset"]["augment"]["theta_sigma"] if "augment" in self.cfg["dataset"] else 60
        )  # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # Track existing dataset if it exists.
        action_path = os.path.join(self._path, "action")
        if os.path.exists(action_path):
            for fname in sorted(os.listdir(action_path)):
                if ".pkl" in fname:
                    seed = int(fname[(fname.find("-") + 1) : -4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

        if self.n_demos > 0:
            self.images = self.cfg["dataset"]["images"]
            self.cache = self.cfg["dataset"]["cache"]

            # Check if there are sufficient demos in the dataset
            if self.n_demos > self.n_episodes:
                raise Exception(
                    f"Requested training on {self.n_demos} demos, but only {self.n_episodes} demos exist in the dataset path: {self._path}."
                )

            episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
            self.set(episodes)
        self.eps_lens = self.get_episode_lens()

    @property
    def p(self):
        """The prioritized replay distribution.
        If None, uniform sampling is used.
        """
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    @property
    def beta(self):
        """The prioritized replay bias compensation exponent."""
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def wmax(self):
        """The prioritized replay maximum weight."""
        return self._wmax

    @wmax.setter
    def wmax(self, wmax):
        self._wmax = wmax

    def get_max_seed(self):
        max_seed = -1
        action_path = os.path.join(self._path, "action")
        if os.path.exists(action_path):
            for fname in sorted(os.listdir(action_path)):
                if ".pkl" in fname:
                    seed = int(fname[(fname.find("-") + 1) : -4])
                    max_seed = max(max_seed, seed)
        return max_seed

    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        for obs, act, r, i in episode:
            if type(obs) is dict:
                color.append(obs["color"])
                depth.append(obs["depth"])
            else:
                depth.append(obs)
            action.append(act)
            reward.append(r)
            info.append(i)

        color = np.uint8(color)
        depth = np.float32(depth)

        def dump(data, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f"{self.n_episodes:06d}-{seed}.pkl"  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), "wb") as f:
                pickle.dump(data, f)

        dump(color, "color")
        dump(depth, "depth")
        dump(action, "action")
        dump(reward, "reward")
        dump(info, "info")

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)
        self.n_demos += 1

        np.random.seed(self.max_seed)
        episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
        self.set(episodes)

        self.eps_lens = self.eps_lens + [len(episode) - 1]

    def remove(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """

        def delete_episode(episode, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                raise Exception(f"Path {field_path} does not exist.")
            fname = f"{episode:06d}-{seed}.pkl"  # -{len(episode):06d}
            os.remove(os.path.join(field_path, fname))

        delete_episode(episode, "color")
        delete_episode(episode, "depth")
        delete_episode(episode, "action")
        delete_episode(episode, "reward")
        delete_episode(episode, "info")

        self.n_episodes -= 1
        self.n_demos -= 1
        self.eps_lens = self.eps_lens[:-1]
        self.max_seed = self.get_max_seed()

        if self.n_demos > 1:
            np.random.seed(self.max_seed)
            episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
            self.set(episodes)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)
            data = pickle.load(open(os.path.join(path, fname), "rb"))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, "action")
        for fname in sorted(os.listdir(path)):
            if f"{episode_id:06d}" in fname:
                seed = int(fname[(fname.find("-") + 1) : -4])

                # Load data.
                color = load_field(episode_id, "color", fname)
                depth = load_field(episode_id, "depth", fname)
                action = load_field(episode_id, "action", fname)
                reward = load_field(episode_id, "reward", fname)
                info = load_field(episode_id, "info", fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    if not images:
                        obs = {}
                    elif len(color) > 0:
                        obs = {"color": color[i], "depth": depth[i]}
                    else:
                        obs = depth[i]
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def get_label(self, p, theta, inp_img, n_rotations):
        theta_i = theta / (2 * np.pi / n_rotations)
        theta_i = np.int32(np.round(theta_i)) % n_rotations
        label_size = inp_img.shape[:2] + (n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(-1)
        label = torch.from_numpy(label.copy()).to(dtype=torch.float)

        return label

    def get_image(self, obs, cam_config=None):
        """Stack color and height images image."""
        if cam_config is None:
            cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(obs, cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap, hmap[Ellipsis, None], hmap[Ellipsis, None], hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def process_sample(self, datum, augment=True):
        # Get training labels from data sample.
        (obs, act, _, info) = datum
        if type(obs) is dict:
            img = self.get_image(obs)
        else:
            img = obs

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        if act:
            p0_xyz, p0_xyzw = act["pose0"]
            p1_xyz, p1_xyzw = act["pose1"]
            p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p0 = np.array(p0, dtype=np.int32)
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p1 = np.array(p1, dtype=np.int32)
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0

        # Data augmentation.
        if augment:
            img, _, (p0, p1), perturb_params = utils.perturb(img, [p0, p1], theta_sigma=self.aug_theta_sigma)

        attn_label, transport_label = None, None
        if act:
            attn_label = self.get_label(p0, p0_theta, img, 1)
            transport_label = self.get_label(p1, p1_theta, img, self.cfg["train"]["n_rotations"])

        sample = {
            "img": torch.from_numpy(img).to(dtype=torch.float),
            "p0": torch.from_numpy(p0.copy()) if p0 is not None else None,
            "attn_label": attn_label,
            "transport_label": transport_label,
        }
        # Add language goal if available.
        if "lang_goal" not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and "lang_goal" in info:
            sample["lang_goal"] = info["lang_goal"]
        else:
            sample["lang_goal"] = "task completed."
        return sample

    def process_goal(self, goal, perturb_params):
        # Get goal sample.
        (obs, act, _, info) = goal
        if type(obs) is dict:
            img = self.get_image(obs)
        else:
            img = obs

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        # Data augmentation with specific params.
        if perturb_params:
            img = utils.apply_perturbation(img, perturb_params)

        sample = {"img": img, "p0": p0, "p0_theta": p0_theta, "p1": p1, "p1_theta": p1_theta, "perturb_params": perturb_params}

        # Add language goal if available.
        if "lang_goal" not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and "lang_goal" in info:
            sample["lang_goal"] = info["lang_goal"]
        else:
            sample["lang_goal"] = "task completed."

        return sample

    def __len__(self):
        return len(self.sample_set)

    def __getitem__(self, idx):
        if self._p is not None:
            assert len(self._p) == np.sum(self.eps_lens), f"{len(self._p)} != {np.sum(self.eps_lens)}"
            sample_id = np.random.choice(len(self._p), p=self._p)
            episode_id, step_id = self.get_episode_and_step_id(sample_id)
            episode, _ = self.load(episode_id, self.images, self.cache)
            sample = episode[step_id]
            sample = self.process_sample(sample, augment=self.augment)

            sample["idx"] = sample_id

            if self._bias_compensation:
                p_sampling = self._p[sample_id]
                p_data = 1 / len(self._p)
                sample["w"] = torch.Tensor([(p_data / p_sampling) ** self._beta / self._wmax])
            else:
                sample["w"] = torch.Tensor([1])
            return sample
        else:
            # Choose random episode.
            if len(self.sample_set) > 0:
                episode_id = np.random.choice(self.sample_set)
            else:
                episode_id = np.random.choice(range(self.n_episodes))
            episode, _ = self.load(episode_id, self.images, self.cache)

            # Return random observation action pair (and goal) from episode.
            if len(episode) <= 1:
                print("Episode too short: {}".format(episode_id))
            i = np.random.choice(range(len(episode) - 1))
            sample = episode[i]

            # Process sample.
            sample = self.process_sample(sample, augment=self.augment)
            sample["idx"] = self.get_sample_id(episode_id, i)
            return sample

    def get_episode_lens(self):
        episode_lens = []
        for episode_id in range(self.n_episodes):
            episode, _ = self.load(episode_id, self.images, self.cache)
            episode_lens.append(len(episode) - 1)
        return episode_lens

    def get_episode_and_step_id(self, sample_idx):
        id = 0
        for episode, eps_len in enumerate(self.eps_lens):
            if sample_idx < id + eps_len:
                return episode, sample_idx - id
            id += eps_len
        raise Exception(f"Sample id {sample_idx} not found. Dataset size: {id-1}")

    def get_sample_id(self, episode_id, step_id):
        id = 0
        for episode, eps_len in enumerate(self.eps_lens):
            if episode == episode_id:
                return id + step_id
            id += eps_len
        raise Exception(f"Episode id {episode_id} not found. Dataset size: {id-1}")
