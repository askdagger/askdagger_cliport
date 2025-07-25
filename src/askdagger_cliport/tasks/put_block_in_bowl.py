# credit: https://github.com/cliport/cliport

"""Put Blocks in Bowl Task."""

import numpy as np
from askdagger_cliport.tasks.task import Task
from askdagger_cliport.utils import utils

import random
import pybullet as p


class PutBlockInBowlUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} blocks in a {place} bowl"
        self.task_completed_desc = "done placing blocks in bowls."
        self.bowls = {}
        self.blocks = {}
        self._colors = None

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        self._colors = colors

    def reset(self, env):
        super().reset(env)
        if self._colors is None:
            self._colors = utils.TRAIN_COLORS if self.mode == "train" else utils.EVAL_COLORS
        self.bowls = {}
        self.blocks = {}
        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = "bowl/bowl.urdf"
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            bowl_poses.append(bowl_pose)
            self.bowls[bowl_id] = utils.get_color_name(colors[1])

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = "stacking/block.urdf"
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))
            self.blocks[block_id] = utils.get_color_name(colors[0])

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))), bowl_poses, False, True, "pose", None, 1))
        self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0], place=selected_color_names[1]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            color_name = utils.get_color_name(color)
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1
            if is_block:
                self.blocks[obj_id] = color_name
            else:
                self.bowls[obj_id] = color_name

    def get_colors(self):
        return self.colors

    def relabel(self, matched_objects, matched_objects_prev):
        # Oracle uses perfect RGB-D orthographic images and segmentation masks.
        her_lang_goal = None
        occupied_bowls_prev = []
        for _, bowl in matched_objects_prev.items():
            occupied_bowls_prev.append(bowl)
        for block_id, bowl_id in matched_objects.items():
            if bowl_id not in occupied_bowls_prev:
                if block_id not in matched_objects_prev:
                    her_lang_goal = self.lang_template.format(pick=self.blocks[block_id], place=self.bowls[bowl_id])
                elif (
                    matched_objects_prev[block_id] != bowl_id
                    and self.bowls[bowl_id] != self.bowls[matched_objects_prev[block_id]]
                ):
                    her_lang_goal = self.lang_template.format(pick=self.blocks[block_id], place=self.bowls[bowl_id])
        return her_lang_goal

    def get_matched_objects(self):
        blocks = []
        for block in self.blocks.keys():
            blocks.append((block, (0, None)))

        # Get target poses.
        targs = []
        target_ids = []
        for bowl in self.bowls.keys():
            targs.append(p.getBasePositionAndOrientation(bowl))
            target_ids.append(bowl)

        matches = np.ones((len(blocks), len(self.bowls)))

        # Match objects to targets without replacement.
        matched_objects = {}

        # Modify a copy of the match matrix.
        matches = matches.copy()

        # Ignore already matched objects.
        for i in range(len(blocks)):
            object_id, (symmetry, _) = blocks[i]
            pose = p.getBasePositionAndOrientation(object_id)
            targets_i = np.argwhere(matches[i, :]).reshape(-1)
            for j in targets_i:
                if self.is_match(pose, targs[j], symmetry) and pose[0][2] > targs[j][0][2] + 0.005:
                    matched_objects[object_id] = target_ids[j]
        return matched_objects


class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()
        self._colors = utils.TRAIN_COLORS


class PutBlockInBowlFull(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()
        self._colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
