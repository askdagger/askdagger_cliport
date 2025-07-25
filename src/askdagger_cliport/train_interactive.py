"""main interactive training script."""

import hydra
import numpy as np
import pybullet as p

from copy import deepcopy
from typing import Dict, Any

import askdagger_cliport
from askdagger_cliport.interactive_agent import InteractiveAgent
from askdagger_cliport import tasks
from askdagger_cliport.utils import utils
from askdagger_cliport.environments.environment import Environment


def train_interactive(icfg):
    # Initialize environment and task.
    env = Environment(icfg["assets_root"], disp=icfg["disp"], shared_memory=icfg["shared_memory"], hz=480)

    # Set task
    task = tasks.names[icfg["train_interactive_task"]]()
    task.mode = "train"
    env.set_task(task)

    # Agent
    interactive_agent = InteractiveAgent(icfg=icfg, mode="train")
    seed = interactive_agent.seed

    while interactive_agent.n_interactive < icfg["interactive_demos"]:
        seed += 2
        interactive_agent.seed = seed

        # Set seeds.
        utils.set_seed(seed)
        print("\nInteractive demos: {}/{} | Seed: {}".format(interactive_agent.n_interactive, icfg["interactive_demos"], seed))

        env.seed(seed)
        obs, info = env.reset()
        D_i = []
        reward = 0
        done = False
        while not done:
            lang_goal = info["lang_goal"]
            print(f"Lang Goal: {lang_goal}")

            # Save model every [save_every] demos.
            if icfg["save_model"] and interactive_agent.n_interactive % icfg["save_every"] == 0:
                interactive_agent.save_checkpoint()

            # Get agent action.
            if len(obs["color"]) == 0:
                obs = env.get_obs()  # Only rendering when the agent acts, since rendering is slow.
            agent_act, query = interactive_agent.act(obs, info)

            # Query oracle if uncertainty is high.
            env, obs, reward, terminated, truncated, info, demo = collect_demo(
                obs=obs,
                reward=reward,
                info=info,
                env=env,
                agent_act=agent_act,
                query=query,
                interactive_agent=interactive_agent,
                relabeling_demos=icfg["relabeling_demos"],
                validation_demos=icfg["validation_demos"],
                p_rand=icfg["train_interactive"]["p_rand"],
            )
            done = terminated or truncated
            interactive_agent.update_stats(demo)
            # Rewards are privileged information, so we update it separately.
            # It is only used for logging purposes
            interactive_agent.update_rewards(done, novice_reward=demo["novice_reward"], system_reward=reward)

            if demo["demo"] is not None:
                D_i.append(demo["demo"])
            if demo["relabeling_demo"] is not None:
                D_i.append(demo["relabeling_demo"])

            if done:
                break

        save = icfg["save_model"] and interactive_agent.episode_count % 20 == 0
        if len(D_i) > 0:
            if len(obs["color"]) == 0:
                obs = interactive_agent.get_image(env.get_obs())
            D_i.append((obs, None, reward, info))
            interactive_agent.add_demo(seed, D_i)
            interactive_agent.prioritize_replay()
        if icfg["update_every_episode"] or len(D_i) > 1:
            interactive_agent.update_model()
        if save:
            interactive_agent.save_model()
            interactive_agent.save_results()
    if icfg["save_model"]:
        interactive_agent.save_model()
        interactive_agent.save_results()
        interactive_agent.save_checkpoint()
    env.shutdown()
    interactive_agent.remove_temp_dir()
    del interactive_agent
    del env


@hydra.main(config_path="./cfg", config_name="train_interactive")
def main(icfg):
    train_interactive(icfg)


def collect_demo(
    obs: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
    env: Environment,
    agent_act: Dict[str, Any],
    query: bool,
    interactive_agent: InteractiveAgent,
    relabeling_demos: bool = True,
    validation_demos: bool = True,
    p_rand: float = 0.0,
):
    """Collect a demonstration from the oracle or the agent.
    Args:
        obs: The previous observation.
        reward: The previous reward.
        info: The previous info dict.
        env: The environment.
        agent_act: The agent's action.
        query: Whether to query the oracle.
        interactive_agent: Instance of interactive agent.
        relabeling_demos: Whether to collect relabeling demos.
        validation_demos: Whether to collect validation demos.
        p_rand: The probability of querying randomly.

    Returns:
        env: The environment.
        obs: The current observation.
        reward: The current reward.
        terminated: Whether the episode is terminated.
        truncated: Whether the episode is truncated.
        info: The current info dict.
        demo_info: The demo info dict.
    """
    matched_objects_prev = env.task.get_matched_objects()
    state_id = p.saveState()
    env_copy = deepcopy(env)
    obs_prev = deepcopy(obs)
    info_prev = deepcopy(info)
    reward_prev = deepcopy(reward)
    max_reward = env_copy.task.goals[0][-1] / len(env_copy.task.goals[0][0])
    obs, reward, terminated, truncated, info = env_copy.step(agent_act, render=False)

    # Keep track of everything that happens in the demo.
    demo = dict(
        r=askdagger_cliport.UNKNOWN_ONLINE, novice_reward=deepcopy(reward), oracle_demo=False, demo=None, relabeling_demo=None
    )

    if np.random.rand() < p_rand or query:
        demo.update(
            r=askdagger_cliport.KNOWN_SUCCESS if np.abs(max_reward - reward) < 0.01 else askdagger_cliport.KNOWN_FAILURE
        )
        if np.abs(max_reward - reward) < 0.01:
            if validation_demos:
                print("Validation demo.")
                obs_prev = interactive_agent.get_image(obs_prev)
                demo.update(demo=(obs_prev, agent_act, reward_prev, info_prev))
            elif query:
                print("Annotation demo.")
                p.restoreState(state_id)
                oracle_agent = env.task.oracle(env)
                oracle_act = oracle_agent.act(obs_prev, info_prev)
                obs, reward, terminated, truncated, info = env.step(oracle_act, render=False)
                env_copy = env
                if np.abs(max_reward - reward) < 0.01:
                    # If the oracle action is successful, we add the oracle demo to the dataset.
                    obs_prev = interactive_agent.get_image(obs_prev)
                    demo.update(demo=(obs_prev, oracle_act, reward_prev, info_prev), oracle_demo=True)
        else:
            if relabeling_demos:
                matched_objects = env_copy.task.get_matched_objects()
                relabeling_lang_goal = env_copy.task.relabel(matched_objects, matched_objects_prev)
                if relabeling_lang_goal is not None:
                    print(f"Relabeling demo: {relabeling_lang_goal}")
                    relabeling_info = deepcopy(info_prev)
                    relabeling_info["lang_goal"] = relabeling_lang_goal
                    obs_prev = interactive_agent.get_image(obs_prev)
                    demo.update(relabeling_demo=(obs_prev, agent_act, reward_prev, relabeling_info))
            if query:
                print("Annotation demo.")
                # In the case of querying the oracle, the agent action will not be executed.
                # In simulation we only execute the novice action to evaluate the success of the action.
                # In reality, the human can predict the success of the action without executing it.
                p.restoreState(state_id)
                oracle_agent = env.task.oracle(env)
                oracle_act = oracle_agent.act(obs_prev, info_prev)
                obs, reward, terminated, truncated, info = env.step(oracle_act, render=False)
                env_copy = env
                if np.abs(max_reward - reward) < 0.01:
                    # If the oracle action is successful, we add the oracle demo to the dataset.
                    if isinstance(obs_prev, dict):
                        obs_prev = interactive_agent.get_image(obs_prev)
                    demo.update(demo=(obs_prev, oracle_act, reward_prev, info_prev), oracle_demo=True)
                else:
                    print("Oracle failed.")
    return env_copy, obs, reward, terminated, truncated, info, demo


if __name__ == "__main__":
    main()
