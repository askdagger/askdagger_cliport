"""main interactive training script."""

import hydra
import numpy as np

from askdagger_cliport.interactive_agent import InteractiveAgent
from askdagger_cliport import tasks
from askdagger_cliport.utils import utils
from askdagger_cliport.environments.environment import Environment
from askdagger_cliport.train_interactive import collect_demo


def train_interactive(icfg, trial=None):
    # Initialize environment and task.
    env = Environment(icfg["assets_root"], disp=icfg["disp"], shared_memory=icfg["shared_memory"], hz=480)

    # Agent
    interactive_agent = InteractiveAgent(icfg=icfg, mode="train")
    seed = interactive_agent.seed

    # Task
    train_tasks = ["packing-seen-shapes", "packing-unseen-shapes", "packing-seen-google-objects-seq"]
    task_idx = interactive_agent.n_interactive // 150
    task = tasks.names[train_tasks[task_idx]]()
    task.mode = "test"
    env.set_task(task)
    print(f"Task: {train_tasks[task_idx]}")

    while interactive_agent.n_interactive < icfg["interactive_demos"]:
        if interactive_agent.n_interactive % 150 == 0:
            task_idx = interactive_agent.n_interactive // 150
            task = tasks.names[train_tasks[task_idx]]()
            task.mode = "test"
            env.set_task(task)
            print(f"Switching to task: {train_tasks[task_idx]}")

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
    if trial is not None:
        return np.mean(interactive_agent._stats["novice_rewards"][-interactive_agent._log_window :])


@hydra.main(config_path="./cfg", config_name="train_interactive")
def main(icfg):
    train_interactive(icfg)


if __name__ == "__main__":
    main()
