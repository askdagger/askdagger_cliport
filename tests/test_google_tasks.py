# credit: https://github.com/cliport/cliport

"""Integration tests for google objects tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from askdagger_cliport import tasks
from askdagger_cliport.environments import environment
import askdagger_cliport

ASSETS_PATH = askdagger_cliport.__path__[0] + "/environments/assets/"


class TaskTest(parameterized.TestCase):
    def _create_env(self):
        assets_root = ASSETS_PATH
        env = environment.Environment(assets_root)
        env.seed(0)
        return env

    def _run_oracle_in_env(self, env):
        agent = env.task.oracle(env)
        obs, info = env.reset()
        info = None
        done = False
        for _ in range(10):
            act = agent.act(obs, info)
            obs, _, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            if done:
                break

    @parameterized.named_parameters(
        (
            "PackingSeenGoogleObjectsSeq",
            tasks.PackingSeenGoogleObjectsSeq(),
        ),
        (
            "PackingUnseenGoogleObjectsSeq",
            tasks.PackingUnseenGoogleObjectsSeq(),
        ),
        (
            "PackingSeenGoogleObjectsGroup",
            tasks.PackingSeenGoogleObjectsGroup(),
        ),
        (
            "PackingUnseenGoogleObjectsGroup",
            tasks.PackingUnseenGoogleObjectsGroup(),
        ),
        (
            "PackingSeenGoogleObjectsOriginalSeq",
            tasks.PackingSeenGoogleObjectsOriginalSeq(),
        ),
        (
            "PackingUnseenGoogleObjectsOriginalSeq",
            tasks.PackingUnseenGoogleObjectsOriginalSeq(),
        ),
        (
            "PackingSeenGoogleObjectsOriginalGroup",
            tasks.PackingSeenGoogleObjectsOriginalGroup(),
        ),
        (
            "PackingUnseenGoogleObjectsOriginalGroup",
            tasks.PackingUnseenGoogleObjectsOriginalGroup(),
        ),
    )
    def test_all_tasks(self, dvnets_task):
        env = self._create_env()
        env.set_task(dvnets_task)
        self._run_oracle_in_env(env)


if __name__ == "__main__":
    absltest.main()
