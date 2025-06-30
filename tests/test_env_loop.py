from promptgym.dataio.dataset import PromptTaskDataset
from promptgym.envs.prompt_opt_env import PromptOptEnv


def test_env_loop():
    ds = PromptTaskDataset().permute(0)
    env = PromptOptEnv(ds, budget=5)
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        obs, r, done, _ = env.step(0)
        steps += 1
    assert env.step_n == steps

