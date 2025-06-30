import pandas as pd
from ..dataio.dataset import PromptTaskDataset


def compute_metrics(log_df: pd.DataFrame, seed: int) -> dict:
    ds = PromptTaskDataset().permute(seed)
    regrets = []
    cum_rewards = []
    total = 0
    for _, row in log_df.iterrows():
        task = row['task']
        best_reward = ds.reward(row['agent_best'], task)
        optimal = ds.mat[:, task].max()
        regrets.append(optimal - best_reward)
        total += row['reward']
        cum_rewards.append(total)
    return {
        'regret': regrets,
        'cumulative_reward': cum_rewards,
    }

