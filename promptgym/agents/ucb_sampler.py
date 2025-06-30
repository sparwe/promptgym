import math
from .base import BaseAgent


class UCBSampler(BaseAgent):
    """Simple UCB agent that sequentially explores tasks for each arm."""

    def __init__(self, config=None):
        super().__init__(config)
        self.num_prompts = self.config.get("num_prompts")
        self.num_tasks = self.config.get("num_tasks")
        self.counts = [0] * self.num_prompts
        self.values = [0.0] * self.num_prompts
        self.task_lists = [list(range(self.num_tasks)) for _ in range(self.num_prompts)]
        self.last_action = None
        self.total_steps = 0

    def act(self, obs) -> tuple[int, int]:
        if 0 in self.counts:
            prompt = self.counts.index(0)
        else:
            ucb_scores = [
                (self.values[i] / self.counts[i]) + math.sqrt(2 * math.log(self.total_steps) / self.counts[i])
                for i in range(self.num_prompts)
            ]
            prompt = max(range(self.num_prompts), key=lambda i: ucb_scores[i])
        # choose next unobserved task for this prompt
        if self.task_lists[prompt]:
            task = self.task_lists[prompt][0]
        else:
            task = self.num_tasks - 1
        self.last_action = (prompt, task)
        return self.last_action

    def update(self, obs, reward):
        prompt, task = self.last_action
        self.counts[prompt] += 1
        self.values[prompt] += reward
        if self.task_lists[prompt] and self.task_lists[prompt][0] == task:
            self.task_lists[prompt].pop(0)
        self.total_steps += 1

    def current_best(self) -> int:
        if 0 in self.counts:
            return self.counts.index(0)
        avgs = [self.values[i] / self.counts[i] for i in range(self.num_prompts)]
        return max(range(self.num_prompts), key=lambda i: avgs[i])
