import random
from .base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__(config)
        self.num_prompts = self.config.get("num_prompts")
        self.num_tasks = self.config.get("num_tasks")

    def act(self, obs) -> tuple[int, int]:
        return (
            random.randrange(self.num_prompts),
            random.randrange(self.num_tasks),
        )

