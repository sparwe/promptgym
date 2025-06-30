import random
from .base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__(config)
        self.num_prompts = self.config.get("num_prompts")

    def act(self, obs) -> int:
        return random.randrange(self.num_prompts)

