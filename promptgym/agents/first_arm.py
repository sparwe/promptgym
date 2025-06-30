from .base import BaseAgent

class FirstArm(BaseAgent):
    def act(self, obs) -> tuple[int, int]:
        return (0, 0)

