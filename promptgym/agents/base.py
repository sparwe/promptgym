class BaseAgent:
    def __init__(self, config=None):
        self.config = config or {}

    def act(self, obs) -> int:
        raise NotImplementedError

    def update(self, obs, reward):
        pass

    def current_best(self) -> int:
        """Return the prompt index the agent currently believes is best."""
        raise NotImplementedError
