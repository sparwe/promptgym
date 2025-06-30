class BaseAgent:
    def __init__(self, config=None):
        self.config = config or {}

    def act(self, obs) -> tuple[int, int]:
        """Return a (prompt_id, task_id) pair."""
        raise NotImplementedError

    def update(self, obs, reward):
        pass

    def current_best(self) -> int:
        """Return the prompt index the agent currently believes is best."""
        raise NotImplementedError
