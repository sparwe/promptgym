class PromptOptEnv:
    def __init__(self, perm_view, budget: int | None = None):
        self.view = perm_view
        self.budget = budget or (perm_view.num_prompts * perm_view.num_tasks)
        self.step_n = 0
        self.current_task = 0
        self.history = []

    def reset(self):
        self.step_n = 0
        self.current_task = 0
        self.history.clear()
        return self._obs()

    def _obs(self):
        return {
            "task_id": self.current_task,
            "history": list(self.history),
        }

    def step(self, action: int):
        r = self.view.reward(action, self.current_task)
        self.history.append((action, r))
        self.step_n += 1
        done = self.step_n >= self.budget or self.current_task == self.view.num_tasks - 1
        if not done:
            self.current_task += 1
        return self._obs(), r, done, {}

