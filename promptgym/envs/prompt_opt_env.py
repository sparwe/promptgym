class PromptOptEnv:
    def __init__(self, perm_view, budget: int | None = None):
        self.view = perm_view
        self.budget = budget or (perm_view.num_prompts * perm_view.num_tasks)
        self.step_n = 0
        self.history = []

    def reset(self):
        self.step_n = 0
        self.history.clear()
        return self._obs()

    def _obs(self):
        return {
            "history": list(self.history),
        }

    def step(self, action: tuple[int, int]):
        prompt_id, task_id = action
        r = self.view.reward(prompt_id, task_id)
        self.history.append((prompt_id, task_id, r))
        self.step_n += 1
        done = self.step_n >= self.budget
        return self._obs(), r, done, {}

