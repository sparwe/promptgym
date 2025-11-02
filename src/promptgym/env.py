from __future__ import annotations
from typing import Literal, Tuple, Dict
import numpy as np

class Environment:
    def n_arms(self) -> int: ...
    def n_tasks(self) -> int: ...
    def allow_create(self) -> bool: ...
    def eval(self, arm: int, task: int) -> int: ...
    def create(self, **kwargs) -> int: ...
    def ground_truth(self): ...

class RaschEnv(Environment):
    """
    Rasch model:
        p_ij = sigmoid(skill_i - difficulty_j)
        Y_ij ~ Bernoulli(p_ij) i.i.d.

    create(template_arm): new arm skill = skill_template + Normal(0, create_scale)
    """
    def __init__(self, n_arms: int, n_tasks: int, seed: int = None,
                 skill_scale: float = 1.0, difficulty_scale: float = 1.0,
                 baseline_difficulty: float = 0.0, allow_create: bool = False,
                 create_scale: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.skill = self.rng.normal(0.0, skill_scale, size=n_arms)
        self.difficulty = self.rng.normal(baseline_difficulty, difficulty_scale, size=n_tasks)
        logits = self.skill[:, None] - self.difficulty[None, :]
        p = 1.0 / (1.0 + np.exp(-logits))
        self.allow_create = allow_create
        self.create_scale = create_scale
        self._observations = self.rng.binomial(p=p, n=1)

    def eval(self, arm: int, task: int) -> int:
        if not (0 <= arm < self.n_arms()):
            raise IndexError(f"arm index {arm} out of range [0, {self.n_arms()-1}]")
        if not (0 <= task < self.n_tasks()):
            raise IndexError(f"task index {task} out of range [0, {self.n_tasks()-1}]")
        return self._observations[arm, task]

    def create(self, template_arm: Optional[int] = None) -> int:
        if not self.allow_create:
            raise ValueError("Creation action not allowed.")
        if template_arm is None:
            template_arm = self._rng.integers(self.n_arms())
        new_skill = self.skill[template_arm] + self.rng.random.normal(0.0, self.create_scale)
        new_logits = new_skill - self.difficulty 
        p = 1.0 / (1.0 + np.exp(-new_logits))
        new_observations = self.rng.random.binomial(p=p, n=1)
        self._observations = np.vstack([self._observations, new_observations])
        self.skill = np.append(self.skill, new_skill)
        return self.n_arms()

    def n_arms(self) -> int:
        return self._observations.shape[0]

    def n_tasks(self) -> int:
        return self._observations.shape[1]

    def is_allow_create(self) -> bool:
        return self.allow_create

    def ground_truth(self) -> Dict[str, Any]:
        return {
            "skill": self.skill.copy(),
            "difficulty": self.difficulty.copy(),
            "labels": self._observations.copy(),
        }
