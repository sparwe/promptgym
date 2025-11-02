from __future__ import annotations
from typing import Literal, Tuple, Dict
import numpy as np

class Enviroment:
    def n_arms(self) -> int: ...
    def n_tasks(self) -> int: ...
    def is_allow_create(self) -> bool: ...
    def eval(self, arm: int, task: int) -> int: ...
    def create(self, **kwargs) -> int: ...

class RaschEnv(Enviroment):
    """
    p_ij = sigmoid(skill_i - difficulty_j)
    Y_ij ~ Bernoulli(p_ij) i.i.d.
    create_arm = skill_i + create_scale*N(0,1)
    """
    def __init__(self, n_arms: int, n_tasks: int, seed: int = None,
                 skill_scale: float = 1.0, difficulty_scale: float = 1.0,
                 baseline_difficulty: float = 0.0, is_allow_create: bool = False,
                 create_scale: float = 0.2):
        rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.skill = rng.normal(0.0, skill_scale, size=n_arms)
        self.difficulty = rng.normal(baseline_difficulty, difficulty_scale, size=n_tasks)
        logits = self.skill[:, None] - self.difficulty[None, :]
        p = 1.0 / (1.0 + np.exp(-logits))
        self.is_allow_create = is_allow_create
        self.create_scale = create_scale
        self.Y = np.random.binomial(p=p, n=1)

    def eval(self, arm: int, task: int) -> int:
        return self.Y[arm, task]

    def create(self, template_arm: int) -> None:
        if not self.is_allow_create:
            raise ValueError("Create action not allowed.")
        self.n_arms += 1
        new_skill = self.skill[template_arm] + np.random.normal(0.0, self.create_scale)
        new_logits = new_skill - self.difficulty 
        p = 1.0 / (1.0 + np.exp(-new_logits))
        new_Y = np.random.binomial(p=p, n=1)
        self.Y = np.vstack([self.Y, new_Y])
        self.skill = np.append(self.skill, new_skill)

    def n_arms(self) -> int:
        return self.n_arms

    def n_tasks(self) -> int:
        return self.n_tasks

    def is_allow_create(self) -> bool:
        return self.is_allow_create
