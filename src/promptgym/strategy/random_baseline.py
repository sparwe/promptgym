from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..env import Environment
from .base import Action, EvalAction, StopAction, EvalResult, CreateResult, Strategy

class RandomBaseline(Strategy):
    """Uniform random unseen (arm, task) pairs until budget or exhaustion."""
    def __init__(self, env: Environment, budget=1000, creation_prob: float = 0.0, seed: int = None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.observations = np.ma.masked_all((self.env.n_arms(), self.env.n_tasks()), dtype=bool)
        self.done = False
        self.budget = budget
        self.creation_prob = creation_prob
        self.spend = 0

    def name(self) -> str: return "random"

    def next_action(self) -> Action:
        if self.done: return StopAction(reason="done")
        if self.env.allow_create and np.random.rand() < self.creation_prob:
            best_arm = self.select_winner()
            return CreateAction(best_arm)
        unseen = np.argwhere(self.observations.mask)
        if unseen.size == 0:
            if self.env.allow_create():
                best_arm = self.select_winner()
                return CreateAction(best_arm)
            return StopAction(rationale="exhausted_all_actions")
        i, j = unseen[self.rng.integers(0, len(unseen))]
        return EvalAction(arm=i, task=j)
       
    def update(self, result: Result) -> None:
        match result:
            case EvalResult(arm=arm, task=task, y=y, meta=meta):
                self.observations[arm, task] = y
                self.spend += 1
            case CreateResult(ok=ok, arm_id=arm_id, meta=meta):
                self.observations = np.ma.vstack(self.observations, np.ma.masked_all(n, dtype=bool))
                self.spend += 1
            case _:
                raise TypeError(f"Unexpected Result type: {type(result)}")
        if self.spend >= self.budget: self.done = True

    def is_done(self) -> bool:
        return self.done

    def select_winner(self) -> int:
        return np.argmax(np.mean(self.observations, axis=1)) # - 2*np.std(self.observations, axis=1, ddof=1))
