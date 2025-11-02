"""Minimal runner utilities for evaluating strategies across multiple seeds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .env import Environment
from .strategy.base import (
    Action,
    CreateAction,
    CreateResult,
    EvalAction,
    EvalResult,
    Result,
    StopAction,
    Strategy,
)


@dataclass(slots=True)
class EpisodeOutcome:
    """Summary of a single strategy run inside an environment."""
    seed: int
    reward: float 

def _step(env: Environment, action: Action) -> Result | None:
    match action:
        case StopAction():
            return None
        case EvalAction(arm=arm, task=task):
            y = env.eval(arm, task)
            return EvalResult(arm=arm, task=task, y=y)
        case CreateAction(spec=spec):
            try:
                arm_id = env.create(spec)
                return CreateResult(ok=True, arm_id=arm_id)
            except Exception as exc:  # defensive guard
                return CreateResult(ok=False, meta={"error": str(exc)})
        case _:
            raise TypeError(f"Unknown action type: {type(action)!r}")

def run_episode(
    env: Environment,
    strategy: Strategy,
    *,
    seed: int,
    max_steps: int | None = None,
) -> EpisodeOutcome:
    """Run a single strategy/environment episode until termination."""

    history: list[Result] = []
    total_reward = 0

    while not strategy.is_done():
        action = strategy.next_action()
        result = _step(env, action)
        if result is None:
            break
        history.append(result)
        strategy.update(result)
    winner = strategy.select_winner()
    labels = env.ground_truth()['labels']
    reward = labels[winner, :].mean()
    best = labels.mean(axis=1)
    return EpisodeOutcome(seed=seed, reward=reward)


def run_many(
    env_factory: Callable[[int], Environment],
    strategy_factory: Callable[[Environment, int], Strategy],
    seeds: Sequence[int] | Iterable[int],
    ) -> list[EpisodeOutcome]:
    """Run a strategy across multiple seeds and collect the outcomes."""

    outcomes: list[EpisodeOutcome] = []
    for seed in seeds:
        env = env_factory(seed)
        strategy = strategy_factory(env, seed)
        outcome = run_episode(env, strategy, seed=seed)
        outcomes.append(outcome)
    return outcomes
