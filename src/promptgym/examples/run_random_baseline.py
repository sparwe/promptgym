"""Example script for running the random baseline across multiple seeds."""
from __future__ import annotations

from collections.abc import Iterable

from promptgym.env import RaschEnv
from promptgym.runner import run_many
from promptgym.strategy.random_baseline import RandomBaseline


def make_env(seed: int) -> RaschEnv:
    return RaschEnv(n_arms=10, n_tasks=100, seed=seed)


def make_strategy(env: RaschEnv, seed: int) -> RandomBaseline:
    return RandomBaseline(env, budget=200, seed=seed)


def main(seeds: Iterable[int] = range(1000)) -> None:
    seed_list = list(seeds)
    outcomes = run_many(make_env, make_strategy, seed_list)
    avg_reward = sum(o.reward for o in outcomes) / len(outcomes)
    print("Ran", len(outcomes), "episodes")
    print("Average reward:", avg_reward)


if __name__ == "__main__":  # pragma: no cover - manual usage
    main()
