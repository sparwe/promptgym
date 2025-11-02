"""Example script for running the random baseline across multiple seeds."""
from __future__ import annotations

from collections.abc import Iterable

from promptgym.env import RaschEnv
from promptgym.runner import run_many
from promptgym.strategy.random_baseline import RandomBaseline


def make_env(seed: int) -> RaschEnv:
    return RaschEnv(n_arms=10, n_tasks=100, seed=seed, allow_create=True)

def make_strategy(
    env: RaschEnv,
    seed: int,
    creation_prob: float,
    budget: int,
) -> RandomBaseline:
    return RandomBaseline(
        env,
        budget=budget,
        creation_prob=creation_prob,
        seed=seed,
    )
def evaluate_configuration(
    creation_prob: float, budget: int, seeds: Iterable[int]
) -> float:
    seed_list = list(seeds)

    def strategy_factory(env: RaschEnv, seed: int) -> RandomBaseline:
        return make_strategy(env, seed, creation_prob, budget)

    outcomes = run_many(make_env, strategy_factory, seed_list)
    return sum(o.reward for o in outcomes) / len(outcomes)


def main(
    seeds: Iterable[int] = range(1000),
    creation_probabilities: Iterable[float] = (0.0, 0.05, 0.1),
    budgets: Iterable[int] = (250, 500, 750, 1000),
) -> None:
    seed_list = list(seeds)

    for budget in budgets:
        print(f"Budget={budget}")
        best_prob: float | None = None
        best_reward = float("-inf")

        for prob in creation_probabilities:
            avg_reward = evaluate_configuration(prob, budget, seed_list)
            print(f"  creation_prob={prob:.2f}: average reward={avg_reward:.4f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_prob = prob

        if best_prob is not None:
            print(
                f"Best creation_prob={best_prob:.2f} with average reward={best_reward:.4f}"
            )
        print()


if __name__ == "__main__":  # pragma: no cover - manual usage
    main()
