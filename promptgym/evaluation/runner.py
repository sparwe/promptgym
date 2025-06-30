import pandas as pd
from pathlib import Path
from ..dataio.dataset import PromptTaskDataset
from ..envs.prompt_opt_env import PromptOptEnv
from ..agents import REGISTRY, BaseAgent


def run_once(agent_name: str, seed: int, budget: int | None = None):
    ds = PromptTaskDataset().permute(seed)
    env = PromptOptEnv(ds, budget)
    agent = REGISTRY[agent_name]({
        "num_prompts": ds.num_prompts,
        "num_tasks": ds.num_tasks,
        "budget": budget,
    })
    obs = env.reset()
    log = []
    for step in range(env.budget):
        a = agent.act(obs)
        obs, r, done, _ = env.step(a)
        agent.update(obs, r)
        if getattr(agent.__class__, "current_best", BaseAgent.current_best) is not BaseAgent.current_best:
            best = agent.current_best()
        else:
            best = a[0]
        log.append({
            "step": step,
            "prompt": a[0],
            "task": a[1],
            "reward": r,
            "agent_best": best,
        })
        if done:
            break
    return pd.DataFrame(log)


def cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(REGISTRY))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--outdir", default="runs")
    args = p.parse_args()
    df = run_once(args.agent, args.seed, args.budget)
    out = Path(args.outdir) / f"{args.agent}_seed{args.seed}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(out)

if __name__ == "__main__":
    cli()

