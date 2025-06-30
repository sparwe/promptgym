import pandas as pd
from promptgym.evaluation.runner import run_once
from promptgym.evaluation.metrics import compute_metrics
from promptgym.dataio.dataset import PromptTaskDataset


def test_omniscient_regret_zero():
    seed = 1
    ds = PromptTaskDataset().permute(seed)
    rows = []
    for t in range(ds.num_tasks):
        best = ds.mat[:, t].argmax()
        rows.append({"step": t, "prompt": best, "task": t, "reward": 1, "agent_best": best})
    df = pd.DataFrame(rows)
    metrics = compute_metrics(df, seed)
    assert all(r == 0 for r in metrics["regret"])


def test_runner_parquet(tmp_path):
    df = run_once("random", seed=0, budget=2)
    out = tmp_path/"test.parquet"
    df.to_parquet(out, engine="pyarrow")
    loaded = pd.read_parquet(out, engine="pyarrow")
    assert not loaded.empty

