from promptgym.evaluation.runner import run_once


def test_ucb_runs():
    df = run_once("ucb", seed=0, budget=3)
    assert not df.empty
    assert {"prompt", "task"}.issubset(df.columns)
