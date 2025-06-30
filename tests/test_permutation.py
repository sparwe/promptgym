from promptgym.dataio.dataset import PromptTaskDataset


def test_permutation_deterministic():
    ds1 = PromptTaskDataset().permute(42)
    ds2 = PromptTaskDataset().permute(42)
    ds3 = PromptTaskDataset().permute(1)
    assert (ds1.mat == ds2.mat).all()
    assert not (ds1.mat == ds3.mat).all()

