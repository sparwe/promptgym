import pandas as pd
import numpy as np
from pathlib import Path

class PromptTaskDataset:
    def __init__(self, csv_path: str | Path = None):
        if csv_path is None:
            csv_path = Path(__file__).parents[2]/"data"/"gsm8k_responses.csv"
        self.df = pd.read_csv(csv_path, dtype={"prompt_id": int, "question_id": int, "is_correct": bool})
        self.prompts = self.df["prompt_id"].unique()
        self.tasks = self.df["question_id"].unique()
        self._matrix = self._build_matrix()

    def _build_matrix(self):
        mat = np.zeros((self.prompts.max()+1, self.tasks.max()+1), dtype=bool)
        for _, row in self.df.iterrows():
            mat[row.prompt_id, row.question_id] = row.is_correct
        return mat

    def permute(self, seed: int):
        rng = np.random.default_rng(seed)
        p_perm = rng.permutation(len(self.prompts))
        t_perm = rng.permutation(len(self.tasks))
        return PermutedView(self._matrix[p_perm][:, t_perm], p_perm, t_perm)

class PermutedView:
    def __init__(self, mat, p_idx, t_idx):
        self.mat = mat
        self.p_idx = p_idx
        self.t_idx = t_idx
        self.num_prompts, self.num_tasks = mat.shape

    def reward(self, prompt_id: int, task_id: int) -> int:
        return int(self.mat[prompt_id, task_id])

