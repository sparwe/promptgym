# Dataset format

The dataset is a CSV with columns:

- `prompt_id`: integer identifier of the prompt.
- `question_id`: integer identifier of the question/task.
- `is_correct`: whether the prompt answered the question correctly.

`PromptTaskDataset.permute(seed)` provides a deterministic shuffling over prompts and tasks.

