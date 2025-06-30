import pandas as pd
from pathlib import Path

def test_load():
    df = pd.read_csv(Path(__file__).parents[1]/"data"/"gsm8k_responses.csv")
    assert not df.empty and set(df.columns)=={"prompt_id","question_id","is_correct"}

