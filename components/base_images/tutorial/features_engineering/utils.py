import logging
from typing import List

import pandas as pd

log = logging.getLogger()


def train_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> List[pd.DataFrame]:
    train = df.query(f"@train_start <= {date_col} <= @train_end").reset_index(drop=True)
    log.info(f"Training dataset shape is {train.shape}")
    test = df.query(f"@test_start <= {date_col} <= @test_end").reset_index(drop=True)
    log.info(f"Training dataset shape is {test.shape}")
    return train, test
