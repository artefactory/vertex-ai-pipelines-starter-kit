import numpy as np
import pandas as pd


def get_time_features(df, date: str = "date"):
    """Calculates time-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    date : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional features.
    """
    df[date] = pd.to_datetime(df[date], format="%Y-%m-%d")

    df["dayofmonth"] = df[date].dt.day.astype(np.int8)
    df["week"] = df[date].dt.isocalendar().week.astype(np.int8)
    df["month"] = df[date].dt.month.astype(np.int8)
    df["year"] = df[date].dt.year
    df["dayofweek"] = df[date].dt.dayofweek.astype(np.int8)
    df["weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["dayofyear"] = df[date].dt.dayofyear.astype(np.int16)

    df["day_temp"] = 1
    df["year_month"] = df.year.astype(str) + "_" + df.month.astype(str)
    df = df.drop("day_temp", axis=1)

    return df
