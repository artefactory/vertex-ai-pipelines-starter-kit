import pandas as pd


def get_lags(
    df: pd.DataFrame,
    lags: list,
    group_by: list = ["item_id"],
    target: str = "sales",
    date: str = "date",
):
    """Calculates the lags features of a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    lags : List
        List of lags to compute (integers).
    group_by : List
        List of columns to apply the aggregation on.
    target : str
        Name of the column to apply the lags on.
    date : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional features.
    """
    df = df.sort_values(group_by + [date]).reset_index(drop=True)
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(group_by)[target].shift(lag)

    return df


def aggregate_lags(
    df: pd.DataFrame,
    lags: list,
    group_by: str,
    target: str = "sales",
    date: str = "date",
):
    """Aggregates lags features already calculated at a more granular level.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    lags : List
        List of lags that have already been computed.
    group_by : str
        Name of the column at which aggregation is made.
    target : str
        Name of the column to apply the lags on.
    date : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional features.
    """
    for lag in lags:
        group = (
            df.groupby([date, group_by])[f"{target}_lag_{lag}"]
            .mean()
            .to_frame()
            .rename(columns={f"{target}_lag_{lag}": f"{target}_lag_{lag}_{group_by}"})
        )
        df = df.merge(group, left_on=[date, group_by], right_index=True)

    return df


def get_rolling_means(
    df: pd.DataFrame,
    lags: list,
    granularity: str,
    target: str = "sales",
    date: str = "date",
):
    """Averages the values of several lags.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    lag_min : List
        List of lags to average.
    granularity : str
        Granularity of the lags.
    target : str
        Name of the column to apply the lags on.
    date : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional features.
    """
    col_name = f"{target}_rolling_mean_{min(lags)}-{max(lags)}{granularity}"
    cols_avg = [f"{target}_lag_{i}{granularity}" for i in lags]
    df[col_name] = df[cols_avg].mean(axis=1)

    return df
