import pandas as pd


def get_pricing_features(
    df: pd.DataFrame,
    group_by: list,
    price_col: str = "sell_price",
):

    avg_price = (
        df.groupby(group_by)[price_col]
        .mean()
        .reset_index()
        .rename(columns={price_col: f"avg_{price_col}"})
    )
    df = df.merge(avg_price, how="left", on=group_by)
    df[f'{price_col}_diff_{"_".join(group_by)}'] = (
        df[price_col] - df[f"avg_{price_col}"]
    ) / df[f"avg_{price_col}"]
    df = df.drop(f"avg_{price_col}", axis=1)
    return df
