import os

from kfp.v2.dsl import Dataset, Input, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def features_engineering_step(
    input_folder: Input[Dataset],
    output_folder: Output[Dataset],
    config: dict,
) -> None:
    from pathlib import Path

    import pandas as pd

    from components.base_images.tutorial.features_engineering.lags_and_rolling import (
        aggregate_lags,
        get_lags,
        get_rolling_means,
    )
    from components.base_images.tutorial.features_engineering.price_features import (
        get_pricing_features,
    )
    from components.base_images.tutorial.features_engineering.time_features import (
        get_time_features,
    )

    input_folder = Path(input_folder.path)
    output_folder = Path(output_folder.path)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_folder / "data_prepared.csv")

    # compute time featues
    df = get_time_features(df)

    for col in config["time_columns_categorical"]:
        df[col] = pd.Categorical(df[col])

    # compute lags
    df = get_lags(df=df, lags=range(14, 35))
    df = get_lags(df=df, lags=[365])

    for granularity in ["cat_id", "dept_id"]:
        df = aggregate_lags(df=df, lags=range(14, 35), group_by=granularity)
        df = aggregate_lags(df=df, lags=[365], group_by=granularity)

    # compute rolling means
    for granularity in ["", "_cat_id", "_dept_id"]:
        df = get_rolling_means(df=df, lags=range(14, 21), granularity=granularity)
        df = get_rolling_means(df=df, lags=range(21, 28), granularity=granularity)
        df = get_rolling_means(df=df, lags=range(28, 35), granularity=granularity)

    for granularity in ["", "_cat_id", "_dept_id"]:
        df = df.drop([f"sales_lag_{i}{granularity}" for i in range(21, 35)], axis=1)

    # prices features
    price_agg = [
        # Price difference with the other products sold the same day in the same store
        ["date", "store_id"],
        # Price difference with the other products sold the same day in the same store and category
        ["date", "store_id", "cat_id"],
        # Price difference with the other products sold the same day in the same store and sub-category
        ["date", "store_id", "dept_id"],
        # Price difference with the average historical price for this product in this store
        ["item_id", "store_id"],
    ]

    for group_by in price_agg:
        df = get_pricing_features(df, group_by)

    X_train = df.loc[df["date"] < config["validation_start_date"]]
    X_val = df.loc[
        (df["date"] >= config["validation_start_date"]) & (~pd.isnull(df["sales"]))
    ]
    X_inference = df.loc[pd.isnull(df["sales"])]

    X_train.to_csv(output_folder / "data_train.csv")
    X_val.to_csv(output_folder / "data_val.csv")
    X_inference.to_csv(output_folder / "data_inference.csv")
