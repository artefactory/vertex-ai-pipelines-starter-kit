import os

from kfp.v2.dsl import Dataset, Input, Model, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}",
    packages_to_install=["catboost"],
)
def train_model_step(
    input_folder: Input[Dataset],
    model_artifact: Output[Model],
    output_folder: Output[Dataset],
    config: dict,
):
    from pathlib import Path

    import lightgbm as lgb
    import pandas as pd

    from components.base_images.tutorial.train_model.utils import log_models

    categorical_columns = (
        config["time_columns_categorical"]
        + config["id_cols"]
        + config["cols_calendar2"]
    )

    input_folder = Path(input_folder.path)
    output_folder = Path(output_folder.path)
    output_folder.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(input_folder / "data_train.csv")
    X_val = pd.read_csv(input_folder / "data_val.csv")
    if X_val.shape[0] == 0:
        X_val = X_train.tail(2)

    X_train, y_train = (
        X_train.drop(config["unnecessary_cols"], axis=1),
        X_train[["sales"]].values,
    )

    X_val, y_val = (
        X_val.drop(config["unnecessary_cols"], axis=1),
        X_val[["sales"]].values,
    )

    categorical_columns_filtered = [
        c for c in categorical_columns if c in X_train.columns
    ]
    for col in categorical_columns_filtered:
        X_train[col] = pd.Categorical(X_train[col])
        X_val[col] = pd.Categorical(X_val[col])

    import re

    X_train = X_train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    X_val = X_val.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_val, y_val)

    model = lgb.train(
        config["lgb_params"],
        train_data,
        early_stopping_rounds=200,
        valid_sets=[valid_data],
        verbose_eval=100,
    )
    log_models(model_artifact, model, "LightGBM")
