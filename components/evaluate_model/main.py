import os
from typing import NamedTuple

from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}",
    packages_to_install=["catboost"],
)
def evaluate_model_step(
    input_folder: Input[Dataset],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    config: dict,
) -> NamedTuple("results", [("FA", float)]):
    from collections import namedtuple
    from pathlib import Path

    import joblib
    import pandas as pd

    from components.base_images.tutorial.evaluation.evaluate import evaluate

    categorical_columns = (
        config["time_columns_categorical"]
        + config["id_cols"]
        + config["cols_calendar2"]
    )

    input_folder = Path(input_folder.path)
    model_artifact = Path(model_artifact.path)

    X_train = pd.read_csv(input_folder / "data_train.csv")
    X_train, y_train = (
        X_train.drop(config["unnecessary_cols"], axis=1),
        X_train[["sales"]].values,
    )
    X_val = pd.read_csv(input_folder / "data_val.csv")
    X_val, y_val = (
        X_val.drop(config["unnecessary_cols"], axis=1),
        X_val[["sales"]].values,
    )

    categorical_columns_filtered = [
        c for c in categorical_columns if c in X_val.columns
    ]
    for col in categorical_columns_filtered:
        X_train[col] = pd.Categorical(X_train[col])
        X_val[col] = pd.Categorical(X_val[col])

    model = joblib.load(model_artifact / "lgb.pkl")

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    _, _ = evaluate(y_train, y_pred_train, "training")
    fa, rmse = evaluate(y_val, y_pred_val, "validation")

    metrics.log_metric("FA", fa)
    metrics.log_metric("RMSE", rmse)

    result_tuple = namedtuple(
        "results",
        ["FA"],
    )

    return result_tuple(FA=fa)
