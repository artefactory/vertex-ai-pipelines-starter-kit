import os

from kfp.v2.dsl import Dataset, Input, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def using_deployed_model(
    input_folder: Input[Dataset],
    output_folder: Output[Dataset],
    bucket_models: str,
    config: dict,
):
    """
    Main of make_forecasts component.

    - Extract trained models and inference data
    - Compute predictions for the inference data
    - Write predictions to next components

    :param input_folder: the input folder of inference data
    :param output_folder: the output folder where predictions are saved
    :param bucket_models: the input folder of trained models
    """
    import sys
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from loguru import logger

    logger.add(
        sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
    )

    input_folder = Path(input_folder.path)
    output_folder = Path(output_folder.path)
    output_folder.mkdir(parents=True, exist_ok=True)

    categorical_columns = (
        config["time_columns_categorical"]
        + config["id_cols"]
        + config["cols_calendar2"]
    )

    X_inference = pd.read_csv(input_folder / "data_inference.csv")
    X_inference = X_inference.drop(config["unnecessary_cols"], axis=1)

    categorical_columns_filtered = [
        c for c in categorical_columns if c in X_inference.columns
    ]
    for col in categorical_columns_filtered:
        X_inference[col] = pd.Categorical(X_inference[col])

    from google.cloud import storage

    def load_model(bucket_name: str, file_name: str):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name).download_to_filename("local_model.pkl")
        # from_file = CatBoostRegressor()
        model = joblib.load("local_model.pkl")
        return model

    model = load_model(
        bucket_models, "lgb.pkl"
    )  # joblib.load("gs://" + bucket_models + "/lgb.pkl")

    X_inference["sales_pred"] = np.array(model.predict(X_inference))

    X_inference.to_csv(output_folder / "inference.csv", index=False)
