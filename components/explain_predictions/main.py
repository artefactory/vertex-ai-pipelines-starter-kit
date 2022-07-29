import os

from kfp.v2.dsl import Dataset, Input, Model, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def explain_predictions_step(
    input_folder: Input[Dataset],
    model_artifact: Input[Model],
    output_folder: Output[Dataset],
    config: dict,
):
    """
    Main of explain_predictions component.

    - Extract trained models for inference and evalutation
    - Compute shap values and scale per article x channel
    - Write explainabilty data to next components

    :param train_folder: the input folder of training data
    :param model_artifact: the input folder of trained models
    :param forecasts_folder: the input folder of training data
    :param output_folder: the output folder where the explainabitlity data is saved
    :param inference: if set to True, run the component for inference. Else, run it for evaluation.
    """
    import logging
    from pathlib import Path

    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap

    categorical_columns = (
        config["time_columns_categorical"]
        + config["id_cols"]
        + config["cols_calendar2"]
    )

    input_folder = Path(input_folder.path)
    model_artifact = Path(model_artifact.path)
    output_folder = Path(output_folder.path)

    model = joblib.load(model_artifact / "lgb.pkl")

    X_val = pd.read_csv(input_folder / "data_val.csv")
    X_val, y_val = (
        X_val.drop(config["unnecessary_cols"], axis=1),
        X_val[["sales"]].values,
    )

    categorical_columns_filtered = [
        c for c in categorical_columns if c in X_val.columns
    ]
    for col in categorical_columns_filtered:
        X_val[col] = pd.Categorical(X_val[col])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    logging.info(f"Model folder name: {model_artifact}")
    output_folder.mkdir(parents=True, exist_ok=True)
    # save these  files in a folder
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig(output_folder / "summary_plot.pdf")
