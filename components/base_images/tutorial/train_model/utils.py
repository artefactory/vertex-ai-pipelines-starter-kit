from datetime import datetime
from pathlib import Path

import joblib


def _get_timestamp_str():
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    return timestamp


def log_models(model_artifact, model, framework) -> dict:
    """
    Saves models for later usage.

    :param model_artifact: Location folder for the model
    :param models: the model itself
    """
    model_artifact.metadata["framework"] = framework

    model_folder = Path(model_artifact.path)
    model_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_folder / "lgb.pkl")
