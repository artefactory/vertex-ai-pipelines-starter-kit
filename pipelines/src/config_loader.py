import json
import os
import pathlib

CONFIG_FOLDER = pathlib.Path(__file__).parent.parent.parent


def load_config_from_path(filepath: str) -> dict:
    """
    Config loader for train_model config.

    :param filepath: file path to json train_model config file
    :return: train_model config dict
    """
    with open(filepath) as json_file:
        config = json.load(json_file)

    return config


def load_component_config(component: str, uc_name: str) -> dict:
    filepath = (
        CONFIG_FOLDER / "config" / "components" / component / uc_name / "config.json"
    )
    with open(filepath) as stream:
        config = json.load(stream)
    return config


def load_pipeline_config():

    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
    UC_NAME = os.getenv("UC_NAME")
    REGION = os.getenv("REGION")

    filepath = CONFIG_FOLDER / "config" / "pipelines" / f"{EXPERIMENT_NAME}.json"

    with open(filepath) as stream:
        pipeline_config = json.load(stream)

    config = {}
    config["PROJECT_ID"] = PROJECT_ID
    config["experiment_name"] = EXPERIMENT_NAME
    config["uc_name"] = UC_NAME
    config["REGION"] = REGION

    config["PACKAGE_PATH"] = "pipelines/runs"

    pipeline_config.update(config)

    return pipeline_config
