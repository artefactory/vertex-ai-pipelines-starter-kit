import os

from kfp.v2.dsl import Dataset, Input, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def writing_data(input_folder: Input[Dataset], config: dict):
    """
    Main of write_data component.

    Load data to Big Query.

    :param input_folder: the input folder for the inference data
    :config configuration file for this component
    """
    import sys
    from pathlib import Path

    import pandas as pd
    import pandas_gbq
    from loguru import logger

    logger.add(
        sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
    )

    input_folder = Path(input_folder.path)
    inference_df = pd.read_csv(input_folder / "inference.csv")

    inference_df = inference_df[[c for c in inference_df.columns if " " not in c]]

    pandas_gbq.to_gbq(
        inference_df,
        f"{config['DATASET']}.{config['TABLE']}",
        if_exists="replace",
        project_id=config["PROJECT_ID"],
    )
