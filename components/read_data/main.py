import os

from kfp.v2.dsl import Dataset, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def get_data_step(input_bucket_raw: str, output_folder: Output[Dataset]):
    from pathlib import Path

    from components.base_images.utils.storage import read_from_gcs

    sales_train = read_from_gcs(input_bucket_raw, "sales_train.csv")
    sales_inference = read_from_gcs(input_bucket_raw, "sales_inference.csv")
    prices = read_from_gcs(input_bucket_raw, "prices.csv")
    calendar = read_from_gcs(input_bucket_raw, "calendar.csv")

    output_folder = Path(output_folder.path)
    output_folder.mkdir(parents=True, exist_ok=True)

    sales_train.to_csv(output_folder / "sales_train.csv")
    sales_inference.to_csv(output_folder / "sales_inference.csv")
    prices.to_csv(output_folder / "prices.csv")
    calendar.to_csv(output_folder / "calendar.csv")
