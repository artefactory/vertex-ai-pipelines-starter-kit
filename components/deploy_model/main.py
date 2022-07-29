import os

from kfp.v2.dsl import Input, Model, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def deploying_model_step(
    input_folder_models: Input[Model],
    origin_bucket: str,
    bucket_models: str,
):
    import sys
    from pathlib import Path

    from google.cloud import storage
    from loguru import logger

    # TODO: transfer to base image
    def copy_blob(
        bucket_name: str,
        blob_name: str,
        destination_bucket_name: str,
        destination_blob_name: str = None,
        delete_origin: bool = False,
    ) -> None:
        """
        Moves a blob from one bucket to another with a new name.

        :param bucket_name: the ID of your GCS bucket
        :param blob_name: the ID of your GCS object
        :param destination_bucket_name: the ID of the bucket to move the object to
        :param destination_blob_name: the ID of your new GCS object (defaults to blob_name if not set)
        """
        storage_client = storage.Client()

        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        if destination_blob_name is None:
            destination_blob_name = blob_name
        destination_bucket = storage_client.bucket(destination_bucket_name)

        blob_copy = source_bucket.copy_blob(
            source_blob, destination_bucket, destination_blob_name
        )
        if delete_origin:
            source_bucket.delete_blob(blob_name)

    logger.add(
        sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
    )

    logger.info("Running Deploy Model component.")

    input_folder_models = Path(input_folder_models.path.replace("/gcs/", ""))
    logger.info(f"input_folder_models: {input_folder_models}")

    origin_bucket = origin_bucket.replace("gs://", "")
    destination_bucket = bucket_models
    logger.info(f"origin bucket: {origin_bucket}")
    logger.info(f"dest bucket: {destination_bucket}")

    origin_name = (
        input_folder_models.as_posix().replace(origin_bucket + "/", "") + "/lgb.pkl"
    )
    logger.info(f"origin name: {origin_name}")
    destination_name = "lgb.pkl"
    copy_blob(
        origin_bucket,
        origin_name,
        destination_bucket,
        destination_name,
        delete_origin=False,
    )
