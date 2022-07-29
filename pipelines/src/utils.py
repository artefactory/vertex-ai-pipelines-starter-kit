import json
from typing import Any, List

from google.cloud import storage


def import_method(module: str, method: str) -> Any:
    return getattr(__import__(module, fromlist=[method]), method)


def upload_file_to_gcs(path: str, bucket_name) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path.split("/")[-1])
    blob.upload_from_filename(path)


def read_file_from_gcs(filename: str, bucket_name: str, local_destination: str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    downloaded_json_file = json.loads(blob.download_as_text(encoding="utf-8"))
    with open(local_destination, "w") as outfile:
        json.dump(downloaded_json_file, outfile)


def get_list_files(bucket_name: str) -> List[str]:
    storage_client = storage.Client()
    return [blob.name for blob in storage_client.list_blobs(bucket_name)]
