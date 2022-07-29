import pandas as pd
import pandas_gbq
from google.oauth2 import service_account

from components.base_images.utils.decorator import shapeit, timeit


@timeit
@shapeit
def read_from_gcs(bucket: str, filename: str):
    filename = filename.replace(".csv", "")
    return pd.read_csv(f"gs://{bucket}/{filename}.csv")


@timeit
@shapeit
def read_from_bq(query, project_id, key_path=None, progress_bar_type=None):
    if key_path:
        credentials = service_account.Credentials.from_service_account_file(key_path)
    else:
        credentials = None

    return pandas_gbq.read_gbq(
        query,
        project_id=project_id,
        credentials=credentials,
        progress_bar_type=progress_bar_type,
    )
