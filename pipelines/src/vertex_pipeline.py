from datetime import datetime

from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient

from pipelines.src.config_loader import load_pipeline_config
from pipelines.src.tutorial import inference_pipeline as inference
from pipelines.src.tutorial import training_pipeline as simple
from pipelines.src.tutorial import training_pipeline_parallelized as parallel
from pipelines.src.tutorial import training_pipeline_with_conditiion as conditional
from pipelines.src.utils import upload_file_to_gcs


def compile_and_run_pipeline(
    sync: bool = False,
    enable_caching: bool = False,
    name: str = None,
    **kwargs,
) -> None:
    """
    Compile the pipeline and run it just after, using the generated JSON file.

    :param sync: defaults to True
    :param enable_caching: defaults to False
    """
    package_path = compile_pipeline(name=name, local=True)
    run_pipeline(
        package_path,
        sync=sync,
        enable_caching=enable_caching,
    )


def compile_pipeline(local: bool = True, **kwargs) -> str:
    """
    Compile the pipeline.

    - Compile the pipeline to a .json file spec saved locally

    :param local: if set to False, store the JSON pipeline file in Cloud Storage.

    :return: The package path created by the compilation. This can be used by the `run_pipeline` function
    """
    # load config
    config = load_pipeline_config()
    pipeline_name = f"{config['uc_name']}-{config['experiment_name']}".replace("_", "-")

    # compile pipeline
    package_path_no_timestamp = f"{config['PACKAGE_PATH']}/{pipeline_name}.json"
    package_path_timestamp = (
        f"{config['PACKAGE_PATH']}/{pipeline_name}-{_get_timestamp_str()}.json"
    )

    def pipeline_func():
        if "simple" in config["experiment_name"]:
            return simple()
        elif "conditional" in config["experiment_name"]:
            return conditional()
        elif "parallel" in config["experiment_name"]:
            return parallel()
        elif "inference" in config["experiment_name"]:
            return inference()

    for package_path in [package_path_no_timestamp, package_path_timestamp]:
        compiler.Compiler().compile(
            pipeline_func=pipeline_func,
            package_path=package_path,
            pipeline_name=pipeline_name,
            type_check=True,
        )
        if not local:
            upload_file_to_gcs(package_path, config["BUCKET_PIPELINE_CONFIGS"])
    return package_path


def run_pipeline(
    package_path: str,
    sync: bool = False,
    enable_caching: bool = False,
) -> None:
    """
    Submit the pipeline to Vertex Pipelines.

    :param package_path: The path to the JSON file used to run the pipeline: either local or GCS URI
    :param sync: if set to True, this function will wait for the the pipeline job to finish before continuing
    :param enable_caching: if set to True, the vertex pipeline will only run from the last component that has
        been changed. Components before this will not be run and artifacts from the previous vertex pipeline
        job will be used. This is very practical during development to test new changes in a specific component.
    """
    config = load_pipeline_config()
    if sync:
        raise NotImplementedError(
            "The syncronous run of the vertex pipeline job is no longer implemented."
        )
    api_client = AIPlatformClient(
        project_id=config["PROJECT_ID"], region=config["REGION"]
    )
    service_account = f"{config['SERVICE_ACCOUNT_NAME']}@{config['PROJECT_ID']}.iam.gserviceaccount.com"

    api_client.create_run_from_job_spec(
        job_spec_path=package_path,
        pipeline_root=config["PIPELINE_ROOT"],
        enable_caching=enable_caching,
        service_account=service_account,
        labels=None,
    )


def run_latest_pipeline(
    sync: bool = False,
    enable_caching: bool = False,
    local: bool = False,
    **kwargs,
) -> None:

    package_path = get_last_pipeline_config_path(local=local)
    run_pipeline(
        package_path,
        sync=sync,
        enable_caching=enable_caching,
    )


def get_last_pipeline_config_path(local: bool = False) -> str:
    """Find the last config of compiled pipeline, either in local or on GCS.

    :param local: Whether to find for the last config in local or on the GCS bucket, defaults to False

    :return: The path to the last config JSON file
    """
    config = load_pipeline_config()

    if local:
        path_prefix = f'{config["PACKAGE_PATH"]}/'
    else:
        path_prefix = f"gs://{config['BUCKET_PIPELINE_CONFIGS']}/"

    valid_name = f"{config['uc_name']}-{config['experiment_name']}".replace("_", "-")

    return f"{path_prefix}{valid_name}.json"


def _get_timestamp_str():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return timestamp
