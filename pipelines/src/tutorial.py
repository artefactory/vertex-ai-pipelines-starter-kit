from kfp.v2 import dsl

from components.deploy_model.main import deploying_model_step
from components.evaluate_model.main import evaluate_model_step
from components.explain_predictions.main import explain_predictions_step
from components.features_engineering.main import features_engineering_step
from components.preprocessing.main import prepare_data_step
from components.read_data.main import get_data_step
from components.train_model.main import train_model_step
from components.use_deployed_model.main import using_deployed_model
from pipelines.src.config_loader import load_component_config, load_pipeline_config

# from components.write_data.main import writing_data


@dsl.pipeline(description="Experiment with a simple training pipeline")
def training_pipeline() -> None:
    """Pipeline for training and conditionnal model deployment.

    This pipeline run a repeatable training and evaluation task.
    It can optionaly deploy a model in a production environment
    if some conditions are met.
    """
    # A. The first step is to load the pipeline config file
    pipeline_config = load_pipeline_config()

    # The next step is to define the pipeline
    #                                                --> model explantion  <--
    #                                               |                         |
    # read_data --> prepare_data --> feature_engineering --> model training --
    #                                               |                         |
    #                                                --> model evaluation  <--

    # 1. Getting the data from BQ
    # Constant inputs can be send in the pipeline's config file
    get_data_task = get_data_step(
        input_bucket_raw=pipeline_config["input_bucket_raw"],
    )

    # 2. Preparing the data
    preprocessing_config = load_component_config(
        "preprocessing", pipeline_config["uc_name"]
    )
    prepare_data_task = prepare_data_step(
        input_folder=get_data_task.outputs["output_folder"], config=preprocessing_config
    )

    # 3. Engineer features
    # One can also decide to load component's config like this:
    features_engineering_config = load_component_config(
        "features_engineering", pipeline_config["uc_name"]
    )
    features_engineering_task = features_engineering_step(
        input_folder=prepare_data_task.outputs["output_folder"],
        config=features_engineering_config,
    ).set_memory_limit("32G")

    # 4. Train a model
    train_model_config = load_component_config(
        "train_model", pipeline_config["uc_name"]
    )
    train_model_task = train_model_step(
        input_folder=features_engineering_task.outputs["output_folder"],
        config=train_model_config,
    ).set_memory_limit("32G")

    # 5. Evaluate a model
    # The previous loaded config can also be used later in the DAG
    evaluate_model_task = evaluate_model_step(
        input_folder=features_engineering_task.outputs["output_folder"],
        model_artifact=train_model_task.outputs["model_artifact"],
        config=train_model_config,
    ).set_memory_limit("32G")

    # 6. Explain predictions
    explain_predictions_task = (
        explain_predictions_step(
            input_folder=features_engineering_task.outputs["output_folder"],
            model_artifact=train_model_task.outputs["model_artifact"],
            config=train_model_config,
        )
        .set_cpu_limit("8")
        .set_memory_limit("32G")
    )


@dsl.pipeline(description="Experiment with parallelization")
def training_pipeline_parallelized() -> None:
    """Pipeline for training and conditionnal model deployment.

    This pipeline run a repeatable training and evaluation task.
    It can optionaly deploy a model in a production environment
    if some conditions are met.
    """
    # A. The first step is to load the pipeline config file
    pipeline_config = load_pipeline_config()

    get_data_task = get_data_step(
        input_bucket_raw=pipeline_config["input_bucket_raw"],
    )

    # 2. Preparing the data
    preprocessing_config = load_component_config(
        "preprocessing", pipeline_config["uc_name"]
    )
    prepare_data_task = prepare_data_step(
        input_folder=get_data_task.outputs["output_folder"], config=preprocessing_config
    )

    # 3. Engineer features
    # One can also decide to load component's config like this:
    features_engineering_config = load_component_config(
        "features_engineering", pipeline_config["uc_name"]
    )
    features_engineering_task = features_engineering_step(
        input_folder=prepare_data_task.outputs["output_folder"],
        config=features_engineering_config,
    ).set_memory_limit("32G")

    # From this step, we will train 3 model with defferent 'num_boost_round'
    # To do so, we will override the training config files using inputs from the pipeline
    # 4. Train a model
    train_model_config = load_component_config(
        "train_model", pipeline_config["uc_name"]
    )
    # // can simply be done with for loops
    for obj in pipeline_config["objectives"]:
        train_model_config["lgb_params"]["objective"] = obj
        train_model_task = train_model_step(
            input_folder=features_engineering_task.outputs["output_folder"],
            config=train_model_config,
        ).set_memory_limit("32G")

        # 5. Evaluate a model
        # The previous loaded config can also be used later in the DAG
        evaluate_model_task = evaluate_model_step(
            input_folder=features_engineering_task.outputs["output_folder"],
            model_artifact=train_model_task.outputs["model_artifact"],
            config=train_model_config,
        ).set_memory_limit("32G")

        # 6. Explain predictions
        explain_predictions_task = (
            explain_predictions_step(
                input_folder=features_engineering_task.outputs["output_folder"],
                model_artifact=train_model_task.outputs["model_artifact"],
                config=train_model_config,
            )
            .set_cpu_limit("8")
            .set_memory_limit("32G")
        )


@dsl.pipeline(description="Experiment with conditional deployment")
def training_pipeline_with_conditiion() -> None:
    """Pipeline for training and conditionnal model deployment.

    This pipeline run a repeatable training and evaluation task.
    It can optionaly deploy a model in a production environment
    if some conditions are met.
    """
    # A. The first step is to load the pipeline config file
    pipeline_config = load_pipeline_config()

    get_data_task = get_data_step(
        input_bucket_raw=pipeline_config["input_bucket_raw"],
    )

    # 2. Preparing the data
    preprocessing_config = load_component_config(
        "preprocessing", pipeline_config["uc_name"]
    )
    prepare_data_task = prepare_data_step(
        input_folder=get_data_task.outputs["output_folder"], config=preprocessing_config
    )

    # 3. Engineer features
    # One can also decide to load component's config like this:
    features_engineering_config = load_component_config(
        "features_engineering", pipeline_config["uc_name"]
    )
    features_engineering_task = features_engineering_step(
        input_folder=prepare_data_task.outputs["output_folder"],
        config=features_engineering_config,
    ).set_memory_limit("32G")

    # From this step, we will train 3 model with defferent 'num_boost_round'
    # To do so, we will override the training config files using inputs from the pipeline
    # 4. Train a model
    train_model_config = load_component_config(
        "train_model", pipeline_config["uc_name"]
    )
    # // can simply be done with for loops
    for obj in pipeline_config["objectives"]:
        train_model_config["lgb_params"]["objective"] = obj
        train_model_task = train_model_step(
            input_folder=features_engineering_task.outputs["output_folder"],
            config=train_model_config,
        ).set_memory_limit("32G")

        # 5. Evaluate a model
        # The previous loaded config can also be used later in the DAG
        evaluate_model_task = evaluate_model_step(
            input_folder=features_engineering_task.outputs["output_folder"],
            model_artifact=train_model_task.outputs["model_artifact"],
            config=train_model_config,
        ).set_memory_limit("32G")

        with dsl.Condition(
            evaluate_model_task.outputs["FA"] >= pipeline_config["FA_THRESHOLD"],
            name=f"deploy_decision_{str(pipeline_config['FA_THRESHOLD']).replace('.', '_')}_threshold",
        ):
            # train with all data
            # update config to train on all data
            # 3. Engineer features
            # One can also decide to load component's config like this:
            features_engineering_config = load_component_config(
                "features_engineering", pipeline_config["uc_name"]
            )
            features_engineering_config["validation_start_date"] = pipeline_config[
                "validation_start_date"
            ]

            features_engineering_inference_task = features_engineering_step(
                input_folder=prepare_data_task.outputs["output_folder"],
                config=features_engineering_config,
            ).set_memory_limit("32G")

            train_model_task_inference = train_model_step(
                input_folder=features_engineering_inference_task.outputs[
                    "output_folder"
                ],
                config=train_model_config,
            ).set_memory_limit("32G")

            # deploy the model to a folder
            deploying_task = deploying_model_step(
                input_folder_models=train_model_task_inference.outputs[
                    "model_artifact"
                ],
                origin_bucket=pipeline_config["PIPELINE_ROOT"],
                bucket_models=pipeline_config["BUCKET_MODELS"],
            ).after(train_model_task_inference)

        # 6. Explain predictions
        explain_predictions_task = (
            explain_predictions_step(
                input_folder=features_engineering_task.outputs["output_folder"],
                model_artifact=train_model_task.outputs["model_artifact"],
                config=train_model_config,
            )
            .set_cpu_limit("8")
            .set_memory_limit("32G")
        )


@dsl.pipeline(description="Experiment with conditional deployment")
def inference_pipeline() -> None:
    """Pipeline for training and conditionnal model deployment.

    This pipeline run a repeatable training and evaluation task.
    It can optionaly deploy a model in a production environment
    if some conditions are met.
    """
    # A. The first step is to load the pipeline config file
    pipeline_config = load_pipeline_config()

    # The next step is to define the pipeline
    # read_data --> prepare_data --> feature_engineering --> inference --> write output(BQ)

    # 1. Getting the data from BQ
    # Constant inputs can be send in the pipeline's config file
    get_data_task = get_data_step(
        input_bucket_raw=pipeline_config["input_bucket_raw"],
    )

    # 2. Preparing the data
    preprocessing_config = load_component_config(
        "preprocessing", pipeline_config["uc_name"]
    )
    prepare_data_task = prepare_data_step(
        input_folder=get_data_task.outputs["output_folder"], config=preprocessing_config
    )

    # 3. Engineer features
    # One can also decide to load component's config like this:
    features_engineering_config = load_component_config(
        "features_engineering", pipeline_config["uc_name"]
    )
    features_engineering_task = features_engineering_step(
        input_folder=prepare_data_task.outputs["output_folder"],
        config=features_engineering_config,
    ).set_memory_limit("32G")

    train_model_config = load_component_config(
        "train_model", pipeline_config["uc_name"]
    )
    inference_task = using_deployed_model(
        input_folder=features_engineering_task.outputs["output_folder"],
        bucket_models=pipeline_config["BUCKET_MODELS"],
        config=train_model_config,
    ).after(features_engineering_task)

    # write_data_task = writing_data(
    #     input_folder=inference_task.outputs["output_folder"], config=pipeline_config
    # )
