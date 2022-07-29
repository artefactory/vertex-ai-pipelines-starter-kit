import os

from kfp.v2.dsl import Dataset, Input, Output, component


@component(
    base_image=f"eu.gcr.io/{os.getenv('GCP_PROJECT_ID')}/base_image_{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
)
def prepare_data_step(
    input_folder: Input[Dataset], output_folder: Output[Dataset], config: dict = None
) -> None:
    from pathlib import Path

    import pandas as pd

    from components.base_images.tutorial.preprocessing.prepare import (
        create_release_date_column,
        filter_out_sales_before_release_date,
        format_calendar_data,
        melt_sales_data,
        merge_sales_calendar,
        merge_sales_prices,
        reduce_memory,
    )

    input_folder = Path(input_folder.path)
    output_folder = Path(output_folder.path)
    output_folder.mkdir(parents=True, exist_ok=True)

    sales_train = pd.read_csv(input_folder / "sales_train.csv")
    sales_inference = pd.read_csv(input_folder / "sales_inference.csv")
    prices = pd.read_csv(input_folder / "prices.csv")
    calendar = pd.read_csv(input_folder / "calendar.csv")

    # Select store CA_1 only
    sales_train = sales_train.loc[sales_train["store_id"] == "CA_1"]

    df = sales_train.merge(sales_inference, how="left", on=config["id_cols"])

    df = (
        df.pipe(melt_sales_data, config["id_cols"])
        .pipe(reduce_memory, config["id_cols"])
        .pipe(create_release_date_column, prices=prices)
        .pipe(merge_sales_calendar, calendar=format_calendar_data(calendar))
        .pipe(filter_out_sales_before_release_date)
        .pipe(merge_sales_prices, prices=prices)
    )

    for col in config["id_cols"]:
        df[col] = pd.Categorical(df[col])

    df.to_csv(output_folder / "data_prepared.csv")
