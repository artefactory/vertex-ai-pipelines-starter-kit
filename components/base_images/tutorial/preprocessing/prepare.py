import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1048576
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type) in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1048576
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def merge_by_concat(df1, df2, merge_on):
    merged_df = df1[merge_on]
    merged_df = merged_df.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_df) if col not in merge_on]
    df1 = pd.concat([df1, merged_df[new_columns]], axis=1)
    return df1


def format_calendar_data(calendar):
    calendar["date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d")
    cols_calendar1 = ["date", "d", "wm_yr_wk"]
    cols_calendar2 = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    calendar = calendar[cols_calendar1 + cols_calendar2]
    for col in cols_calendar2:
        calendar[col] = calendar[col].astype("category")
    return calendar


def melt_sales_data(sales, id_cols):
    sales = sales.melt(id_vars=id_cols, var_name="date", value_name="sales")
    return sales


def reduce_memory(sales, id_cols):
    for col in id_cols:
        sales[col] = sales[col].astype("category")
    sales = reduce_mem_usage(sales)
    return sales


def create_release_date_column(sales, prices):
    release_df = (
        prices.groupby(["store_id", "item_id"])["wm_yr_wk"].agg(["min"]).reset_index()
    )
    release_df.columns = ["store_id", "item_id", "release"]
    sales = merge_by_concat(sales, release_df, ["store_id", "item_id"])
    return sales


def merge_sales_calendar(sales, calendar):
    sales = sales.rename(columns={"date": "d"})
    sales = merge_by_concat(sales, calendar, ["d"])
    return sales


def filter_out_sales_before_release_date(sales):
    # Filter out rows for which date < item release date (not "real zeros")
    sales = sales[sales["wm_yr_wk"] >= sales["release"]]
    sales = sales.reset_index(drop=True)
    # Normalize release column
    sales["release"] = sales["release"] - sales["release"].min()
    sales["release"] = sales["release"].astype(np.int16)
    return sales


def merge_sales_prices(sales, prices):
    sales = merge_by_concat(sales, prices, ["store_id", "item_id", "wm_yr_wk"])
    sales = sales.drop(["d"], axis=1)
    return sales
