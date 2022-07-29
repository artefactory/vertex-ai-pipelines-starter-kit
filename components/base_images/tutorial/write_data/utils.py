from datetime import datetime

import numpy as np
import pandas as pd


def add_months(row):
    return row["FirstLaunchDate"] + pd.DateOffset(months=row["MonthSinceLaunch"] - 1)


def convert_yearmonth(row):
    return datetime.strptime(str(row["YearMonth"]) + "01", "%Y%m%d")


def get_prediction_date(df, novelties=True):
    if novelties:
        return df.assign(
            FirstLaunchDate=lambda x: pd.to_datetime(x["FirstLaunchDate"])
        ).assign(PredictionDate=lambda x: x.apply(add_months, axis=1))
    else:
        return df.assign(
            FirstLaunchDate=lambda x: pd.to_datetime(x["FirstLaunchDate"])
        ).assign(PredictionDate=lambda x: x.apply(convert_yearmonth, axis=1))


def get_fy_forecasts(df, agg_cols, novelties=True):
    df["FiscalYearLaunch"] = df["FiscalYearLaunch"].astype(int)
    if novelties:
        df["FiscalYearPredictionDate"] = np.vectorize(compute_fiscal_year)(
            df["PredictionDate"].dt.year,
            df["PredictionDate"].dt.month,
            df["AfterLaunch"],
        )
        df_grouped = df.loc[df["FiscalYearPredictionDate"] == df["FiscalYearLaunch"], :]
    else:
        df_grouped = df.copy()
    return (
        df_grouped.groupby(agg_cols, as_index=False)
        .agg({"predicted": sum})
        .rename(columns={"predicted": "forecasts_fy"})
    )


def get_yearly_forecasts(df, inference_agg_cols):
    return (
        df.groupby(inference_agg_cols, as_index=False)
        .agg({"predicted": sum})
        .rename(columns={"predicted": "forecasts_12m"})
    )


def compute_fiscal_year(year, month, after_launch):
    if after_launch == 1:
        if month <= 3:
            return year - 1
        else:
            return year
    elif after_launch == 0:
        return year
    else:
        raise NotImplementedError()


def pivot_shap(df):
    return (
        df.set_index(["Category", "Article", "Channel"])
        .stack()
        .reset_index(name="feature_importance")
        .rename(columns={"level_3": "features"})
        .assign(features=lambda x: x["features"].str.replace("shap_", ""))
    )


def process_shap_values(df, config):
    for feature_name, feature_list in config["features_to_add"].items():
        df = add_features(df, feature_name, feature_list)

    df = rename_features(df, config).sort_values(
        ["Article", "Channel", "feature_importance"], ascending=False
    )
    return df


def add_features(df, feature_name, feature_list):
    feature_df = df[df.features.isin(feature_list)]
    feature_df = (
        feature_df.groupby(["Category", "Article", "Channel"])
        .agg({"feature_importance": "sum"})
        .reset_index()
    )
    feature_df["features"] = feature_name
    other_features_df = df[~df.features.isin(feature_list)]
    df = pd.concat([other_features_df, feature_df]).sort_values(
        ["Category", "Channel", "Article"]
    )
    return df


def rename_features(df, config):
    feature_renaming = pd.DataFrame(
        config["features_to_rename"].items(),
        columns=["features", "renamed_features"],
    )

    df = df.merge(feature_renaming, on="features", how="left")

    if len(df[df.renamed_features.isnull()]) != 0:
        raise ValueError(
            f"Some values are not filled in config: {df[df.renamed_features.isnull()].features.unique()}"
        )

    df = df.drop(columns=["features"]).rename(columns={"renamed_features": "features"})

    return df
