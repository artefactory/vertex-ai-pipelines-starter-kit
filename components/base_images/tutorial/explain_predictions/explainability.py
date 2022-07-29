import pandas as pd
from shap import TreeExplainer


def compute_shap_value_df(
    explainer: TreeExplainer,
    explain_df: pd.DataFrame,
    all_ohe_features: list,
    categorical_features: list,
):
    """
    Function to compute shap values per article x channel.

    - get df of absolute shap values
    - sum encoded features shap values
    - take mean of shap to get average values per article, category, channel
    - scale shap values

    :param explainer: the shap model
    :param explain_df: the dataframe of features to explain
    :param all_ohe_features: one hot encoded feature names
    :param categorical_features: categorical feature names
    """
    shap_values = explainer(explain_df)
    raw_shap_df = _shap_to_df(shap_values, explain_df, all_ohe_features)

    scaled_shap_df = (
        raw_shap_df.pipe(
            _sum_ohe_shap_values,
            shap_features=all_ohe_features,
            categorical_features=categorical_features,
        )
        .pipe(_mean_shap_per_article)
        .pipe(_scale_shap_values)
    )
    return raw_shap_df, scaled_shap_df


def _shap_to_df(shap_values, explain_df, shap_features):
    return (
        pd.DataFrame(
            shap_values.values, columns=[f"shap_{col}" for col in shap_features]
        )
        .abs()
        .assign(Article=explain_df.index)
    )


def _sum_ohe_shap_values(shap_raw_df, shap_features, categorical_features):
    for col in categorical_features:
        ohe_features = [f"shap_{x}" for x in shap_features if x.startswith(col)]
        shap_raw_df[f"shap_{col}"] = shap_raw_df[ohe_features].sum(axis=1)
        shap_raw_df = shap_raw_df.drop(columns=ohe_features)
    return shap_raw_df


def _mean_shap_per_article(shap_raw_df):
    return shap_raw_df.groupby("Article").mean()


def _scale_shap_values(shap_raw_df):
    shap_raw_df["shap_sum"] = shap_raw_df.sum(axis=1)
    for col in shap_raw_df.columns:
        shap_raw_df[col] = (100 * shap_raw_df[col] / shap_raw_df["shap_sum"]).round(1)
    return shap_raw_df.drop(columns=["shap_sum"])
