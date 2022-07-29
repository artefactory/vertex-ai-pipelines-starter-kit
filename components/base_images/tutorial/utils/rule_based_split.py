def apply_rule_based_monthly_split(
    forecasts_df,
    train_df,
    train_fiscal_year_range,
    column_to_compute_split,
    month_launch_to_apply="all",
):
    ratios = train_df.copy()
    ratios = ratios[ratios.FiscalYear >= train_fiscal_year_range[0]]
    ratios = ratios[ratios.FiscalYear <= train_fiscal_year_range[1]]
    ratios = (
        ratios.groupby([column_to_compute_split, "Channel", "MonthSinceLaunch"])
        .agg({"Quantities": "sum"})
        .reset_index()
        .rename(columns={"Quantities": "total_per_month"})
    )
    ratios_total = (
        ratios.groupby([column_to_compute_split, "Channel"])
        .agg({"total_per_month": "sum"})
        .reset_index()
        .rename(columns={"total_per_month": "total"})
    )
    ratios = ratios.merge(
        ratios_total, on=[column_to_compute_split, "Channel"], how="left"
    )
    ratios["Coefficient"] = ratios["total_per_month"] / ratios["total"]

    forecasts_modified_df = forecasts_df.copy()
    total_per_channel = (
        forecasts_modified_df.groupby(["Article", "Channel"])
        .agg({"predicted": "sum"})
        .reset_index()
        .rename(columns={"predicted": "total_per_channel"})
    )

    forecasts_modified_df = forecasts_modified_df.merge(
        total_per_channel, on=["Article", "Channel"], how="left"
    )

    forecasts_modified_df = forecasts_modified_df.merge(
        ratios[[column_to_compute_split, "Channel", "MonthSinceLaunch", "Coefficient"]],
        how="left",
        on=[column_to_compute_split, "Channel", "MonthSinceLaunch"],
    )

    forecasts_modified_df["predicted"] = forecasts_modified_df.apply(
        _compute_rule_based_predictions, args=(month_launch_to_apply,), axis=1
    )
    forecasts_modified_df = forecasts_modified_df.drop(
        columns=["Coefficient", "total_per_channel"]
    )

    return forecasts_modified_df


def _compute_rule_based_predictions(row, month_launch_to_apply):
    if month_launch_to_apply == "all":
        return row["Coefficient"] * row["total_per_channel"]
    elif row["MonthLaunch"] in month_launch_to_apply:
        return row["Coefficient"] * row["total_per_channel"]
    else:
        return row["predicted"]
