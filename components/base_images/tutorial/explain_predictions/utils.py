from catboost import CatBoostRegressor


def extract_model(model_folder, scope_str, models_sub_folder):
    from_file = CatBoostRegressor()
    model = from_file.load_model(
        model_folder / models_sub_folder / f"{scope_str}_catboost.cbm", format="cbm"
    )
    return model
