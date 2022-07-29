import numpy as np


def reverse_scale_target(y_data, config):
    if config["model"]["target_scaling"] == "log":
        y_data = np.expm1(y_data).round()
    elif config["model"]["target_scaling"] == "normalize":
        y_data = y_data * config["model"]["mean_target"] + config["model"]["std_target"]
    elif config["model"]["target_scaling"].startswith("box_cox"):
        coefficient = float(config["model"]["target_scaling"].split(":")[1])
        y_data = (y_data * coefficient + 1).abs().pow(1 / coefficient).round()
    elif config["model"]["target_scaling"] == "no_scaling":
        pass
    else:
        raise NotImplementedError()
    return y_data
