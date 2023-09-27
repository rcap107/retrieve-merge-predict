def get_model(model_name: str = "", model_parameters: dict = None):
    if model_name == "catboost":
        pass
    elif model_name == "linear":
        pass
    else:
        raise ValueError(f"Unknown model_name {model_name}")


def check_params_catboost(params: dict):
    params_dict = {
        "iterations": 100,
        "od_type": None,
        "od_wait": None,
        "l2_leaf_reg": 0.01,
        "verbose": 0,
    }
    if params is None:
        return params_dict
    else:
        params_dict.update(params)
        return params_dict
