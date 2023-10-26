# %%
import itertools
from pprint import pprint

import toml
from sklearn.model_selection import ParameterGrid

# %%
base_config = toml.load("config/proto_config.toml")
base_config


# %%
def convert_to_list(thing):
    if isinstance(thing, dict):
        return {k: convert_to_list(v) for k, v in thing.items()}
    elif isinstance(thing, list):
        return thing
    else:
        return [thing]


# %%
def get_comb(config_dict):
    keys, values = zip(*config_dict.items())
    run_variants = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return run_variants


# %%
def flatten(key, this_dict):
    flattened_dict = {}
    for k, v in this_dict.items():
        if key == "":
            this_key = f"{k}"
        else:
            this_key = f"{key}.{k}"
        if isinstance(v, list):
            flattened_dict.update({this_key: v})
        else:
            flattened_dict.update(flatten(this_key, v))
    return flattened_dict


# %%
dd = convert_to_list(base_config)

r = flatten("", dd)
comb = get_comb(r)
# %%
this_dict = comb[0]


# %%
def pack(dict_to_pack):
    packed = {}
    for key, value in dict_to_pack.items():
        splits = key.split(".")
        n_splits = len(splits)
        if n_splits == 2:
            s0, s1 = splits
            if s0 not in packed:
                packed[s0] = {s1: value}
            else:
                packed[s0][s1] = value
        elif n_splits > 2:
            s0 = splits[0]
            pp = pack(
                {
                    k_.replace(s0 + ".", ""): v_
                    for k_, v_ in dict_to_pack.items()
                    if k_.startswith(s0)
                }
            )
            packed[s0] = pp
        else:
            raise ValueError
    return packed


pprint(pack(this_dict))

# %%
