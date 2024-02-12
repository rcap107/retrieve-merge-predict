#%%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2

import json
from pathlib import Path

import matplotlib.pyplot as plt

#%%
import polars as pl
import seaborn as sns

import src.utils.plotting as plotting
from src.utils.logging import read_and_process, read_logs

sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
run_path = "results/logs/0365-9u7yho23"
df_raw = read_logs(exp_name=None, exp_path=run_path)

#%%
ll = []
for f in Path(run_path, "json").iterdir():
    if f.is_file():
        with open(f, "r") as fp:
            loaded = json.load(fp)
            d = {"scenario_id": loaded["scenario_id"], "top_k": loaded["top_k"]}
            ll.append(d)
df_fix = pl.from_dicts(ll)

df = df_raw.join(df_fix, on="scenario_id")

f = {"chosen_model": "catboost", "estimator": ""}

# %%
fig, ax = plt.subplots(
    squeeze=True,
    #    figsize=(4,1.5), layout="constrained"
)
sns.boxplot(
    data=(
        df.filter(
            (pl.col("chosen_model") == "catboost") & (pl.col("estimator") != "nojoin")
        )
        .select(pl.col("top_k"), pl.col("r2score"))
        .with_columns(pl.col("top_k").cast(pl.String))
        .to_pandas()
    ),
    x="r2score",
    y="top_k",
    ax=ax,
)
# ax.set_ylabel("")
# ax.set_xlabel("Prediction performance")

# mapping = {
#     "wordnet_full": "YADL Wordnet",
#     "binary_update": "YADL Binary",
#     "open_data_us": "Open Data US",
# }

# ax.set_yticklabels(
#     [mapping[x.get_text()] for x in ax.get_yticklabels()]
# )

# fig.savefig("images/performance_data_lakes.png")
# fig.savefig("images/performance_data_lakes.pdf")
#%%
case = "dep"
results_full, results_depleted = read_and_process(df)

if case == "dep":
    current_results = results_depleted.clone()
    current_results = current_results.filter(pl.col("estimator") != "nojoin")
elif case == "full":
    current_results = results_full.clone()

# %%
var = "top_k"
scatter_d = "case"
plotting.draw_pair_comparison(
    current_results,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=True,
    savefig_type=["png", "pdf"],
    case=case,
    colormap_name="Set1",
    jitter_factor=0.01,
    qle=0.05,
    add_titles=False,
    # sorting_variable="top_k"
)

# %%
