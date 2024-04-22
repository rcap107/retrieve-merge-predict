"""
Figure 5(b): prediction performance with respect to containment, with regression plot.
"""
#%%
# %cd ~/bench
import matplotlib.pyplot as plt

#%%
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

import src.utils.plotting as plotting


#%%
def plot_reg(X, y, ax, label):
    model = LinearRegression()
    model.fit(X, y)
    ax.scatter(
        X + plotting.prepare_jitter(X.shape, offset_value=0, factor=0.0),
        y,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.plot(X, model.predict(X), label=label)
    ax.set_ylabel("Prediction score")
    ax.set_xlabel("Containment")


#%%
def prepare_data(df_raw, df_analysis):
    df_agg = (
        df_analysis.filter(pl.col("top_k") == 200)
        .group_by(
            [
                "retrieval_method",
                "data_lake_version",
                "table_name",
                "query_column",
                "aggregation",
            ]
        )
        .agg(
            pl.col("containment").mean().alias("avg_containment"),
            pl.col("containment").median().alias("median_containment"),
            pl.col("containment").top_k(30).mean().alias("top_30_avg_containment"),
            pl.col("containment").top_k(30).median().alias("top_30_median_containment"),
            pl.col("cnd_nrows").mean().alias("avg_cnd_nrows"),
            pl.col("cnd_nrows").median().alias("median_cnd_nrows"),
            pl.col("join_time").mean().alias("avg_join_time"),
            pl.col("join_time").median().alias("median_join_time"),
            pl.col("matched_rows").mean().alias("avg_matched_rows"),
            pl.col("matched_rows").median().alias("median_matched_rows"),
        )
        .sort("retrieval_method", "table_name")
    )

    res = df_raw.join(
        df_agg,
        left_on=["target_dl", "jd_method", "base_table", "query_column"],
        right_on=[
            "data_lake_version",
            "retrieval_method",
            "table_name",
            "query_column",
        ],
    )

    f = {"jd_method": "exact_matching", "chosen_model": "catboost"}
    r = res.filter(**f)
    X = r.select("avg_containment").to_numpy()
    y = r["r2score"].to_numpy()
    return X, y


#%%
df_raw_od = pl.read_parquet("results/overall/open_data_all_first.parquet")
df_raw_od = df_raw_od.filter(pl.col("estimator") != "nojoin")

df_analysis_od = pl.read_csv("analysis_query_results_open_data_us-fixed.csv")
df_analysis_od = df_analysis_od.with_columns(
    (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
)
#%%
df_raw_wn = pl.read_parquet("results/overall/wordnet_general_first.parquet")
df_raw_wn = df_raw_wn.filter(pl.col("estimator") != "nojoin")

df_analysis_wn = pl.read_csv("analysis_query_results.csv")
df_analysis_wn = df_analysis_wn.with_columns(
    (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
)

#%%
fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
X, y = prepare_data(df_raw_wn, df_analysis_wn)
plot_reg(X, y, ax, "YADL Wordnet")
X, y = prepare_data(df_raw_od, df_analysis_od)
plot_reg(X, y, ax, "Open Data")
h, labels = ax.get_legend_handles_labels()
fig.legend(
    h,
    labels,
    loc="upper left",
    fontsize=10,
    ncols=2,
    bbox_to_anchor=(0, 1.0, 1, 0.1),
    mode="expand",
)
fig.savefig("images/regplot.pdf", bbox_inches="tight")
fig.savefig("images/regplot.png", bbox_inches="tight")
# %%
