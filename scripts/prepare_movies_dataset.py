# %%
# %cd ..

# %%
import polars as pl
import xgboost
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

import os
# %%
os.chdir(Path("~/work/benchmark-join-suggestions").expanduser())
data_dir = Path("data/source_tables/ken_datasets/the-movies-dataset/")
final_table_path = Path(data_dir, "movies-revenues.parquet")

# %%
df = pd.read_csv(Path(data_dir, "movies_metadata.csv"), engine="python")
df = df.drop(
    [
        "belongs_to_collection",
        "homepage",
        "imdb_id",
        "overview",
        "tagline",
        "poster_path",
    ],
    axis=1,
)


# %%


def clean_genres(ll):
    g = ast.literal_eval(ll)
    try:
        l1 = g[0]["name"]
        return l1
    except IndexError:
        return ""


def clean_production_companies(ll):
    try:
        g = ast.literal_eval(ll)
    except ValueError:
        return ""
    except SyntaxError:
        print(ll)
    try:
        l1 = g[0]["name"]
        return l1
    except IndexError:
        return ""
    except TypeError:
        return ""


def clean_production_country(ll):
    try:
        g = ast.literal_eval(ll)
    except ValueError:
        return ""
    try:
        l1 = g[0]["iso_3166_1"]
        return l1
    except IndexError:
        return ""
    except TypeError:
        return ""


def clean_spoken_language(ll):
    try:
        g = ast.literal_eval(ll)
    except ValueError:
        return ""
    try:
        l1 = g[0]["name"]
        return l1
    except IndexError:
        return ""
    except TypeError:
        return ""


# %%

df.genres = df.genres.apply(clean_genres)
df.production_companies = df.production_companies.apply(clean_production_companies)
df.production_countries = df.production_countries.apply(clean_production_country)
df.spoken_languages = df.spoken_languages.apply(clean_spoken_language)


# %%
df = df.dropna(subset=["title", "release_date"])

# %%
df["release_date"] = df["release_date"].apply(lambda x: str(x[:4]))

# %%
df = df.loc[df["revenue"] > 0]

# %%
df_mapped = pd.read_parquet(Path(data_dir, "movies-revenues.parquet"))

# %%
subset = df_mapped[["id", "col_to_embed", "revenue_right"]]

# %%
merged = df.merge(subset, how="inner", on=["id"])

# %%
merged["target"] = merged["revenue_right"]
df_to_save = merged.drop(["revenue_right", "revenue"], axis=1)

df_to_save.to_parquet(Path(data_dir, "movies-prepared.parquet"), index=False)

# %%
pl.read_parquet(Path(data_dir, "movies-prepared.parquet"))

# %%
