LABEL_MAPPING = {
    "base_table": {
        "movies_vote-depleted_title-open_data": "(D) Movies Vote",
        "us_elections-depleted_county_name-open_data": "(D) US Elections",
        "us_accidents-depleted_County-open_data": "(D) US Accidents",
        "movies-depleted_title-open_data": "(D) Movies",
        "company_employees-depleted_name-open_data": "(D) Employees",
        "company-employees-yadl-depleted": "(D) Employees",
        "company_employees-yadl-depleted": "(D) Employees",
        "movies-yadl-depleted": "(D) Movies",
        "movies-vote-yadl-depleted": "(D) Movies Vote",
        "movies_vote-yadl-depleted": "(D) Movies Vote",
        "housing-prices-yadl-depleted": "(D) Housing Prices",
        "housing_prices-yadl-depleted": "(D) Housing Prices",
        "us-accidents-yadl-depleted": "(D) US Accidents",
        "us_accidents-yadl-depleted": "(D) US Accidents",
        "us-elections-yadl-depleted": "(D) US Elections",
        "us_elections-yadl-depleted": "(D) US Elections",
        "us_county_population-depleted-yadl": "(D) US County Population",
        "us_county_population-yadl-depleted": "(D) US County Population",
        "company-employees-yadl": "Employees",
        "company_employees-yadl": "Employees",
        "movies-yadl": "Movies",
        "movies-vote-yadl": "Movies Vote",
        "movies_vote-yadl": "Movies Vote",
        "housing-prices-yadl": "Housing Prices",
        "housing_prices-yadl": "Housing Prices",
        "us-accidents-yadl": "US Accidents",
        "us_accidents-yadl": "US Accidents",
        "us-elections-yadl": "US Elections",
        "us_elections-yadl": "US Elections",
    }
}

scatterplot_dimension = "base_table"
plotting_variable = "estimator"
colormap_name = "Set1"
df = pl.read_parquet(Path(dest_path, "wordnet_open_data_first.parquet"))
# Prepare the labels for the scatter plot and the corresponding colors.
df = df.with_columns(
    pl.col(scatterplot_dimension)
    .replace(LABEL_MAPPING[scatterplot_dimension])
    .alias("scatterplot_label")
)
scatterplot_labels = (
    df.group_by(pl.col(scatterplot_dimension))
    .agg(pl.mean(plotting_variable))
    .sort(plotting_variable)
    .select(pl.col(scatterplot_dimension).unique())
    .to_numpy()
    .squeeze()
)

# labels = [LABEL_MAPPING[scatterplot_dimension][v] for v in scatterplot_labels]
colors = plt.colormaps[colormap_name].resampled(len(scatterplot_labels)).colors
scatterplot_mapping = dict(
    zip(
        scatterplot_labels,
        colors,
    )
)

df = df.with_columns(
    pl.col(scatterplot_dimension)
    .replace(LABEL_MAPPING[scatterplot_dimension])
    .alias("scatterplot_label")
)

SCATTERPLOT_LABELS = [
    "(D) Employees",
    "(D) Housing Prices",
    "(D) Movies",
    "(D) Movies Vote",
    "(D) US Accidents",
    "(D) US County Population",
    "(D) US Elections",
]

colors = plt.colormaps[colormap_name].resampled(len(SCATTERPLOT_LABELS)).colors

pl.DataFrame({"scatterplot_label": SCATTERPLOT_LABELS, "colors": colors})

df.select(["base_table", "scatterplot_label"]).unique().join(
    pl.DataFrame({"scatterplot_label": SCATTERPLOT_LABELS, "colors": colors}),
    on="scatterplot_label",
)
