"""
This script gathers all the constants that tare used across different scripts
(mainly for plotting) to have consistent ordering and formatting of labels across
the project.
"""

# Supported models
SUPPORTED_MODELS = ["catboost", "realmlp", "resnet", "ridge", "ridgecv"]

# Supported retrieval methods
SUPPORTED_RETRIEVAL_METHODS = ["exact_matching", "minhash", "minhash_hybrid", "starmie"]


# Grouping keys used to find the "difference from the mean"
GROUPING_KEYS = [
    "jd_method",
    "estimator",
    "chosen_model",
    "target_dl",
    "base_table",
    "aggregation",
    "fold_id",
]

# Colormaps used for the scatterplots
COLORMAP_DATALAKE_MAPPING = {
    "binary_update": "gray",
    "wordnet_full": "spring",
    "open_data_us": "winter",
    "wordnet_vldb_10": "cool",
    "wordnet_vldb_50": "seismic",
}

# Formatting the labels in the big legend, also used for filtering in some places.
LEGEND_LABELS = {
    "company_employees": "Company Employees",
    "housing_prices": "Housing Prices",
    "movies_large": "Movie Revenue",
    # "movies_vote": "Movie Ratings",
    "us_accidents_2021": "US Accidents 2021",
    "us_accidents_large": "US Accidents Large",
    "us_county_population": "US County Population",
    "us_elections": "US Elections",
    "schools": "Schools",
}

# Used for manual ordering. Labels will be printed in the exact order reported here.
ORDER_MAPPING = {
    "estimator": [
        "nojoin",
        "top_k_full_join",
        "highest_containment",
        "full_join",
        "best_single_join",
        "stepwise_greedy_join",
    ],
    "estimator_comp": [
        "stepwise_greedy_join",
        "full_join",
        "best_single_join",
        "highest_containment",
    ],
    "jd_method": ["exact_matching", "minhash", "minhash_hybrid", "starmie"],
    "target_dl": [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "open_data_us",
    ],
}

# Exhaustive list of all the possible values that can be plotted, and the label
# that they should be assigned for the legend and ticklabels.
LABEL_MAPPING = {
    "base_table": {
        "company_employees-depleted_name-open_data": "Employees",
        "company_employees-yadl-depleted": "Employees",
        "company_employees-yadl": "Employees",
        "company_employees-wordnet_vldb_10": "Employees",
        "company_employees-open_data": "Employees",
        "movies-yadl": "Movie Revenue",
        "movies-open_data": "Movie Revenue",
        "movies_vote-yadl": "Movie Vote",
        "movies_vote-depleted_title-open_data": "Movie Vote",
        "movies-depleted_title-open_data": "Movie Revenue",
        "movies-yadl-depleted": "Movie Revenue",
        "movies_vote-yadl-depleted": "Movie Vote",
        "housing_prices-yadl-depleted": "Housing Prices",
        "housing_prices-depleted_County-open_data": "Housing Prices",
        "housing_prices-yadl": "Housing Prices",
        "housing_prices-open_data": "Housing Prices",
        "us_accidents-yadl-depleted": "US Accidents",
        "us_elections-yadl-depleted": "US Elections",
        "us_county_population-depleted-yadl": "US County Population",
        "us_county_population-yadl-depleted": "US County Population",
        "us_accidents-yadl": "US Accidents",
        "us_accidents-depleted_County-open_data": "US Accidents",
        "us_accidents-open_data": "US Accidents",
        "us_elections-depleted_county_name-open_data": "US Elections",
        "us_elections-yadl": "US Elections",
        "us_elections-open_data": "US Elections",
        "schools-open_data_us": "Schools",
        "schools-open_data": "Schools",
    },
    "case": {
        "company_employees-binary_update": "Employees",
        "company_employees-open_data_us": "Employees",
        "company_employees-wordnet_full": "Employees",
        "company_employees-wordnet_vldb_3": "Employees",
        "company_employees-wordnet_vldb_10": "Employees",
        "company_employees-wordnet_vldb_50": "Employees",
        "housing_prices-binary_update": "Housing Prices",
        "housing_prices-open_data_us": "Housing Prices",
        "housing_prices-wordnet_full": "Housing Prices",
        "housing_prices-wordnet_vldb_3": "Housing Prices",
        "housing_prices-wordnet_vldb_10": "Housing Prices",
        "housing_prices-wordnet_vldb_50": "Housing Prices",
        "movies-binary_update": "Movie Revenue",
        "movies-wordnet_full": "Movie Revenue",
        "movies-open_data_us": "Movie Revenue",
        "movies-wordnet_vldb_3": "Movie Revenue",
        "movies-wordnet_vldb_10": "Movie Revenue",
        "movies-wordnet_vldb_50": "Movie Revenue",
        "movies_large-binary_update": "Movie Revenue",
        "movies_large-wordnet_full": "Movie Revenue",
        "movies_large-open_data_us": "Movie Revenue",
        "movies_large-wordnet_vldb_3": "Movie Revenue",
        "movies_large-wordnet_vldb_10": "Movie Revenue",
        "movies_large-wordnet_vldb_50": "Movie Revenue",
        "movies_vote-binary_update": "Movie Vote",
        "movies_vote-open_data_us": "Movie Vote",
        "movies_vote-wordnet_full": "Movie Vote",
        "movies_vote-wordnet_vldb_3": "Movie Vote",
        "movies_vote-wordnet_vldb_10": "Movie Vote",
        "movies_vote-wordnet_vldb_50": "Movie Vote",
        "movies_vote_large-binary_update": "Movie Vote",
        "movies_vote_large-open_data_us": "Movie Vote",
        "movies_vote_large-wordnet_full": "Movie Vote",
        "movies_vote_large-wordnet_vldb_3": "Movie Vote",
        "movies_vote_large-wordnet_vldb_10": "Movie Vote",
        "movies_vote_large-wordnet_vldb_50": "Movie Vote",
        "us_accidents-binary_update": "US Accidents",
        "us_accidents-open_data_us": "US Accidents",
        "us_accidents-wordnet_full": "US Accidents",
        "us_accidents-wordnet_vldb_3": "US Accidents",
        "us_accidents-wordnet_vldb_10": "US Accidents",
        "us_accidents_large-binary_update": "US Accidents",
        "us_accidents_large-open_data_us": "US Accidents",
        "us_accidents_large-wordnet_full": "US Accidents",
        "us_accidents_large-wordnet_vldb_3": "US Accidents",
        "us_accidents_large-wordnet_vldb_10": "US Accidents",
        "us_accidents_large-wordnet_vldb_50": "US Accidents",
        "us_accidents_2021-binary_update": "US Accidents",
        "us_accidents_2021-open_data_us": "US Accidents",
        "us_accidents_2021-wordnet_full": "US Accidents",
        "us_accidents_2021-wordnet_vldb_3": "US Accidents",
        "us_accidents_2021-wordnet_vldb_10": "US Accidents",
        "us_accidents_2021-wordnet_vldb_50": "US Accidents",
        "us_elections-open_data_us": "US Elections",
        "us_elections-binary_update": "US Elections",
        "us_elections-wordnet_full": "US Elections",
        "us_elections-wordnet_vldb_3": "US Elections",
        "us_elections-wordnet_vldb_10": "US Elections",
        "us_elections-wordnet_vldb_50": "US Elections",
        "us_county_population-binary_update": "US County Population",
        "us_county_population-wordnet_full": "US County Population",
        "us_county_population-wordnet_vldb_3": "US County Population",
        "us_county_population-wordnet_vldb_10": "US County Population",
        "us_county_population-wordnet_vldb_50": "US County Population",
        "schools-open_data_us": "Schools",
    },
    "jd_method": {
        "exact_matching": "Exact",
        "minhash": "MinHash",
        "starmie": "Starmie",
        "minhash_hybrid": "H. MinHash",
    },
    "chosen_model": {
        "catboost": "CatBoost",
        "ridgecv": "RidgeCV",
        "realmlp": "RealMLP",
        "resnet": "ResNet",
    },
    "estimator": {
        "full_join": "Full Join",
        "best_single_join": "Best Single J.",
        # "stepwise_greedy_join": "Step. Greedy Join",
        "stepwise_greedy_join": "Step. Greedy",
        "highest_containment": "Highest JC J.",
        # "highest_containment": "Highest JC Join",
        "nojoin": "No Join",
        "top_k_full_join": r"Top-$1$ Join",
    },
    "variables": {
        "estimator": "Selector",
        "jd_method": "Retrieval method",
        "chosen_model": "ML model",
        "base_table": "Base table",
    },
    "aggregation": {"first": "Any", "mean": "Mean", "dfs": "DFS"},
    "budget_amount": {10: 10, 30: 30, 100: 100},
    "target_dl": {
        "binary_update": "Binary",
        "wordnet_full": "YADL Base",
        "open_data_us": "Open Data",
        "wordnet_vldb_10": "YADL 10k",
        "wordnet_vldb_3": "YADL 3k",
        "wordnet_vldb_50": "YADL 50k",
    },
    "top_k": {
        10: "10",
        30: "30",
        100: "100",
    },
    "single_label": {"chosen_model": "Catboost\nVS Linear"},
}
