from pathlib import Path
import polars as pl
import polars.selectors as cs
from skrub import AggJoiner
from glob import glob


def load_tables_from_path(path_to_tables: str | Path):
    """Given `path_to_tables`, load all tables in memory and return them as a list.

    Args:
        path_to_tables (str | Path): Path to the tables.
    """
    table_list = []
    # path expansion, search for tables
    for table_path in glob(path_to_tables):
        table = pl.read_parquet(table_path)
        # load tables in memory
        table_list.append(table)
    return table_list


def find_unique_values(table: pl.DataFrame, columns: list[str] = None) -> dict:
    """Given a dataframe and either a list of columns or None, find the unique values
    in each column in the list or all columns, then return the list of values as a dictionary
    with {column_name: [list_of_values]}

    Args:
        table (pl.DataFrame): Table to evaluate.
        columns (list[str], optional): List of columns to evaluate. If None, consider all columns.
    """
    # select the columns of interest
    if columns is not None:
        # error checking columns
        if len(columns) == 0:
            raise ValueError("No columns provided.")
        for col in columns:
            if col not in table.columns:
                raise pl.ColumnNotFoundError
    else:
        columns = table.columns

    # find the unique values
    unique_values = dict(
        table.select(cs.by_name(columns).implode().list.unique())
        .transpose(include_header=True)
        .rows()
    )
    # return the dictionary of unique values
    return unique_values


def measure_containment_tables(
    unique_values_base: dict, unique_values_candidate: dict
) -> list:
    """Given `unique_values_base` and `unique_values_candidate`, measure the containment for each pair.

    The result will be returned as a list with format `[(col_base_table_1, col_cand_table_1, similarity), (col_base_table_1, col_cand_table_2, similarity),]`

    Args:
        unique_values_base (dict): Dictionary that contains the set of unique values for each column in the base (query) table.
        unique_values_candidate (dict): Dictionary that contains the set of unique values for each column in the candidate table.
    """
    # for each value in unique_values_base, measure the containment for every value in unique_values_candidate

    # return the containment result as a list with format

    return containment_list


def measure_containment(unique_values_query: set, unique_values_candidate: set):
    """Given `unique_values_query` and `unique_values_candidate`, measure the Jaccard Containment of the query in the
    candidate column. Return only the containment

    Args:
        unique_values_query (set): Set of unique values in the query.
        unique_values_candidate (set): Set of unique values in the candidate column.
    """
    # measure containment
    set_query = set(unique_values_query)
    containment = len(set_query.intersection(set(unique_values_candidate))) / len(
        set_query
    )
    # return containment
    return containment


def prepare_ranking(containment_list: list[tuple], budget: int):
    """Sort the containment list and cut all candidates past a certain budget.

    Args:
        containment_list (list[tuple]): List of candidates with format (query_column, cand_table, cand_column, similarity).
        budget (int): Number of candidates to keep from the list.
    """

    # Sort the list

    # Somewhere here we might want to do some fancy filtering of the candidates in the ranking (with profiling)

    # Return `budget` candidates
    return ranking


def execute_join(
    base_table: pl.DataFrame, candidate_list: dict[pl.DataFrame], ranking: list[tuple]
):
    """Execute a full join between the base table and all candidates.

    Args:
        base_table (pl.DataFrame): _description_
        candidate_list (dict[pl.DataFrame]): _description_
        ranking (list[tuple]): _description_
    """

    # Use the Skrub MultiAggJoiner to join the base table and all candidates.

    # Return the joined table
    pass


class Discover:
    def __init__(
        self,
        path_tables: list,
        query_columns: list,
        path_cache: str | Path = None,
        budget=30,
    ) -> None:
        # Having more than 1 colummn is not supported yet.
        if len(query_columns > 1):
            raise NotImplementedError
        self.query_columns = query_columns
        # TODO: error checking budget
        self.budget = budget

        # load path from cache
        if path_cache is not None:
            # TODO: load ranking
            self.ranking = None
            pass
        # not loading from cache
        else:
            self.ranking = None
            # load list of tables
            self.candidate_tables = load_tables_from_path(path_tables)

            # find unique values for each table
            for tab_path in self.candidate_tables:
                table = pl.read_parquet(tab_path)
                self.unique_values_candidates = find_unique_values(table)

    def fit(self, X: pl.DataFrame, y=None):
        # error checking query columns
        for col in self.query_columns:
            if col not in X.columns:
                raise pl.ColumnNotFoundError(f"Column {col} not found in X.")
        # find unique values for the query columns
        unique_values_X = find_unique_values(X, self.query_columns)
        # measure containment
        containment_list = measure_containment_tables(
            unique_values_X, self.unique_values_candidates
        )
        # prepare ranking
        self.ranking = prepare_ranking(containment_list, budget=self.budget)

    def transform(self, X):
        join_tables = []
        join_keys = []
        for candidate in self.ranking:
            main_table_key, aux_table, aux_table_key, similarity = candidate
            table = pl.read_parquet(aux_table)
            join_tables.append(table)
            join_keys.append(aux_table_key)

        aggjoiner = AggJoiner(
            aux_table=join_tables,
            aux_key=join_keys,
            # TODO: write this properly
            main_key=main_table_key,
        )
        # execute join between X and the candidates
        joined_table = aggjoiner.fit_transform(X)

        # return joined table
        return joined_table

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    pass
