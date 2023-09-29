import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.metadata import CandidateJoin


class BaseJoinMethod(BaseEstimator):
    def __init__(
        self,
        scenario_logger=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        task="regression",
    ) -> None:
        super().__init__()
        self.scenario_logger = scenario_logger
        self.model_parameters = model_parameters
        self.chosen_model = chosen_model
        self.target_column = target_column
        self.task = task

        if task not in ["regression", "classification"]:
            raise ValueError(f"Task {task} not supported.")

        self.model = None
        self.cat_features = None

    def build_model(self, X, cat_features=None):
        if self.chosen_model == "catboost":
            return self.build_catboost(cat_features)
        elif self.chosen_model == "linear":
            return self.build_linear()
        else:
            raise ValueError(f"Chosen model {self.chosen_model} is not recognized.")

    def build_catboost(self, cat_features):
        defaults = {
            "l2_leaf_reg": 0.01,
            "od_type": None,
            "od_wait": None,
            "iterations": 100,
            "verbose": 0,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)
        model = CatBoostRegressor(cat_features=cat_features, **parameters)

        return model

    def build_linear(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def get_cat_features(self, X):
        if type(X) == pl.DataFrame:
            cat_features = X.select(cs.string()).columns
        elif type(X) == pd.DataFrame:
            cat_features = X.select_dtypes("object").columns.to_list()
        else:
            raise TypeError(f"Type {type(X)} unsupported.")

        return cat_features

    @staticmethod
    def prepare_table(table):
        if type(table) == pd.DataFrame:
            table = pl.from_pandas(table)

        table = table.fill_null(value="null").fill_nan(value=np.nan)
        table = ju.cast_features(table)
        return table.to_pandas()


class BaseJoinWithCandidatesMethod(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_joins=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )
        self.candidate_joins = candidate_joins
        self.n_candidates = len(candidate_joins)

        _join_params = {"aggregation": "first"}
        _join_params.update(join_parameters)
        self.join_parameters = _join_params


class NoJoin(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )
        self.name = "nojoin"

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        cat_features = self.get_cat_features(X_train)
        self.model = self.build_model(X_train, cat_features)

        # TODO: This needs to be generalized
        X_train = self.prepare_table(X_train)
        self.model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))

    def predict(self, X):
        X = self.prepare_table(X)
        return self.model.predict(X)

    def transform(self, X):
        return X


class SingleJoin(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_join_mdata: CandidateJoin = None,  # dtype is "candidate_join.mdata"
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )
        _join_params = {"aggregation": "first"}
        _join_params.update(join_parameters)
        self.join_parameters = _join_params
        self.candidate_join_mdata = candidate_join_mdata
        (
            _,
            cnd_md,
            self.left_on,
            self.right_on,
        ) = candidate_join_mdata.get_join_information()
        self.candidate_table = pl.read_parquet(cnd_md["full_path"])

    def fit(self, X, y):
        X = self.prepare_table(X)
        merged_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            self.candidate_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_train = self.prepare_table(merged_train)
        self.cat_features = self.get_cat_features(merged_train)

        self.model = self.build_model(merged_train, self.cat_features)

        self.model.fit(merged_train, y)

    def predict(self, X):
        X = self.prepare_table(X)
        merged_test = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            self.candidate_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_test = self.prepare_table(merged_test)
        return self.model.predict(merged_test)


class HighestContainmentJoin(BaseJoinWithCandidatesMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_joins=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            chosen_model,
            model_parameters,
            join_parameters,
            task,
        )
        self.name = "highest_containment"
        self.candidate_ranking = None
        self.cat_features = []
        self.best_candidate_hash = None

    @staticmethod
    def _measure_containment(
        source_table: pl.DataFrame, cand_table: pl.DataFrame, left_on, right_on
    ):
        unique_source = source_table[left_on].unique()
        unique_cand = cand_table[right_on].unique()

        s1 = set(unique_source[left_on].to_series().to_list())
        s2 = set(unique_cand[right_on].to_series().to_list())
        return len(s1.intersection(s2)) / len(s1)

    def fit(self, X, y):
        containment_list = []
        for hash_, mdata in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="Training on candidates",
        ):
            _, cnd_md, left_on, right_on = mdata.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])
            # cnd_table = self.prepare_table(cnd_table)
            containment = self._measure_containment(
                pl.from_pandas(X), cnd_table, left_on, right_on
            )
            containment_list.append({"candidate": hash_, "containment": containment})

        self.candidate_ranking = pl.from_dicts(containment_list)
        self.best_candidate_hash = self.candidate_ranking.top_k(1, by="containment")[
            "candidate"
        ].item()
        mdata = self.candidate_joins[self.best_candidate_hash]
        _, best_cnd_md, left_on, right_on = mdata.get_join_information()
        best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])
        merged_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            best_cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_train = self.prepare_table(merged_train)
        self.cat_features = self.get_cat_features(merged_train)

        self.model = self.build_model(merged_train, self.cat_features)

        self.model.fit(merged_train, y)

    def predict(self, X):
        X = self.prepare_table(X)
        best_cnd_mdata = self.candidate_joins[self.best_candidate_hash]
        _, cnd_md, left_on, right_on = best_cnd_mdata.get_join_information()
        cnd_table = pl.read_parquet(cnd_md["full_path"])
        merged_test = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_test = self.prepare_table(merged_test)
        return self.model.predict(merged_test)


class BestSingleJoin(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_joins=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        task="regression",
        join_parameters=None,
    ) -> None:
        raise NotImplementedError
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )

        self.candidate_ranking = None

        self.best_candidate_hash = None
        self.best_candidate_r2 = -np.inf

    def get_best_candidates(self, top_k=None):
        if top_k is None:
            return self.candidate_ranking
        else:
            return self.candidate_ranking.limit(top_k)

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        ranking = []

        for hash_, mdata in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="Training on candidates",
        ):
            src_md, cnd_md, left_on, right_on = mdata.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])

            # TODO: dataframe typing needs to be fixed
            merged_train = ju.execute_join_with_aggregation(
                pl.from_pandas(X_train),
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix="_right",
            )

            merged_valid = ju.execute_join_with_aggregation(
                pl.from_pandas(X_valid),
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix="_right",
            )

            cat_features = merged_train.select(cs.string()).columns
            self.model = self.build_model(merged_train, cat_features)

            # TODO: This needs to be generalized
            self.model.fit(X=merged_train, y=y_train, eval_set=(merged_valid, y_valid))
            y_pred = self.model.predict(merged_valid)
            r2 = r2_score(y_valid, y_pred)

            ranking.append({"candidate": hash_, "r2": r2})

            if r2 > self.best_candidate_r2:
                self.best_candidate_r2 = r2
                self.best_candidate_hash = hash_

        self.candidate_ranking = pl.from_dicts(ranking).sort("r2", descending=True)

    def predict(self, X):
        best_cnd_mdata = self.candidate_joins[self.best_candidate_hash]
        src_md, cnd_md, left_on, right_on = best_cnd_mdata.get_join_information()
        cnd_table = pl.read_parquet(cnd_md["full_path"])
        merged_test = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )

        return self.model.predict(merged_test)

    def transform(self, X):
        pass


class FullJoin(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_joins=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )
        raise NotImplementedError
        self.candidate_joins = candidate_joins
        self.join_parameters = join_parameters

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def transform(self, X):
        pass


class GreedyGreedyJoin(BaseJoinMethod):
    pass


class OptimumGreedyJoin(BaseJoinMethod):
    pass
