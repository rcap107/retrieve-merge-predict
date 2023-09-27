import numpy as np
import polars as pl
import polars.selectors as cs
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split

import src.utils.joining as utils


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
        if X.shape[0] != y.shape[0]:
            raise ValueError

        cat_features = X.select_dtypes("category").columns.to_list()
        self.model = self.build_model(X, cat_features)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        # TODO: This needs to be generalized
        self.model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))

    def predict(self, X):
        return self.model.predict(X)


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


class SingleJoin(BaseJoinMethod):
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

        self.candidate_ranking = None

    def get_best_candidates(self, top_k=None):
        if top_k is None:
            return self.candidate_ranking
        else:
            return self.candidate_ranking.limit(top_k)

    def fit(X, y):
        for hash_, mdata in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="Training on candidates",
        ):
            src_md, cnd_md, left_on, right_on = mdata.get_join_information()
            cand_parameters = {
                "candidate_table": hash_,
                "index_name": index_name,
                "left_on": left_on,
                "right_on": right_on,
                "join_strategy": "single_join",
            }
            cnd_table = pl.read_parquet(cnd_md["full_path"])


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

        self.candidate_joins = candidate_joins
        self.join_parameters = join_parameters
