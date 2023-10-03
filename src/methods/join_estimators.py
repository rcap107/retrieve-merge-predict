import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import NotFittedError, check_is_fitted
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import CandidateJoin


class BaseJoinMethod(BaseEstimator):
    """Base JoinEstimator class. This class is extended by the other estimators."""

    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        target_column: str = None,
        chosen_model: str = None,
        model_parameters: dict = None,
        with_validation: bool = True,
        task: str = "regression",
    ) -> None:
        """Base JoinEstimator class. This class is supposed to be extended by the
        other estimators. It uses the `fit`/`predict` paradigm to implement the
        evaluation of a join suggestion method.

        Args:
            scenario_logger (ScenarioLogger): ScenarioLogger object to be used for tracking
            the parameters and the current experiment.
            target_column (str, optional): Column that contains the target to be used for
            supervised learning. Defaults to None.
            chosen_model (str, optional): Which model to choose, either `catboost` or `linear`. Defaults to None.
            model_parameters (dict, optional): Parameters to be passed to the model. Defaults to None.
            task (str, optional): Task to be executed, either `regression` or `classification`. Defaults to "regression".

        Raises:
            ValueError: Raise ValueError if the value of `task` is not `regression` or `classification`.
            ValueError: Raise ValueError if the value of `chosen_model` is not `catboost` or `linear`.
        """
        super().__init__()

        if task not in ["regression", "classification"]:
            raise ValueError(f"Task {task} not supported.")

        if chosen_model not in ["catboost", "linear"]:
            raise ValueError(f"Model {chosen_model} not supported.")

        self.scenario_logger = scenario_logger
        self.model_parameters = model_parameters
        self.chosen_model = chosen_model
        self.target_column = target_column
        self.task = task
        self.joined_columns = None
        self.with_validation = with_validation

        self.model = None
        self.cat_features = None
        if self.chosen_model == "linear":
            self.with_validation = False
            self.cat_encoder = None
        else:
            self.cat_encoder = None

    def build_model(self, X=None):
        """Build the model using the parameters supplied during __init__.
        Note that multiple models may be created by a JoinEstimator during training.

        Args:
            X (pd.DataFrame, optional): Input dataframe used to build the linear model. Defaults to None.
            cat_features (list, optional): List of features to be considered categorical by CatBoost. Defaults to None.

        Raises:
            ValueError: Raise ValueError if the model name provided is not `linear` or `catboost`.

        Returns:
            The required model.
        """
        if self.chosen_model == "catboost":
            cat_features = self.get_cat_features(X)
            self.build_catboost(cat_features)
        elif self.chosen_model == "linear":
            self.build_linear()
        else:
            raise ValueError(f"Chosen model {self.model} is not recognized.")

    def build_catboost(self, cat_features):
        """Build a catboost model with the model parameters defined in the __init__
        and with the categorical features provided in `cat_features`.

        Args:
            cat_features (list): List of features to be considered categorical.

        Returns:
            CatBoostRegressor or CatBoostClassifier: The initialized model.
        """
        defaults = {
            "l2_leaf_reg": 0.01,
            "od_type": "Iter",
            "od_wait": 10,
            "iterations": 100,
            "verbose": 0,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)
        if self.task == "regression":
            self.model = CatBoostRegressor(cat_features=cat_features, **parameters)
        elif self.task == "classification":
            raise NotImplementedError
            self.model = CatBoostClassifier(cat_features=cat_features, **parameters)
        else:
            raise ValueError

    def build_linear(self):
        if self.task == "regression":
            self.model = LinearRegression()
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore")
        elif self.task == "classification":
            raise NotImplementedError
        else:
            raise ValueError

    def fit(self, X, y):
        """Abstract method, to be extended by other estimators. It should take a
        prepared table `X` and a suitable target array `y`. It will execute all
        the operations required to fit prepare and fit the model to be used later
        in the `predict` step.

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series): Target column.
        """
        raise NotImplementedError

    def predict(self, X):
        """Abstract method, to be extended by other estimators.

        It should take a prepared table `X`, which is fed to the prepared model.
        If additional operations are required, they are executed (e.g., joining
        X on a candidate table) before calling self.model.predict.

        Args:
            X (pd.DataFrame): Input dataframe.
        """
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

    def fit_model(self, X_train, y_train, X_valid=None, y_valid=None):
        if self.chosen_model == "linear":
            assert type(self.model) == LinearRegression
            X_train = self.prepare_table(X_train)
            self.cat_encoder.fit(X_train)
            X_enc = self.cat_encoder.transform(X_train)
            self.model.fit(X=X_enc, y=y_train)
        elif self.chosen_model == "catboost":
            assert (type(self.model) == CatBoostRegressor) or (
                type(self.model) == CatBoostClassifier
            )
            X_train = self.prepare_table(X_train)

            if self.with_validation:
                X_valid = self.prepare_table(X_valid)
                self.model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
            else:
                self.model.fit(X=X_train, y=y_train)
        else:
            raise ValueError

    def predict_model(self, X):
        if self.chosen_model == "linear":
            X_enc = self.cat_encoder.transform(X)
            y_pred = self.model.predict(X_enc)
        elif self.chosen_model == "catboost":
            y_pred = self.model.predict(X)
        else:
            raise ValueError
        return y_pred

    def get_estimator_parameters(self):
        raise NotImplementedError

    def get_additional_info(self):
        raise NotImplementedError

    def retrain_cat_encoder(self):
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit_cat_enc(self, table):
        if self.chosen_model == "linear":
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore")
            self.cat_encoder.fit(table)
        else:
            return None

    def prepare_table(self, table):
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

    def _execute_joins(self, X_train, X_valid=None):
        merged_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X_train),
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                pl.from_pandas(X_valid),
                self.best_cnd_table,
                left_on=self.left_on,
                right_on=self.right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid
        else:
            return merged_train


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
        self.joined_columns = len(X.columns)

        self.build_model(X)

        self.fit_model(X, y)

    def predict(self, X):
        return self.predict_model(X)

    def transform(self, X):
        return X

    def get_estimator_parameters(self):
        return {"estimator": self.name}

    def get_additional_info(self):
        return {
            "best_candidate_hash": "nojoin",
            "left_on": "",
            "right_on": "",
            "joined_columns": self.joined_columns,
        }


class SingleJoin(BaseJoinMethod):
    def __init__(
        self,
        scenario_logger=None,
        cand_join_mdata: CandidateJoin = None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
    ) -> None:
        super().__init__(
            scenario_logger, target_column, chosen_model, model_parameters, task
        )
        self.name = "single_join"
        _join_params = {"aggregation": "first"}
        _join_params.update(join_parameters)
        self.join_parameters = _join_params
        self.candidate_join_mdata = cand_join_mdata
        (
            _,
            cnd_md,
            self.left_on,
            self.right_on,
        ) = cand_join_mdata.get_join_information()
        self.candidate_table = pl.read_parquet(cnd_md["full_path"])
        self.best_candidate_hash = cnd_md["hash"]

    def fit(self, X, y):
        self.fit_cat_enc(X)
        if self.with_validation:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        else:
            X_train = X
            y_train = y
        merged_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X_train),
            self.candidate_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_train = self.prepare_table(merged_train)
        self.joined_columns = len(merged_train.columns)
        self.cat_features = self.get_cat_features(merged_train)

        if self.with_validation:
            merged_valid = ju.execute_join_with_aggregation(
                pl.from_pandas(X_valid),
                self.candidate_table,
                left_on=self.left_on,
                right_on=self.right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix="_right",
            )
            merged_valid = self.prepare_table(merged_valid)

        self.model = self.build_model(merged_train)

        self.fit_model(X_train, y_train, X_valid, y_valid)

        if self.with_validation:
            self.model.fit(merged_train, y_train, eval_set=(merged_valid, y_valid))
        else:
            self.model.fit(merged_train, y_train)

    def predict(self, X):
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

    def get_estimator_parameters(self):
        return {"estimator": self.name, **self.join_parameters}

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_candidate_hash,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "joined_columns": self.joined_columns,
        }


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
        self.best_cnd_hash = None
        self.best_cnd_table = None
        self.left_on = self.right_on = None

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
        # Find the exact containment ranking
        containment_list = []
        for hash_, mdata in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="HighestContainment",
            position=0,
        ):
            _, cnd_md, left_on, right_on = mdata.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])
            # cnd_table = self.prepare_table(cnd_table)
            containment = self._measure_containment(
                pl.from_pandas(X), cnd_table, left_on, right_on
            )
            containment_list.append({"candidate": hash_, "containment": containment})
        self.candidate_ranking = pl.from_dicts(containment_list)
        # Select the top-1 candidate
        self.best_cnd_hash = self.candidate_ranking.top_k(1, by="containment")[
            "candidate"
        ].item()

        mdata = self.candidate_joins[self.best_cnd_hash]
        _, best_cnd_md, self.left_on, self.right_on = mdata.get_join_information()
        self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])

        if self.with_validation:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
            merged_train, merged_valid = self._execute_joins(X_train, X_valid)
            self.joined_columns = len(merged_train.columns)
            self.build_model(merged_train)
            self.fit_model(merged_train, y_train, merged_valid, y_valid)
        else:
            merged_train = self._execute_joins(X)
            self.joined_columns = len(merged_train.columns)
            self.build_model(merged_train)
            self.fit_model(merged_train, y)

    def predict(self, X):
        merged_test = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_test = self.prepare_table(merged_test)
        return self.predict_model(merged_test)

    def _execute_joins(self, X_train, X_valid=None):
        merged_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X_train),
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                pl.from_pandas(X_valid),
                self.best_cnd_table,
                left_on=self.left_on,
                right_on=self.right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid
        else:
            return merged_train

    def get_estimator_parameters(self):
        return {"estimator": self.name, **self.join_parameters}

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cnd_hash,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "joined_columns": self.joined_columns,
        }


class BestSingleJoin(BaseJoinWithCandidatesMethod):
    def __init__(
        self,
        scenario_logger=None,
        candidate_joins=None,
        target_column=None,
        chosen_model=None,
        model_parameters=None,
        join_parameters=None,
        task="regression",
        valid_size=0.2,
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
        self.name = "best_single_join"
        self.candidate_ranking = None

        self.best_cnd_hash = None
        self.best_cnd_r2 = -np.inf
        self.best_cnd_table = None
        self.left_on = self.right_on = None
        self.valid_size = valid_size

    def get_best_candidates(self, top_k=None):
        if top_k is None:
            return self.candidate_ranking
        else:
            return self.candidate_ranking.limit(top_k)

    def fit(self, X, y):
        self.fit_cat_enc(X)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.valid_size
        )

        ranking = []

        for hash_, mdata in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="BestSingleJoin",
            position=0,
        ):
            _, cnd_md, left_on, right_on = mdata.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])

            merged_train = ju.execute_join_with_aggregation(
                pl.from_pandas(X_train),
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )

            merged_valid = ju.execute_join_with_aggregation(
                pl.from_pandas(X_valid),
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )

            merged_train = self.prepare_table(merged_train)
            merged_valid = self.prepare_table(merged_valid)
            self.model = self.build_model(merged_train)

            self.fit_model(merged_train, y_train, merged_valid, y_valid)

            y_pred = self.model.predict(merged_valid)
            r2 = r2_score(y_valid, y_pred)

            ranking.append({"candidate": hash_, "r2": r2})

            if r2 > self.best_cnd_r2:
                self.best_cnd_r2 = r2
                self.best_cnd_hash = hash_

        # RETRAINING THE MODEL
        best_join_mdata = self.candidate_joins[self.best_cnd_hash]
        (
            _,
            best_cnd_md,
            self.left_on,
            self.right_on,
        ) = best_join_mdata.get_join_information()
        self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])
        best_train = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        self.retrain_cat_encoder()
        self.joined_columns = len(best_train.columns)
        self.fit_cat_enc(best_train)
        best_train = self.prepare_table(best_train)
        self.model = self.build_model(best_train)
        self.model.fit(best_train, y)

        self.candidate_ranking = pl.from_dicts(ranking).sort("r2", descending=True)

    def predict(self, X):
        merged_test = ju.execute_join_with_aggregation(
            pl.from_pandas(X),
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        merged_test = self.prepare_table(merged_test)
        return self.model.predict(merged_test)

    def transform(self, X):
        pass

    def get_estimator_parameters(self):
        return {"estimator": self.name, **self.join_parameters}

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cnd_hash,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "joined_columns": self.joined_columns,
        }


class FullJoin(BaseJoinWithCandidatesMethod):
    # TODO: add a note mentioning that the joins in `candidate_joins` have been
    # vetted in some previous step and that this estimator will join them all
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
        self.name = "full_join"
        if self.join_parameters["aggregation"] == "dfs":
            print("Full join not available with DFS.")
            return None

    def fit(self, X, y):
        self.fit_cat_enc(X)
        if self.with_validation:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        else:
            X_train = X
            y_train = y
        merged_train = pl.from_pandas(X_train).clone().lazy()
        merged_train = ju.execute_join_all_candidates(
            merged_train, self.candidate_joins, self.join_parameters["aggregation"]
        )
        self.joined_columns = len(merged_train.columns)

        merged_train = self.prepare_table(merged_train)

        if self.with_validation:
            merged_valid = pl.from_pandas(X_valid).clone().lazy()
            merged_valid = ju.execute_join_all_candidates(
                merged_valid, self.candidate_joins, self.join_parameters["aggregation"]
            )
            merged_valid = self.prepare_table(merged_valid)

        self.model = self.build_model(merged_train)

        if self.with_validation:
            self.model.fit(merged_train, y_train, eval_set=(merged_valid, y_valid))
        else:
            self.model.fit(merged_train, y_train)

    def predict(self, X):
        merged_test = pl.from_pandas(X).clone().lazy()
        merged_test = ju.execute_join_all_candidates(
            merged_test, self.candidate_joins, self.join_parameters["aggregation"]
        )
        merged_test = self.prepare_table(merged_test)
        return self.model.predict(merged_test)

    def transform(self, X):
        pass

    def get_estimator_parameters(self):
        return {"estimator": self.name, **self.join_parameters}

    def get_additional_info(self):
        # TODO: add the list of all left_on, right_on
        return {
            "best_candidate_hash": "full_join",
            "left_on": "",
            "right_on": "",
            "joined_columns": self.joined_columns,
        }


class StepwiseGreedyJoin(BaseJoinMethod):
    pass


class OptimumGreedyJoin(BaseJoinMethod):
    pass
