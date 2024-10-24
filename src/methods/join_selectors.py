import datetime as dt
import random
import warnings
from collections import deque
from itertools import islice
from multiprocessing import cpu_count
from typing import Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostClassifier, CatBoostRegressor
from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier,
    RealMLP_TD_Regressor,
    Resnet_RTDL_D_Classifier,
    Resnet_RTDL_D_Regressor,
)
from sklearn.base import BaseEstimator
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import (  # HistGradientBoostingClassifier,; HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skrub import MinHashEncoder, TableVectorizer, tabular_learner
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import CandidateJoin
from src.utils.constants import SUPPORTED_MODELS

warnings.filterwarnings("ignore")

MAX_THREADS = min([cpu_count(), 32])

SUPPORTED_BUDGET_TYPES = ["iterations"]
SUPPORTED_RANKING_METHODS = ["containment"]

CACHE_PATH = "results/cache"

# For reproducibility
CATBOOST_RANDOM_SEED = 42


def measure_containment(
    source_table: pl.DataFrame, cand_table: pl.DataFrame, left_on: list, right_on: list
):
    """This function measures the exact Jaccard containment of the query column
    `left_on` in `source_table` in column `right_on` in table `cand_table`.

    Containment is defined as JC(Q,X) = |Q intersection X| / |Q|

    Args:
        source_table (pl.DataFrame): Source table.
        cand_table (pl.DataFrame): Candidate table to query on.
        left_on (list): Query column.
        right_on (list): Candidate column

    Returns:
        float: Containment value.

    Raises:
        NotImplmentedError if more than one column is provided.
    """
    if len(left_on) > 1 or len(right_on) > 1:
        raise NotImplementedError("Only single query columns are supported")

    unique_source = source_table[left_on].unique()
    unique_cand = cand_table[right_on].unique()

    s1 = set(unique_source[left_on].to_series().to_list())
    s2 = set(unique_cand[right_on].to_series().to_list())
    return len(s1.intersection(s2)) / len(s1)


def build_containment_ranking(X: pd.DataFrame, candidate_joins: dict):
    """Given a table X and a list of candidates in any ranking, re-rank everything
    based on Jaccard containment.

    Args:
        X (pd.DataFrame): The base table to evaluate the containment on.
        candidate_joins (dict): All the candidate join objects to be ranked.

    Returns:
        pl.DataFrame: The candidate ranking.
    """
    containment_list = []
    for hash_, mdata in tqdm(
        candidate_joins.items(),
        total=len(candidate_joins),
        leave=False,
        desc="Building Containment Ranking: ",
        position=2,
    ):
        _, cnd_md, left_on, right_on = mdata.get_join_information()
        cnd_table = pl.read_parquet(cnd_md["full_path"])
        containment = measure_containment(X, cnd_table, left_on, right_on)
        containment_list.append(
            {
                "candidate": hash_,
                "table_hash": mdata.candidate_table,
                "containment": containment,
            }
        )
    ranking_df = pl.from_dicts(containment_list).sort(by="containment", descending=True)

    # If a table is present multiple times, drop the candidates with lower containment
    # ranking_df = ranking_df.unique("table_hash", keep="first", maintain_order=True)

    return ranking_df


def _split_X_y(X, y, test_size=0.2):
    # I want the indices for the
    s = ShuffleSplit(test_size=test_size)
    train_idx, valid_idx = next(s.split(X, y))

    return train_idx, valid_idx


def _build_preprocessor():

    num_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # Transformer for high cardinality categorical features
    high_card_transformer = Pipeline(
        steps=[
            (
                "onehot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            ("pca", PCA(n_components=30)),
        ],
    )

    low_card_transformer = Pipeline(
        steps=[
            (
                "onehot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ],
    )

    tv = TableVectorizer(
        low_cardinality=low_card_transformer,
        high_cardinality=high_card_transformer,
        numeric=num_transformer,
        n_jobs=1,
    )

    ct = make_column_transformer(
        (
            SimpleImputer(strategy="mean", keep_empty_features=True),
            make_column_selector(dtype_include=np.number),
        ),
        (
            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            make_column_selector(dtype_include=object),
        ),
    )

    preprocessor = Pipeline(
        steps=[
            ("table_vectorizer", tv),
            ("column_transformer", ct),
        ]
    )

    return preprocessor


def _reorder_indices_for_validation(X_train, X_valid, y_train, y_valid):
    # Concatenate the two splits for fitting
    X_concat = np.concatenate([X_train, X_valid])
    y_concat = np.concatenate([y_train.to_numpy(), y_valid.to_numpy()])

    # Prepare the indices for fitting
    n_train = len(X_train)
    n_val = len(X_valid)

    train_idx = np.arange(0, n_train)
    valid_idx = np.arange(n_train, n_train + n_val)
    return (X_concat, y_concat, train_idx, valid_idx)


class BaseJoinSelector(BaseEstimator):
    """BaseJoinSelector class. This class is extended by the other estimators.
    Note that, despite it being called `BaseJoinSelector`, this class is used as base for the NoJoin estimator as well.
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        target_column: str | None = None,
        model_parameters: dict | None = None,
        task: str = "regression",
    ) -> None:
        """Base Selector class. This class is supposed to be extended by the
        other selectors. It uses the `fit`/`predict` paradigm to implement the
        evaluation of a join suggestion method.

        Args:
            scenario_logger (ScenarioLogger): ScenarioLogger object to be used for tracking
            the parameters and the current experiment.
            target_column (str, optional): Column that contains the target to be used for
            supervised learning. Defaults to None.
            model_parameters (dict, optional): Parameters to be passed to the model. Defaults to None.
            task (str, optional): Task to be executed, either `regression` or `classification`. Defaults to "regression".

        Raises:
            ValueError: Raise ValueError if the value of `task` is not `regression` or `classification`.
            ValueError: Raise ValueError if the value of `chosen_model` is not `catboost` or `linear`.
        """
        super().__init__()

        if task == "regression":
            self.target_type = "continuous"
        elif task == "classification":
            self.target_type = "binary"
        else:
            raise ValueError(f"Task {task} not supported.")

        self.chosen_model = model_parameters["chosen_model"]

        if self.chosen_model not in SUPPORTED_MODELS:
            raise ValueError(f"Model {self.chosen_model} not supported.")

        self.name = "base_estimator"
        self.scenario_logger = scenario_logger
        self.model_parameters = model_parameters.get(
            model_parameters["chosen_model"], {}
        )

        self.target_column = target_column
        self.task = task
        self.n_joined_columns = None
        self.with_validation = True

        self.timestamps = {}
        self.durations = {}

        self.model = None
        self.cat_features = None

        self.preprocessor = _build_preprocessor()

        if self.chosen_model == "linear":
            self.with_validation = False

    def build_model(self, X: pd.DataFrame | None = None):
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
        elif self.chosen_model == "ridgecv":
            self.build_linear()
        elif self.chosen_model == "resnet":
            self.build_resnet()
        elif self.chosen_model == "realmlp":
            self.build_realmlp()
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
            "iterations": 300,
            "verbose": 0,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)
        parameters["random_seed"] = CATBOOST_RANDOM_SEED
        if self.task == "regression":
            self.model = CatBoostRegressor(cat_features=cat_features, **parameters)
        elif self.task == "classification":
            self.model = CatBoostClassifier(cat_features=cat_features, **parameters)
        else:
            raise ValueError

    def build_linear(self):
        """Build either a Ridge regressor, or a LogisticRegression classifier
        with default parameters.

        Raises:
            ValueError: Raise ValueError if the value in `self.task` is not a
            valid task (`regression` or `classification`).
        """
        if self.task == "regression":
            self.model = RidgeCV()
        elif self.task == "classification":
            self.model = LogisticRegression()
        else:
            raise ValueError

    def build_realmlp(self):
        """Build an estimator based on RealMLP."""
        defaults = {
            "n_threads": 32,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)

        if self.task == "regression":
            self.model = RealMLP_TD_Regressor(**parameters)
        elif self.task == "classification":
            self.model = RealMLP_TD_Classifier(**parameters)
        else:
            raise ValueError

    def build_resnet(self):
        """Build an estimator based on ResNet."""
        if self.task == "regression":
            self.model = Resnet_RTDL_D_Regressor(n_threads=32)
        elif self.task == "classification":
            self.model = Resnet_RTDL_D_Classifier(n_threads=32)
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

    def get_cat_features(self, X: pl.DataFrame | pd.DataFrame) -> list:
        """Given an input table X, return the list of features that are considered
        to be categorical by the appropriate type selection function.

        Args:
            X (pl.DataFrame | pd.DataFrame): The input table to be evaluated.

        Raises:
            TypeError: Raise TypeError if X is not pl.DataFrame or pd.DataFrame.

        Returns:
            list: The list of categorical features.
        """
        if type(X) == pl.DataFrame:
            cat_features = X.select(cs.string()).columns
        elif type(X) == pd.DataFrame:
            cat_features = X.select_dtypes("object").columns.to_list()
        else:
            raise TypeError(f"Type {type(X)} unsupported.")

        return cat_features

    def fit_model(
        self,
        X,
        y,
        validation_set: tuple | None = None,
    ):
        X_train, y_train = X, y
        X_valid, y_valid = validation_set
        if self.chosen_model == "catboost":
            X_train = (self.prepare_table(X_train)).to_pandas()
            X_valid = (self.prepare_table(X_valid)).to_pandas()
            y_train, y_valid = y_train.to_pandas(), y_valid.to_pandas()
            self.model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        elif self.chosen_model in ["resnet", "realmlp"]:
            X_train, _ = self.prepare_table_preprocessing(X_train)
            X_train_enc = self.preprocessor.fit_transform(X_train, y_train)
            X_valid, _ = self.prepare_table_preprocessing(X_valid)
            X_valid_enc = self.preprocessor.transform(X_valid)

            (
                X_concat,
                y_concat,
                _train_idx,
                _valid_idx,
            ) = _reorder_indices_for_validation(
                X_train_enc, X_valid_enc, y_train, y_valid
            )

            self.model.fit(
                X=X_concat,
                y=y_concat,
                val_idxs=_valid_idx,
            )
        else:
            # No validation set, keep all passed samples as train
            X_train, y_train = X, y
            # Linear
            X_train, _ = self.prepare_table_preprocessing(X_train)
            X_train_enc = self.preprocessor.fit_transform(X_train, y_train)
            self.model.fit(X=X_train_enc, y=y_train)

    def predict_model(self, X):
        if self.chosen_model == "catboost":
            X = (self.prepare_table(X)).to_pandas()
            y_pred = self.model.predict(X)
        else:
            X, _ = self.prepare_table_preprocessing(X)
            X_enc = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_enc)
        return y_pred

    def get_estimator_parameters(self):
        return {"estimator": self.name, "chosen_model": self.chosen_model}

    def get_additional_info(self):
        raise NotImplementedError

    def _prepare_inner_model(self):

        if self.task == "regression":
            inner_model = make_pipeline(
                TableVectorizer(
                    high_cardinality=MinHashEncoder(),
                    low_cardinality=OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    n_jobs=1,
                ),
                RandomForestRegressor(n_jobs=1),
                memory=CACHE_PATH,
            )
        else:
            inner_model = make_pipeline(
                TableVectorizer(
                    high_cardinality=MinHashEncoder(),
                    low_cardinality=OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    n_jobs=32,
                ),
                RandomForestClassifier(n_jobs=32),
                memory=CACHE_PATH,
            )
        return inner_model

    def _fit_predict_inner_model(self, X_train, X_valid, y, inner_model):
        # if isinstance(X_train, pl.DataFrame):
        #     X_train = X_train.to_pandas()

        # if isinstance(X_valid, pl.DataFrame):
        #     X_valid = X_valid.to_pandas()

        inner_model.fit(X_train, y)
        self._end_time("model_train")
        self._start_time("prepare")

        return inner_model.predict(X_valid)

    def _fit_inner_model(self, X, y, method="rf"):
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # prep = _build_preprocessor()
        # prep.fit(X, y)
        # X_enc = prep.transform(X)
        if self.task == "regression":
            if method == "rf":
                self.inner_model = tabular_learner(RandomForestRegressor())
        elif self.task == "classification":
            self.inner_model = RandomForestClassifier()
        else:
            raise ValueError
        self.inner_model.fit(X, y)
        # self.inner_model.fit(X_enc, y)
        return None
        # return prep

    def _predict_inner_model(self, X, preprocessor):
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # X_enc = preprocessor.transform(X)
        X_enc = X
        return self.inner_model.predict(X_enc)

    def _start_time(self, label: str):
        self.timestamps[label] = [dt.datetime.now(), None]
        if "time_" + label not in self.durations:
            self.durations["time_" + label] = 0

    def _end_time(self, label: str):
        self.timestamps[label][1] = dt.datetime.now()
        this_segment = self.timestamps[label]
        self.durations["time_" + label] += (
            this_segment[1] - this_segment[0]
        ).total_seconds()

    def clean_timers(self):
        self.timestamps = {}
        self.durations = {}

    def get_durations(self):
        return self.durations

    def update_durations(self, new_durations):
        for k, v in new_durations.items():
            self.durations[k] += v

    def prepare_table(self, table):
        if isinstance(table, pd.DataFrame):
            table = pl.from_pandas(table)

        table = table.fill_null(value="null").fill_nan(value=np.nan)
        return table.to_pandas()

    def prepare_table_preprocessing(self, table):
        """Prepare the given table applying a pre-processing step that fills
        missing values and encodes categorical variables.

        Args:
            table (pl.DataFrame, pd.DataFrame): Table to prepare

        Returns:
            pd.DataFrame: Prepared dataframe
            list: List of categorical columns
        """
        cat_features = table.select(cs.string()).columns
        # Storing the original column order
        _columns = table.columns

        # We should try new and more clever imputation methods
        table = table.select(
            cs.string().fill_null("null"),
            cs.numeric().fill_null("mean"),
            cs.datetime().dt.to_string("%Y-%m-%d").fill_null("null"),
        ).select(_columns)
        return table.to_pandas(), cat_features


class BaseJoinWithCandidatesMethod(BaseJoinSelector):
    """Base class that extends `BaseJoinMethod` to account for cases where multiple candidate joins are provided.

    Args:
        BaseJoinMethod (_type_): _description_
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins=None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
    ) -> None:
        super().__init__(scenario_logger, target_column, model_parameters, task)
        self.candidate_joins = candidate_joins
        self.n_candidates = len(candidate_joins)

        _join_params = {"aggregation": "first"}
        _join_params.update(join_parameters)
        self.join_parameters = _join_params

    def _execute_joins(
        self, X_train, X_valid=None, cnd_table=None, left_on=None, right_on=None
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        # Execute the join on the main table
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        # If a validation split is provided, join that separately
        # Join separately to avoid aggr. leakage
        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid

        return merged_train

    def _evaluate_candidate(self, y_true, y_pred):
        if self.task == "regression":
            return r2_score(y_true, y_pred)
        return f1_score(y_true, y_pred)


class NoJoin(BaseJoinSelector):
    """Reference selector that just trains a model on the base table and predicts on it, with no joins."""

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        target_column: str = None,
        model_parameters: dict = None,
        task: str = "regression",
    ) -> None:
        super().__init__(scenario_logger, target_column, model_parameters, task)
        self.name = "nojoin"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_set: tuple | None = None,
    ):
        if X.shape[0] != y.shape[0]:
            raise ValueError

        self._start_time("prepare")

        if validation_set is not None:
            X_valid, y_valid = validation_set
            X_train, y_train = X, y
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        self.build_model(X_train)

        self.n_joined_columns = len(X_train.columns)
        self._end_time("prepare")

        self._start_time("model_train")
        self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
        self._end_time("model_train")
        # else:
        #     self._start_time("prepare")
        #     self.build_model(X)
        #     self.n_joined_columns = len(X.columns)
        #     self._end_time("prepare")

        #     self._start_time("model_train")
        #     self.fit_model(X, y)
        #     self._end_time("model_train")

    def predict(self, X):
        self._start_time("model_predict")
        pred = self.predict_model(X)
        self._end_time("model_predict")
        return pred

    def get_additional_info(self):
        return {
            "best_candidate_hash": "nojoin",
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }


class HighestContainmentJoin(BaseJoinWithCandidatesMethod):
    def __init__(
        self,
        scenario_logger: ScenarioLogger | None = None,
        candidate_joins: dict | None = None,
        target_column: str | None = None,
        model_parameters: dict | None = None,
        join_parameters: dict | None = None,
        task: str = "regression",
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
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

    def fit(self, X, y):
        self._start_time("prepare")
        # Find the exact containment ranking
        self.candidate_ranking = build_containment_ranking(X, self.candidate_joins)
        # Select the top-1 candidate
        self.best_cnd_hash = self.candidate_ranking.top_k(1, by="containment")[
            "candidate"
        ].item()

        mdata = self.candidate_joins[self.best_cnd_hash]  # type: ignore
        _, best_cnd_md, self.left_on, self.right_on = mdata.get_join_information()

        self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])

        # Split in train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        self._end_time("prepare")
        self._start_time("join_train")
        # Execute join separately on each split (to avoid contamination with aggr)
        merged_train, merged_valid = self._execute_joins(
            X_train=X_train,
            X_valid=X_valid,
            cnd_table=self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
        )
        self.n_joined_columns = len(merged_train.columns)
        self._end_time("join_train")
        self._start_time("model_train")
        self.build_model(merged_train)

        # Fit the given model
        self.fit_model(merged_train, y_train, validation_set=(merged_valid, y_valid))
        self._end_time("model_train")
        # else:
        #     self._end_time("prepare")
        #     self._start_time("join_train")
        #     merged_train = self._execute_joins(
        #         X_train=X,
        #         cnd_table=self.best_cnd_table,
        #         left_on=self.left_on,
        #         right_on=self.right_on,
        #     )
        #     self.n_joined_columns = len(merged_train.columns)
        #     self._end_time("join_train")
        #     self._start_time("model_train")
        #     self.build_model(merged_train)
        #     self.fit_model(merged_train, y)
        #     self._end_time("model_train")

    def predict(self, X):
        self._start_time("join_predict")
        merged_test = ju.execute_join_with_aggregation(
            X,
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        self._end_time("join_predict")
        self._start_time("model_predict")
        pred = self.predict_model(merged_test)
        self._end_time("model_predict")
        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cnd_hash,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "n_joined_columns": self.n_joined_columns,
        }


class BestSingleJoin(BaseJoinWithCandidatesMethod):
    # pylint: disable=too-many-instance-attributes
    """
    The `BestSingleJoin` estimator takes at init time a set of candidates that will be trained at `fit` time to
    find the best single candidate among them.

    The selection of the "best candidate" is done at fit time.
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
        valid_size: float = 0.2,
        use_rf: bool = False,
    ) -> None:
        """
        Args:
            scenario_logger (ScenarioLogger, optional): ScenarioLogger object that contains the parameters relative to this run
            candidate_joins (dict, optional): Dictionary that includes all generated candidate joins
            target_column (str, optional): Target column.
            model_parameters (dict, optional): Additional parameters to be passed to the evaluation model (catboost or linear).
            join_parameters (dict, optional): Additional parameters to be considered when executing the join.
            task (str, optional): The task to be executed. Either "classification" or "regression". Defaults to "regression".
            valid_size (float, optional): The size of the validation set to be held out for evaluation of the best join. Defaults to 0.2.
        """
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )
        self.use_rf = use_rf
        self.name = "best_single_join"
        if use_rf:
            self.name += "_rf"
        self.candidate_ranking = None

        self.best_cnd_hash = None
        self.best_cjoin = None
        self.best_cnd_metric = -np.inf
        self.best_cnd_table = None
        self.left_on = self.right_on = None
        self.valid_size = valid_size

    def get_best_candidates(self, top_k=None):
        if top_k is None:
            return self.candidate_ranking
        return self.candidate_ranking.limit(top_k)

    def fit(self, X, y):
        self._start_time("prepare")
        # Prepare splits
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self._end_time("prepare")
        ranking = []

        for hash_, cjoin in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="BestSingleJoin",
            position=2,
        ):
            # Join on the candidate
            self._start_time("join_train")
            _, cnd_md, left_on, right_on = cjoin.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])

            merged_train, merged_valid = self._execute_joins(
                X_train, X_valid, cnd_table, left_on, right_on
            )

            self._end_time("join_train")

            self._start_time("model_train")
            # Build the model
            self.build_model(merged_train)

            method = "rf"
            if self.use_rf:
                inner_model = self._prepare_inner_model()
                y_pred = self._fit_predict_inner_model(
                    merged_train, merged_valid, y_train, inner_model
                )
            else:
                self.fit_model(
                    merged_train, y_train, validation_set=(merged_valid, y_valid)
                )
                self._end_time("model_train")
                self._start_time("prepare")
                y_pred = self.predict_model(merged_valid)
            metric = self._evaluate_candidate(y_valid, y_pred)

            ranking.append({"candidate": hash_, "metric": metric})

            if metric > self.best_cnd_metric:
                self.best_cnd_metric = metric
                self.best_cnd_hash = hash_
                self.best_cjoin = cjoin.candidate_id
            self._end_time("prepare")

        # RETRAINING THE MODEL
        self._start_time("prepare")
        best_join_mdata = self.candidate_joins[self.best_cnd_hash]
        (
            _,
            best_cnd_md,
            self.left_on,
            self.right_on,
        ) = best_join_mdata.get_join_information()

        self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])
        self._end_time("prepare")
        self._start_time("join_train")
        best_train, best_valid = self._execute_joins(
            X_train, X_valid, self.best_cnd_table, self.left_on, self.right_on
        )
        self.n_joined_columns = len(best_train.columns)
        self._end_time("join_train")

        self._start_time("model_train")
        self.build_model(best_train)

        self.fit_model(best_train, y_train, validation_set=(best_valid, y_valid))
        self.candidate_ranking = pl.from_dicts(ranking).sort("metric", descending=True)
        self._end_time("model_train")

    def predict(self, X):
        self._start_time("join_predict")
        merged_test = ju.execute_join_with_aggregation(
            X,
            self.best_cnd_table,
            left_on=self.left_on,
            right_on=self.right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix="_right",
        )
        self._end_time("join_predict")

        self._start_time("model_predict")
        pred = self.predict_model(merged_test)
        self._end_time("model_predict")

        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cjoin,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "n_joined_columns": self.n_joined_columns,
        }

    def _execute_joins(
        self, X_train, X_valid=None, cnd_table=None, left_on=None, right_on=None
    ):
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid
        else:
            return merged_train


class FullJoin(BaseJoinWithCandidatesMethod):
    """The `FullJoin` estimator executes the full left join (with aggregation) between the
    input table and all provided candidates, then trains a model on the resulting table.
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
        top_k: None | int = None,
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )
        if top_k is None:
            self.name = "full_join"
        else:
            if top_k < 1:
                raise ValueError(f"k must be >= 1, found {top_k}.")
            if not isinstance(top_k, int):
                raise TypeError(f"top_k must be integer, found {type(top_k)}")
            self.top_k = top_k
            self.name = f"top_{top_k}_full_join"
            self.candidate_joins = {
                k: v
                for k, v in self.candidate_joins.items()
                if k in islice(self.candidate_joins, top_k)
            }

        if self.join_parameters["aggregation"] == "dfs":
            print("Full join not available with DFS.")
            return None

    def fit(self, X=None, y=None):
        self._start_time("prepare")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        merged_train = X_train.clone().lazy()
        self._end_time("prepare")
        self._start_time("join_train")
        merged_train = ju.execute_join_all_candidates(
            merged_train, self.candidate_joins, self.join_parameters["aggregation"]
        )
        merged_valid = X_valid.clone().lazy()
        merged_valid = ju.execute_join_all_candidates(
            merged_valid, self.candidate_joins, self.join_parameters["aggregation"]
        )
        self._end_time("join_train")
        self._start_time("model_train")
        self.build_model(merged_train)
        self.fit_model(merged_train, y_train, validation_set=(merged_valid, y_valid))
        self._end_time("model_train")

        # else:
        #     self._start_time("prepare")
        #     merged_train = X.clone().lazy()
        #     self._end_time("prepare")
        #     self._start_time("join_train")
        #     merged_train = ju.execute_join_all_candidates(
        #         merged_train, self.candidate_joins, self.join_parameters["aggregation"]
        #     )
        #     self._end_time("join_train")
        #     self._start_time("model_train")
        #     self.build_model(merged_train)
        #     self.fit_model(merged_train, y)
        #     self._end_time("model_train")
        self.n_joined_columns = len(merged_train.columns)

    def predict(self, X: pd.DataFrame):
        self._start_time("join_predict")
        merged_test = X.clone().lazy()
        merged_test = ju.execute_join_all_candidates(
            merged_test, self.candidate_joins, self.join_parameters["aggregation"]
        )
        self._end_time("join_predict")

        self._start_time("model_predict")
        pred = self.predict_model(merged_test)
        self._end_time("model_predict")

        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        # TODO: add the list of all left_on, right_on
        return {
            "best_candidate_hash": "full_join",
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }


class StepwiseGreedyJoin(BaseJoinWithCandidatesMethod):
    """StepwiseGreedyJoin produces a full join path employing a greedy strategy. It evaluates"""

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        budget_type: str = None,
        budget_amount: str = None,
        epsilon: float = None,
        ranking_metric: str = "containment",
        valid_size: float = 0.2,
        max_candidates: int = 50,
        task: str = "regression",
        use_rf: bool = False,
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )

        self.name = "stepwise_greedy_join"
        self.use_rf = use_rf
        if use_rf:
            self.name += "_rf"
        self.budget_type, self.budget_amount = self._check_budget(
            budget_type, budget_amount
        )
        self.ranking_metric = self._check_ranking_method(ranking_metric)
        self.valid_size = valid_size

        # Calling this "current_metric" to be more generic
        self.current_metric = -np.inf
        self.current_X_train = self.current_X_valid = None
        self.selected_candidates = {}
        self.valid_size = valid_size
        self.blacklist = deque([], maxlen=max_candidates)
        self.candidate_ranking = None
        self.max_candidates = max_candidates
        self.with_validation = True
        self.already_joined_tables = []
        # TODO: account for multiple candidate joins on the same table
        self.already_evaluated = {cjoin: 0 for cjoin in self.candidate_joins.keys()}
        self.base_epsilon = epsilon
        self.epsilon = epsilon

        self.inner_model = None

        self.wrap_up_joiner = None
        self.wrap_up_joiner_params = model_parameters

    def fit(self, X: pd.DataFrame, y):
        self._start_time("prepare")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        self.candidate_ranking = self._build_ranking(X)
        ########## BASE TABLE
        self._start_time("model_train")
        self.build_model(X_train)

        # Fit the given model
        self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
        self._end_time("model_train")
        self._start_time("model_predict")
        y_pred = self.predict_model(X_valid)
        self._end_time("model_predict")
        metric = self._evaluate_candidate(y_valid, y_pred)
        self._end_time("prepare")
        self.current_metric = metric

        self.current_X_train = X_train
        self.current_X_valid = X_valid

        for _ in tqdm(
            range(self.budget_amount),
            total=self.budget_amount,
            desc="StepwiseGreedyJoin - Iterating: ",
            leave=False,
            position=2,
        ):
            self._start_time("prepare")
            cjoin = self._get_candidate()
            if cjoin is None:
                # No more candidates:
                break
            cjoin_id = cjoin.candidate_id
            _, cnd_md, left_on, right_on = cjoin.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])
            cnd_hash = cnd_md["hash"]
            self._end_time("prepare")
            if cjoin_id not in self.already_joined_tables:
                self._start_time("join_train")
                temp_X_train, temp_X_valid = self._execute_joins(
                    self.current_X_train,
                    self.current_X_valid,
                    cnd_table,
                    left_on,
                    right_on,
                    suffix=cjoin.candidate_id[:10],
                )

                self._end_time("join_train")

                self._start_time("model_train")
                self.build_model(temp_X_train)

                # Fit the given model
                if self.use_rf:
                    y_pred = self._prepare_inner_model(
                        merged_train, y_train, merged_valid, method
                    )
                else:
                    self.fit_model(
                        temp_X_train,
                        y_train,
                        validation_set=(temp_X_valid, y_valid),
                    )
                    self._end_time("model_train")
                    self._start_time("prepare")
                    y_pred = self.predict_model(temp_X_valid)

                metric = self._evaluate_candidate(y_valid, y_pred)
                self._end_time("prepare")
            else:
                self._start_time("prepare")
                metric = -np.inf
                temp_X_train = temp_X_valid = None
                self._end_time("prepare")
            self._start_time("prepare")
            self._update_ranking(cjoin, metric, temp_X_train, temp_X_valid)
            self._end_time("prepare")

        if len(self.selected_candidates) > 0:
            # Full join all the selected candidates
            self._start_time("prepare")
            self.wrap_up_joiner = FullJoin(
                scenario_logger=self.scenario_logger,
                candidate_joins=self.selected_candidates,
                target_column=self.target_column,
                model_parameters=self.wrap_up_joiner_params,
                join_parameters=self.join_parameters,
                task=self.task,
            )
            self._end_time("prepare")
            self.wrap_up_joiner.fit(X=X, y=y)

        else:
            # No good candidates found, train on the base table
            self._start_time("prepare")
            self.wrap_up_joiner = NoJoin(
                scenario_logger=self.scenario_logger,
                target_column=self.target_column,
                model_parameters=self.wrap_up_joiner_params,
                task=self.task,
            )
            self._end_time("prepare")
            self.wrap_up_joiner.fit(X=X, y=y)

        self.n_joined_columns = self.wrap_up_joiner.n_joined_columns
        self.update_durations(self.wrap_up_joiner.get_durations())

    def _build_ranking(self, X):
        if self.ranking_metric == "containment":
            _ranking = build_containment_ranking(X, self.candidate_joins)
            candidate_ranking = deque(
                _ranking["candidate"].to_list(), maxlen=self.max_candidates
            )

        return candidate_ranking

    def _get_candidate(self) -> CandidateJoin | None:
        if len(self.candidate_ranking) > 0:
            _cnd = self.candidate_ranking.popleft()
            return self.candidate_joins[_cnd]
        elif 0 < len(self.blacklist) < self.n_candidates:
            # If the length of the blacklist is == n_candidates, then all candidates have already been discarded, stopping
            # All candidates have been evaluated, selecting one of the discarded candidates
            random.shuffle(self.blacklist)
            _cnd = self.blacklist.popleft()
            return self.candidate_joins[_cnd]
        else:
            # No more candidates are left, stop iterations
            return None

    def _get_curr_eps(self, n):
        # TODO: expand as needed
        return 1 / (np.log(n) + 1) * self.base_epsilon

    def _update_ranking(self, cjoin, metric, temp_X_train, temp_X_valid):
        cjoin_hash = cjoin.candidate_id
        self.already_evaluated[cjoin_hash] += 1
        if metric > self.current_metric + self.epsilon:
            self.selected_candidates.update({cjoin_hash: cjoin})
            self.current_metric = metric
            self.current_X_train = temp_X_train
            self.current_X_valid = temp_X_valid
            n_cnd = len(self.selected_candidates)
            self.epsilon = self._get_curr_eps(n_cnd)
            self.already_joined_tables.append(cjoin_hash)
        else:
            # self.candidate_ranking.append(cnd_hash)
            if self.already_evaluated[cjoin_hash] < 2:
                self.blacklist.append(cjoin_hash)

    def predict(self, X):
        pred = self.wrap_up_joiner.predict(X)
        self.durations["time_join_predict"] = self.wrap_up_joiner.durations.get(
            "time_join_predict", 0
        )
        self.durations["time_model_predict"] = self.wrap_up_joiner.durations.get(
            "time_model_predict", 0
        )
        return pred

    def _check_ranking_method(self, ranking_method):
        if ranking_method not in SUPPORTED_RANKING_METHODS:
            raise ValueError(
                f"`ranking_method` {ranking_method} not in SUPPORTED_RANKING_METHODS."
            )
        return ranking_method

    def _check_budget(self, budget_type, budget_amount):
        if budget_type == "iterations":
            try:
                budget_amount = int(budget_amount)
            except:
                raise ValueError("A non-numeric number of iterations was provided.")
            if not isinstance(budget_amount, int):
                raise ValueError("`budget_amount` must be integer")
            if budget_amount <= 0:
                raise ValueError("The number of iterations must be strictly positive.")
            return budget_type, budget_amount

        raise ValueError(
            f"Provided budget_type {budget_type} not in SUPPORTED_BUDGET_TYPES."
        )

    def _execute_joins(
        self,
        X_train,
        X_valid=None,
        cnd_table=None,
        left_on=None,
        right_on=None,
        suffix="_right",
    ):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix=suffix,
        )

        if X_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix=suffix,
            )
            return merged_train, merged_valid
        else:
            return merged_train

    def get_additional_info(self):
        # TODO: add the list of all left_on, right_on
        return {
            "n_joined_columns": self.n_joined_columns,
            "budget_type": self.budget_type,
            "budget_amount": self.budget_amount,
            "epsilon": self.base_epsilon,
            "chosen_candidates": [
                cjoin.candidate_id for cjoin in self.selected_candidates.values()
            ],
        }
