import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.neural_network import MLPRegressor


def _score_weights(weights, y_true, probs, activation=np.square):
    return activation(np.multiply(probs, weights) - y_true)


def _solve_weights(y_true, expert_proba):
    """

    Parameters
    ----------
    y_true : Class Label or Regression Target
    expert_proba : Output from experts. Positive class probability in the case of classifiers.

    Returns
    -------
    result.x = Minimized weights for scoring.
    """
    result = minimize_scalar(
        fun=_score_weights, args=(y_true, expert_proba), bounds=[0, 1]
    )
    return result.x


class FuzzyApplicator(BaseEstimator, MetaEstimatorMixin):

    def __init__(
        self,
        n_experts,
        layer_sizes,
        activation,
        random_state=0,
        early_stopping=True,
        verbose=1,
    ):
        # self.gating_function = MLPClassifier(hidden_layer_sizes=layer_sizes, activation=activation, random_state=random_state, early_stopping=early_stopping, verbose=verbose)
        self.n_experts = n_experts
        self.gating_function = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            activation=activation,
            warm_start=True,
            random_state=random_state,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        # self.annealers = annealers
        # self.predictor_type = None
        # self.predictors = None
        # self.label_type = None
        # self.labels = None

    def fit(
        self,
        X,
        y,
        pos_label=1,
        sample_weight=None,
    ):
        y_true = y[X.index]
        expert_predicts = X.iloc[:, : self.n_experts].loc[y.index]
        fp_predictors = X.iloc[:, self.n_experts :].loc[y.index]
        sample_wts = sample_weight.loc[y.index]
        opt_weights = np.array(
            [_solve_weights(y_t, p) for y_t, p in zip(expert_predicts, y)]
        )
        self._train_gate(
            fp_predictors=fp_predictors.to_numpy(), opt_weights=opt_weights
        )
        return self.gating_function

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.sum(
            np.multiply(
                X[:, : self.n_experts],
                self.gating_function.predict(X[:, self.n_experts :]),
            )
        )

    def predict(self, X):
        pred_prob = self.predict_proba(X)
        return np.where(lambda x: x < 0.5, [0, 1])

    def _train_gate(self, fp_predictors, opt_weights):
        self.gating_function.fit(X=fp_predictors, y=opt_weights)
