import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import clone
from sklearn.metrics import confusion_matrix, mean_squared_log_error
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._param_validation import HasMethods

import cv_tools
from samples import sample_wt_opt


def get_confusion_weights():
    return np.array([[1.0, -0.5, -1.0], [0.1, 1.0, 0.0], [0.0, 0.25, 1.0]])


def three_class_solubility(y_true, y_pred, sample_weight=None, **kwargs):
    # For balanced accuracy, with W = I: np.diag(C) = np.sum(C * W)
    # In MCC, W = 2 * I - 1 (ie. off diagonals are -1 instead of 0)
    W = get_confusion_weights()
    try:
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    except UserWarning:
        print("True, Predicted, and Confusion Weighting")
        print(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(C), np.int16))
            )
        )
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # with np.errstate(divide="ignore", invalid="ignore"):
    with np.errstate(divide="print", invalid="print"):
        per_class = np.sum(C * W, axis=1) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        raise UserWarning.add_note(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(per_class), np.int16))
            )
        )
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def model_prediction_distance(
    predicts_list,
    metric=mean_squared_log_error,
    symmetric=False,
    sample_weight=None,
    include_thresholds=None,
    clip_thresholds=None,
):
    """

    Parameters
    ----------
    predicts_list
    metric : callable
    sample_weight : pd.Series
    symmetric : bool
    Is the metric symmetric if the inputs are switched?
    include_thresholds : tuple[float, float]
    Removes samples outside this range when calculating distances.
    clip_thresholds : tuple[float, float]
    Clips all differences at these values but includes in distance calculations.
    Warning! Not implemented.

    Returns
    -------
    distances_arr : ndarray
    Metric measures between sets of predicts. "Distance" dependent on measure chosen.
    Size (n_sets, n_sets)
    """
    n_predicts = len(predicts_list)
    distances_arr = np.zeros(shape=(n_predicts, n_predicts), dtype=np.float32)
    for i, j in itertools.combinations(np.arange(n_predicts), r=2):
        included_idx = predicts_list[i].index
        if include_thresholds is not None:
            if include_thresholds[0] is not None:
                lower_idx = predicts_list[i][
                    (predicts_list[j] - predicts_list[i]) >= include_thresholds[0]
                ].index
                included_idx = predicts_list[i].index.intersection(lower_idx)
            if include_thresholds[1] is not None:
                upper_idx = predicts_list[i][
                    (predicts_list[j] - predicts_list[i]) <= include_thresholds[0]
                ].index
                included_idx = predicts_list[i].index.intersection[upper_idx]
        y_1 = predicts_list[i][included_idx]
        y_2 = predicts_list[j][included_idx]
        distances_arr[i, j] = sample_wt_opt(
            metric,
            sample_weight[included_idx],
            y_1,
            y_2,
        )
        if not symmetric:
            distances_arr[j, i] = sample_wt_opt(
                metric,
                sample_weight,
                y_1,
                y_2,
            )
        else:
            distances_arr[j, i] = distances_arr[i, j]
    return distances_arr


def cv_model_prediction_distance(
    feature_df,
    labels,
    name_model_subset_tup,
    dist_metric=cosine_distances,
    cv=StratifiedKFold,
    response="predict_proba",
    **kwargs
):
    # Make regressor/response-compatible use sklearn estimator checks. Correct default CV splitter.
    test_idx_list = list()
    distance_dict = {"test": list(), "train": list()}
    predict_dict = dict()
    for name, model, subset in name_model_subset_tup:
        predict_dict[name] = {"test": list(), "train": list()}
    for train_X, train_y, test_X, test_y in cv_tools.split_df(
        feature_df, labels, splitter=cv, **kwargs
    ):
        split_X, split_y = {"train": train_X, "test": test_X}, {
            "train": train_y,
            "test": test_y,
        }
        split_X["train"] = train_X
        split_X["test"] = test_X
        test_idx_list.append(test_y.index)
        assert not split_X["train"].empty
        estimator_list = list()
        for name, estimator, subset in name_model_subset_tup:
            if (
                HasMethods("sample_weight").is_satisfied_by(estimator)
                and "sample_weight" in kwargs.keys()
            ):
                print("Using sample weights to score subsets...")
                fit_est = clone(estimator).fit(
                    split_X["train"][subset],
                    split_y["train"],
                    sample_weight=kwargs["sample_weight"].loc[train_y.index],
                )
            else:
                fit_est = clone(estimator).fit(
                    split_X["train"][subset], split_y["train"]
                )
            estimator_list.append((fit_est, subset))
            for split_set in predict_dict[name].keys():
                result_df = pd.DataFrame(
                    getattr(fit_est, response)(X=split_X[split_set][subset]),
                    index=split_X[split_set].index,
                ).squeeze()
                if isinstance(result_df, pd.Series):
                    result_df.name = split_set
                predict_dict[name][split_set].append(result_df)
    split_distplot_dict = dict()
    for split_set in distance_dict.keys():
        dist_list = list()
        for r in zip(
            [predict_dict[name][split_set] for name, _, _ in name_model_subset_tup]
        ):
            zip_list = [pd.concat(df) for df in r]
            print(zip_list)
            # dist_list.append(r)
            # dist_list.append(model_prediction_distance(r, dist_metric))
            dist_list.append(np.vstack(zip_list))
        distance_dict[split_set] = pd.DataFrame(np.hstack(dist_list)).merge(
            labels, left_index=True, right_index=True
        )
        split_distplot_dict[split_set] = sns.pairplot(data=distance_dict[split_set])
    return split_distplot_dict


def weigh_single_proba(onehot_true, probs, prob_thresholds=None):
    if prob_thresholds is not None:
        probs.clip(lower=prob_thresholds[0], upper=prob_thresholds[1], inplace=True)
    unweighted = (
        probs.add(onehot_true).abs().multiply(onehot_true).sum(axis=1).squeeze()
    )
    return unweighted


def compare_models_to_predictions(
    feature_df, model_subsets, metric, predictions=None, response="predict"
):
    new_predicts = list()
    for m, s in model_subsets:
        if response == "predict":
            new_predicts.append(m.predict(feature_df[s]))
        else:
            new_predicts.append(m.predict_proba(feature_df[s]))
    distances = pd.concat(
        [np.corrwith(other=predictions, axis=1, method=metric) for np in new_predicts],
        keys=[model_subsets.keys()],
    )
    return distances
