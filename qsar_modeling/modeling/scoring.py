import copy
import numbers
from collections import defaultdict
from functools import partial
from inspect import signature

import numpy as np
import pandas as pd
import sklearn.pipeline
from scipy.stats import gmean
from sklearn import clone, metrics
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    hamming_loss,
    jaccard_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, LearningCurveDisplay, StratifiedKFold
from sklearn.utils import compute_sample_weight
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import _check_y

import samples
from qsar_modeling.utils import cv_tools

# TODO: Implement sample weight. Will probably need to be a separate function for cv_model_generalized that selects after the split.

learning_curve = ("Learning Curve", partial(LearningCurveDisplay.from_estimator))


def get_displays():
    displays = [
        ("Confusion Matrix", partial(metrics.ConfusionMatrixDisplay.from_estimator)),
        ("ROC", partial(metrics.RocCurveDisplay.from_estimator)),
        ("DET", partial(metrics.DetCurveDisplay.from_estimator)),
        ("Precision-Recall", partial(metrics.PrecisionRecallDisplay.from_estimator)),
        ("Prediction Error", metrics.PredictionErrorDisplay.from_estimator),
    ]
    return displays


def get_pred_score_funcs():
    # Returns tuples of scoring metric names and their callable functions. Option for one that only return scalar values.
    pred_scores = [
        ("Balanced Accuracy", balanced_accuracy_score),
        ("Matthews Correlation Coefficient", matthews_corrcoef),
        # ('Hinge Loss', hinge_loss),
        ("Recall Micro", partial(recall_score, average="micro")),
        ("Hamming Loss", hamming_loss),
        ("Negative Log Loss", partial(log_loss)),
        ("Negative Log Loss Normalized", partial(log_loss, normalize=True)),
        ("Jaccard score", partial(jaccard_score, pos_label=0)),
    ]
    # ('Classification Report', partial(classification_report, target_names=['Insoluble', 'Soluble'])),
    # ('Classification Report - Weighted', partial(classification_report, target_names=['Insoluble', 'Soluble']))
    score_list = [
        (name, make_scorer(f, response_method="predict")) for name, f in pred_scores
    ]
    return score_list


def get_score_bounds():
    # Returns bounds to check reasonableness of calculated scores for model performance.
    bounds = dict(
        [
            ("Balanced Accuracy", (0.5, 1.0)),
            ("Matthews Correlation Coefficient", (-1.0, 1.0)),
            # ('Hinge Loss', hinge_loss),
            ("Recall Micro", (0.0, 1.0)),
            ("Hamming Loss", (0.0, 1.0)),
            ("Negative Log Loss", (0, 999999)),
            ("Negative Log Loss Normalized", (0, 999999)),
            ("Jaccard score", (0.0, 1.0)),
        ]
    )
    return bounds


def get_prob_sc_dict(pos_label=0):
    # Returns classification scores that require probability or non-prediction based outputs.
    prob_scores = dict(
        [
            (
                "Average Precision Macro",
                partial(average_precision_score, pos_label=pos_label, average="macro"),
            ),
            (
                "Average Precision Micro",
                partial(average_precision_score, average="micro"),
            ),
            (
                "Average Precision Weighted",
                partial(
                    average_precision_score, pos_label=pos_label, average="weighted"
                ),
            ),
            ("Negative Brier Score", partial(brier_score_loss, pos_label=pos_label)),
            ("Precision Micro", partial(precision_score, average="micro")),
            # ('ROC Score', partial(roc_curve, pos_label=pos_label))
        ]
    )
    score_list = [
        (name, make_scorer(f, response_method="predict_proba"))
        for name, f in prob_scores
    ]
    return score_list


def score_cv_results(
    results_dict, score_func_dict, y_true=None, by_fold=True, **score_kws
):
    """
    Applies scoring function to nested dictionary of estimator outputs.
    results_dict:

    Parameters
    ----------
    results_dict: dict[str[str[list]]], ex. {"predict": {"test": [pd.Series]}}
    score_func_dict: dict[str, callable], Accepts estimator output from results_dict
    y_true: pd.Series | Iterable[pd.Series], arg for score_func, if by_fold == False, must be pd.Series
    by_fold: bool, whether to apply score_func to individual folds or score after concatenating folds
    score_kws: Applied to each score_func call.

    Returns
    -------
    score_dict: dict[dict[list[pd.Series]]], Same structure as results_dict, but contains output from score_funcs
    {score_name: {split_name: list[pd.Series] | pd.Series}}
    """
    score_dict = dict.fromkeys(score_func_dict.keys(), dict())
    score_df_dict = dict.fromkeys(score_func_dict.keys())
    for score_name, score_func in score_func_dict.items():
        output_name = get_input_for_scorer(score_func)
        for split_name, result in results_dict[output_name].items():
            score = list()
            if by_fold:
                for res in result:
                    if (
                        len(res.shape) > 1
                        and res.shape[1] > 1
                        and "pos_label" in score_kws
                    ):
                        r = res.iloc[:, score_kws["pos_label"]]
                    elif len(res.shape) > 1 and res.shape[1] > 1:
                        r = res.iloc[:, 1]
                    else:
                        r = res
                    new_kws = dict()
                    for k, v in score_kws.items():
                        if k in score_func_dict.keys():
                            if isinstance(v, (pd.Series, pd.DataFrame)):
                                new_kws[k] = v[r.index]
                            else:
                                new_kws[k] = v
                    # print(score_func(y_true[r.index], r))
                    if y_true is not None:
                        if output_name == "predict":
                            # print(r.value_counts())
                            score.append(
                                score_func(y_true=y_true[r.index], y_pred=r, **new_kws)
                            )
                        elif output_name == "predict_proba":
                            score.append(
                                score_func(y_true=y_true[r.index], y_proba=r, **new_kws)
                            )
                        else:
                            score.append(
                                score_func(y_true=y_true[r.index], y_proba=r, **new_kws)
                            )
                    else:
                        score.append(score_func(r, **new_kws))
            else:
                if y_true is not None:
                    score = score_func(y_true, pd.concat(result), **score_kws)
                else:
                    score = score_func(pd.concat(result), **score_kws)
            score_dict[score_name][split_name] = score
        # print(score_dict[score_name])
        score_df_dict[score_name] = (
            pd.DataFrame.from_records(score_dict[score_name])
            .reset_index(names="CV Fold")
            .melt(id_vars=["CV Fold"], var_name="Split", value_name="Score")
        )
        # print(score_df_dict[score_name])
        score_df_dict[score_name].insert(loc=0, column="Metric", value=score_name)
    score_df = pd.concat(score_df_dict.values(), ignore_index=True)
    return score_df


def get_input_for_scorer(score_func):
    score_func = signature(score_func).parameters.keys()
    if "y_proba" in score_func:
        output_name = "predict_proba"
    # elif "y_pred" in score_keys:
    elif "y_pred" in score_func:
        output_name = "predict"
    else:
        output_name = "predict"
    return output_name


def proba_from_meta_estimator(
    base_estimator,
    meta_est,
    feature_df,
    labels,
    subsets,
    cv=StratifiedKFold,
    sample_weights=None,
):
    """
    for model, weights in itertools.zip_longest(base_estimator, sample_weights):
        meta_fit = StackingClassifier(estimators=estimator_tups, cv=cv, n_jobs=-1).fit(
            X=feature_df, y=feature_df, sample_weight=weights
        )
    """
    return None


def calculate_proba_from_model(
    estimator,
    feature_df,
    labels,
    select_params,
    cv=StratifiedKFold,
    preprocessor=None,
    sample_weight=None,
):
    """
    Convenience function to calculate model predict_proba for one or more sample_weightings.

    Parameters
    ----------
    select_params
    estimator: BaseEstimator
    feature_df: pd.DataFrame
    labels: pd.Series
    cv: default=StratifiedKFold
    preprocessor: BaseEstimator, fold-wise preprocessing before fitting
    sample_weight: None | pd.Series | Iterable[pd.Series], weights for model fitting

    Returns
    -------
    results_list: list[pd.Series], shape[n_samples], List of sample weights for each input sample weighting. Intended for use in feature selection.
    """
    results_list = list()
    if cv is None and is_classifier(estimator):
        cv = partial(StratifiedKFold, shuffle=True, random_state=0)
    elif cv is None and is_regressor(estimator):
        cv = partial(KFold, shuffle=True, random_state=0)
    if sample_weight is None:
        sample_weight = [
            pd.Series(
                data=np.ones(shape=feature_df.shape[0], dtype=np.float32),
                index=feature_df.index,
            )
        ]
    elif len(sample_weight) == 1 or isinstance(
        sample_weight, (pd.Series, pd.DataFrame)
    ):
        sample_weight = [sample_weight]
    for weights in sample_weight:
        results, idx_fold_list = cv_model_generalized(
            estimator,
            feature_df,
            labels,
            cv=cv,
            preprocessor=preprocessor,
            sample_weight=weights,
        )
        results_list.append(results)
    return [pd.concat(r[select_params["model_output"]]["test"]) for r in results_list]


def scramble_classes(labels, random_state=None):
    scrambled_labels = pd.Series(
        data=labels.copy().sample(frac=1.0, random_state=random_state).to_list(),
        index=labels.index.copy(name="INCHI_KEY"),
    )
    randomized_labels = pd.Series(
        _check_y(scrambled_labels), index=scrambled_labels.index
    )
    return randomized_labels


# noinspection PyUnresolvedReferences
def cv_model_generalized(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    preprocessor=None,
    methods=None,
    return_train=False,
    clone_model=True,
    sample_weight=None,
    random_state=None,
    pos_label=0,
    randomize_classes="both",
    **kwargs,
):
    """
    Generalized, all-in-one wrapper for cross-validated models. Desired outputs are given as nested tuples of name, callable.
    Results are returned as a dictionary of dictionaries of lists.
    Schema: results = {randomized: {load_cv_results: {split_set: [pd.Series] } } }
    Example: results = {"Randomized": {"predict": {"test": [pd.Series, pd.Series, ...] } } }

    Parameters
    ----------

    estimator: BaseEstimator, Unfit estimator or pipeline
    feature_df : pd.DataFrame, features
    labels : pd.Series, labels for supervised training
    cv : "Cross-Validator", default=StratifiedKFold
    preprocessor : TransformerMixin or implements "fit" and "transform"
    return_train : bool, whether to also calculate and return results for training data, adds "train" to results keys
    methods : [str], values must be methods from estimator ("predict", "predict_proba", "decision_function")
    clone_model : bool, Clone model or use fitted model
    sample_weight : pd.Series
    random_state : int | None, Seed for scrambling classes, ignored if randomized_classes is False
    pos_label : int, Label for positive class, only used if "pos_label" is in a scoring function's signature.
    randomize_classes : {"both" | True | False}, whether to return true and randomized label results ("both"), only randomized (True), or only true (False)
    results["Randomized" | "original"]

    Returns
    -------
    results_dict : dict[str, dict[str, dict[str, list[pd.Series | pd.DataFrame]]]], nested dictionary of {randomization: {callable name: {"test" | "train: list[pd.Series]}}}, where the list contain results from callable over all CV folds.
    melted_dict : dict[str, pd.DataFrame], Dictionary of long-form DataFrames. Same content as results. keys=methods.
    Columns: ["CV_Fold", "Split", "Label", "INCHI_KEY", RESULTS, "True"]
    test_idx_list: tuple[pd.Index], tuple of indices in the order in which they appear during cross-validation

    Example Usage: Scoring
    ----------------------
        group_cols = ["CV_Fold", "Split", "Labels"]
        predict_df = melted_dict["predict"]
        predict_groups = predict_df.groupby(group_cols, as_index=False, group_keys=False)
        [print(g[1]) for g in predict_groups]
        print(
            [
                score_long_form(
                    balanced_accuracy_score,
                    g[1].drop(columns=["CV_Fold", "Split", "Labels"]),
                    true_col="True",
                    remove_cols="INCHI_KEY",
                )
                for g in predict_groups
            ]
        )
    """
    # score_dict: dict[dict[list(pd.Series)], Contains output from score_funcs
    # {score_name: {split_name: list[pd.Series]}}
    # TODO: Add functionality for fitted estimators.
    from sklearn.utils.estimator_checks import is_regressor, is_classifier

    sklearn.set_config(transform_output="pandas")

    sc_dict, results_dict, long_forms = dict(), dict(), dict()
    test_idx_list = list()
    if methods is None:
        if is_classifier(estimator):
            methods = ("predict", "predict_proba")
        # elif is_regressor(estimator):
        else:
            methods = tuple(
                "predict",
            )
    elif isinstance(methods, str):
        methods = [methods]
    elif not all([isinstance(m, str) for m in methods]):
        print("Methods for model evaluation returned: \n{}".format(methods))
        raise UserWarning
    long_lists = dict([(k, list()) for k in methods])
    methods_list = methods
    # methods_list = [s for s in methods if HasMethods(methods=s).is_satisfied_by(estimator)]
    if randomize_classes == "both":
        label_orders, order_names = [True, False], ["randomized", "original"]
    elif randomize_classes:
        label_orders, order_names = [False], ["randomized"]
    else:
        label_orders, order_names = [True], ["original"]
    if return_train:
        results = dict([(k, {"train": list(), "test": list()}) for k in methods_list])
    else:
        results = dict([(k, {"test": list()}) for k in methods_list])
    func_kwargs_dict = dict()
    for func in methods_list:
        func_kwargs_dict[func] = [
            k
            for k in signature(getattr(estimator, func)).parameters.keys()
            if k != "self" and k != "X"
        ]
    # results = dict([(k[0], splits_dictcopy()) for k in return_list])
    assert isinstance(feature_df, pd.DataFrame) and not feature_df.empty
    for o_name in order_names:
        sc_dict[o_name] = list()
        if return_train:
            results_dict[o_name] = dict(
                [(k, {"train": list(), "test": list()}) for k in methods_list]
            )
        else:
            results_dict[o_name] = dict([(k, {"test": list()}) for k in methods_list])
    split_X, split_y = dict(), dict()
    i = 0
    for train_X, train_y_orig, test_X, test_y_orig in cv_tools.split_df(
        feature_df, labels, splitter=cv, **kwargs
    ):
        if preprocessor is not None:
            preprocessor.fit(
                train_X,
            )
            split_X["train"] = preprocessor.transform(train_X).to_frame()
            split_X["test"] = preprocessor.transform(test_X).to_frame()
        else:
            split_X["train"] = train_X
            split_X["test"] = test_X
        assert isinstance(split_X["train"], pd.DataFrame)
        assert not split_X["train"].empty
        test_idx_list.append(test_y_orig.index)
        for ran, r_name in zip(label_orders, order_names):
            if ran:
                train_y = scramble_classes(
                    copy.deepcopy(train_y_orig), random_state=random_state
                )
                test_y = scramble_classes(
                    copy.deepcopy(test_y_orig), random_state=random_state
                )
            else:
                train_y = copy.deepcopy(train_y_orig)
                test_y = copy.deepcopy(test_y_orig)
            train_y.name = "True"
            test_y.name = "True"
            split_y = {"train": train_y.squeeze(), "test": test_y.squeeze()}
            if (clone_model or not _is_fitted(estimator)) and HasMethods(
                "fit"
            ).is_satisfied_by(estimator):
                model_to_fit = clone(estimator)
                if "sample_weight" in kwargs.keys() and sample_weight is not None:
                    if "sample_weight" in kwargs.keys():
                        sample_weight = kwargs["sample_weight"]
                    if isinstance(sample_weight, pd.Series):
                        weights = sample_weight
                    # if isinstance(sample_weight, dict):
                    elif isinstance(sample_weight, pd.DataFrame):
                        weights = sample_weight[-1].squeeze()
                    else:
                        weights = pd.Series(sample_weight)
                    if split_y["train"].index.difference(weights.index).size > 0:
                        print(
                            "\n\nWeights index differs from training index!!!\n\n{}".format(
                                split_y["train"].index.difference(weights.index)
                            )
                        )
                        weights.index = split_y["train"].index
                    fit_est = model_to_fit.fit(
                        split_X["train"],
                        split_y["train"],
                        sample_weight=weights[split_y["train"].index],
                    )
                else:
                    fit_est = model_to_fit.fit(X=split_X["train"], y=split_y["train"])
            else:
                fit_est = estimator
            for fname in results.keys():
                for split_set in results[fname].keys():
                    # print(getattr(fit_est, fname)(X=split_X[split_set]))
                    result_df = pd.DataFrame(
                        getattr(fit_est, fname)(X=split_X[split_set]),
                        index=split_X[split_set].index,
                    ).squeeze()
                    if isinstance(result_df, pd.Series):
                        total_df = result_df.copy().to_frame()
                        total_df.columns = [str(fname)]
                        result_df.name = split_set
                    else:
                        total_df = result_df.copy()
                        total_df.columns = [
                            "{}_{}".format(fname, i)
                            for i in np.arange(total_df.shape[1])
                        ]
                    results[fname][split_set].append(result_df)
                    results_dict[r_name][fname][split_set].append(result_df)
                    total_df = total_df.merge(
                        split_y[split_set], right_index=True, left_index=True
                    )
                    total_df.insert(loc=0, column="CV_Fold", value=i)
                    total_df.insert(loc=0, column="Split", value=split_set)
                    # total_df.insert(loc=0, column="Method", value=fname)
                    total_df.insert(loc=0, column="Labels", value=r_name)
                    total_df.reset_index(names="INCHI_KEY", inplace=True)
                    # print(total_df)
                long_lists[fname].append(total_df)
        i += 1
    if i > 0:
        melted_dict = dict()
        for k, v in long_lists.items():
            melted_dict[k] = pd.concat(v)
    else:
        raise ValueError
    return results_dict, melted_dict, tuple([test_idx_list])


def score_long_form(func, x, true_col, data_cols=None, remove_cols=None, **kwargs):
    """

    Parameters
    ----------
    func : callable, scoring function
    x : pd.DataFrame, lambda variable containing data
    true_col : str, y_true
    data_cols : pd.Series | pd.DataFrame | None, predicted/calculated per-sample values
    kwargs : dict, kwargs for scoring function

    Returns
    -------

    """
    if remove_cols is not None:
        x = x.copy().drop(columns=remove_cols)
    true_df = x[true_col]
    if data_cols is None:
        pred_df = x.drop(columns=true_col)
    else:
        pred_df = x[data_cols]
    score = func(true_df, pred_df, **kwargs)
    return score


def relative_brier_score(
    y_true,
    y_proba,
    y_prior=None,
    clips=(0, 1.0),
    sample_weight=None,
    class_weight="balanced",
    normalize="improved",
    improve_threshold=(0.0, 1.0),
    prob_thresholds=(0.0, 1.0),
    power=2,
    best_k=None,
):
    """

     Parameters
     ----------
     best_k
     improve_threshold
     y_true : pd.Series
     Ground truth class labels
     y_proba : pd.Series | pd.DataFrame
     Predicted probabilities. If Series, should be correct-class probabilities.
     y_prior : pd.Series | pd.DataFrame
     Prior probability of classes. If Series, should be correct-class probabilities.
     clips : tuple[float, float]
     Lower and upper bounds for skill score (y_proba - y_prior)
     sample_weight : pd.Series
     Weights for calculating final statistic
     class_weight : "Balanced" | None
     normalize : float | "correct"
     Criteria for which skill scores to include in score calculation.
     power

     Returns
     -------
    rel_brier_score : float
    Relative Brier Score
    y_proba_brier : pd.Series
    Observations (transformed but not sample/class weighted) used in calculating score.
    Same as difference in correct-class probabilities of all samples if normalize="improved"
    brier_obs.index: pd.Index
    Index of samples used to calculate score.
    """
    y_proba_correct = correct_class_probs(y_true, y_proba)
    y_prior_correct = correct_class_probs(y_true, y_prior)
    y_proba_thresh = y_proba_correct[
        (y_proba_correct > prob_thresholds[0]) & (y_proba_correct < prob_thresholds[1])
    ]
    y_prior_thresh = y_prior_correct[y_proba_thresh.index]
    # Relative scoring
    proba_diff = y_proba_thresh - y_prior_thresh
    proba_diff_thresh = proba_diff[
        proba_diff > improve_threshold[0]  # & (proba_diff < improve_threshold[1])
    ]
    proba_power = proba_diff_thresh.pow(power)
    if class_weight == "balanced":
        class_wts = pd.Series(
            compute_sample_weight(class_weight="balanced", y=y_true), index=y_true.index
        )[proba_diff_thresh.index]
    elif isinstance(class_weight, pd.Series):
        class_wts = class_weight
    else:
        class_wts = pd.Series(1, index=proba_diff_thresh)
    if isinstance(class_wts, pd.Series):
        if sample_weight is not None and sample_weight.sum() > 0:
            combo_weight = sample_weight * class_wts
        else:
            combo_weight = class_weight
    else:
        if sample_weight is not None and sample_weight.sum() > 0:
            combo_weight = sample_weight
        else:
            combo_weight = pd.Series(1, index=proba_power.index)
    balanced_brier = proba_power.copy() * combo_weight
    if best_k is not None:
        if 0.0 < best_k <= 1.0:
            n_best = int(min(best_k * y_true.shape[0], balanced_brier.shape[0]))
        else:
            n_best = int(min(best_k, balanced_brier.shape[0]))
    else:
        n_best = -1
    if combo_weight is not None and combo_weight.sum() > 0:
        weighted_brier = combo_weight.loc[balanced_brier.index] * balanced_brier
        best_brier = weighted_brier.sort_values(ascending=False).iloc[
            : min(n_best, weighted_brier.shape[0])
        ]
        normalized = best_brier.sum() / combo_weight.loc[best_brier.index]
    else:
        normalized = balanced_brier.sort_values(ascending=False)[:n_best].sum() / n_best
    # Where 2r - r^2 is quadratic "scoring".
    sample_prop = proba_diff_thresh.shape[0] / y_true.shape[0]
    score = normalized / np.sqrt(sample_prop)
    score_samples = proba_diff_thresh.index.to_series()
    print("{} improvements in probability for score of {}.".format(best_brier, score))
    return score, score_samples


def correct_class_probs(y_true, y_proba):
    onehot_labels, _ = samples.one_hot_conversion(y_true)
    y_proba_correct = y_proba.mul(onehot_labels).sum(axis=1).squeeze()
    return y_proba_correct


def _format_cv_generalized_score(results_correct, i, hue_category=None):
    # results_dict_df = [pd.DataFrame.from_records(r) for r, v in results_correct.items()]
    # print(results_dict_df)
    # print(results_dict_df.explode().reset_index())
    # print([ser.explode().reset_index() for col, ser in results_dict_df.items()])
    # exploded_list = results_dict_df.melt()
    exploded = results_correct.melt(
        id_vars=["Metric", "Split", "CV Fold"], value_name="score"
    )
    exploded.insert(loc=0, column="Subset", value=i)
    if hue_category is not None:
        exploded.insert(loc=0, column="Labels", value=hue_category)
    return exploded


def collate_score_dicts(score_dict_list):
    # Rearrange a list of model results into a dictionary whose keys are score names and values are lists of model scores.
    assert len(score_dict_list) > 0
    collated = dict.fromkeys(score_dict_list[1].keys(), list())
    # print("Score dict list: {}".format(score_dict_list))
    for score_dict in score_dict_list:
        for k, v in score_dict.items():
            collated[k].append(v)
    # [print("Collated {}: {}".format(k, c)) for k, c in collated.items()]
    check_model_bounds(collated)
    return collated


def check_model_bounds(collated_scores):
    # Checks that metric scores fall within their range of possible value. For debugging erroneous score results.
    bounds_dict = get_score_bounds()
    for name, scores in collated_scores.items():
        bounds = bounds_dict[name]
        if np.isscalar(scores):
            values = [scores]
        else:
            values = scores
        for v in values:
            if bounds[0] > v > bounds[1]:
                print("{} score  of {} is out of bounds!".format(name, v))
                raise ValueError
    return True


def summarize_scores(score_dict_list, check_scores=True):
    # Calculates summary statistics of CV runs.
    assert len(score_dict_list) > 0
    collated_scores = collate_score_dicts(score_dict_list)
    score_series, score_series_dict = list(), dict()
    summary_zip = dict(
        zip(
            ["Mean", "GeoMean", "StDev", "Median", "Min", "Max"],
            [np.mean, gmean, np.std, np.median, np.min, np.max],
        )
    )
    # print(collated_scores)
    for k, v in collated_scores.items():
        if not np.iterable(v) or not all(
            [isinstance(val, numbers.Number) for val in v]
        ):
            print("Collated values for {} are not scalars: {}".format(k, v))
            continue
        # print(v)
        # print("Dictionary values: {}".format([(name, s(v)) for name, s in summary_zip.items()]))
        score_summary = defaultdict(
            None, [(name, s(v)) for name, s in summary_zip.items()]
        )
        score_ser = pd.Series(score_summary, name=k)
        # print("Calculating {} summary: \n{}".format(k, score_ser))
        if not score_ser.empty:
            score_series.append(score_ser)
            score_series_dict[k] = score_ser
        else:
            print("No scores found for {}: {}".format(k, v))
            raise RuntimeError
    assert len(score_series) > 0 and not any([a.empty for a in score_series])
    score_series_dict = pd.DataFrame.from_dict(score_series_dict, orient="columns")
    score_df = pd.concat(score_series, axis=1)
    if check_scores:
        check_model_bounds(score_df.to_dict(orient="list"))
    assert not score_df.empty
    return score_df


def combine_scores(
    cv_dev_score_dict,
    cv_eval_score_dict,
    dev_score_summary,
    eva_score_summary,
    score_name_list,
):
    for score_name in score_name_list:
        dev_score_summary[score_name] = list()
        eva_score_summary[score_name] = list()
        for stat_name, stat in zip(
            ["Mean", "StDev", "Median", "Min", "Max"],
            [np.mean, np.std, np.median, np.min, np.max],
        ):
            dev_score_summary[stat_name].append(
                "{:.5f}".format(stat([[s] for s in cv_dev_score_dict[score_name]]))
            )
            eva_score_summary[stat_name].append(
                "{:.5f}".format(stat([[s] for s in cv_eval_score_dict[score_name]]))
            )
    dev_df = pd.DataFrame.from_records(
        data=cv_dev_score_dict,
        index=[cv_dev_score_dict.keys()],
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    eva_df = pd.DataFrame.from_records(
        data=cv_eval_score_dict,
        index=[cv_eval_score_dict.keys()],
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    return dev_df, eva_df


def score_model(
    estimator,
    feature_df,
    labels,
    scorer_tuple=None,
    scalar_only=True,
    check_bounds=False,
):
    # Returns a dictionary of scores for a model, given the ground truth.
    cv_score_dict = defaultdict()
    preds = estimator.predict(feature_df)
    # select_dev_prob = fitted.predict_proba(X_dev)
    if scorer_tuple is None:
        # score_dict = get_pred_score_funcs()
        scorer_tuple = ("Balanced Accuracy", balanced_accuracy_score)
        # scorer_tuple = ("MCC", matthews_corrcoef)
        # for score_name, score_obj in scorer_tuple:
    score_name, score_obj = scorer_tuple
    score = score_obj(labels, preds)
    if (not scalar_only or np.isscalar(score)) and np.real(score):
        cv_score_dict[score_name] = score
    if check_bounds:
        check_model_bounds(cv_score_dict)
    return cv_score_dict
