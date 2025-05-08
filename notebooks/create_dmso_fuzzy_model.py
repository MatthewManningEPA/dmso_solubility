import logging
import os
import pickle
import pprint
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_log_error,
)
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    KFold,
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import _check_y

import cv_tools
import fp_regressor
import samples
import scoring
import scoring_metrics
from dataset_creation import (
    _set_paths,
    assemble_dataset,
    assemble_dmso_dataset,
    preprocess_data,
)
from epa_enamine_visualizer import plot_clf_model_displays, plot_model_scores
from fuzzy_controller import (
    create_submodels,
    save_plot_submodel_results,
)
from scoring_metrics import get_confusion_weights


def optimize_tree(feature_df, labels, model, scoring, cv, path_dict):
    """
    forest_params = {
        "bootstrap": ["False"],
        "n_estimators": [10, 50, 100, 250],
        "min_impurity_decrease": [0, 0.0005, 0.001, 0.005, 0.01],
        "max_features": [3, 5, 6, 7, 9, None],
        "max_leaf_nodes": [100, 150, 200, 250, None],
        "random_state": [0],
    }
        gscv = GridSearchCV(
            m,
            param_grid=forest_params,
            scoring=make_scorer(matthew_correlation_coeff),
            n_jobs=-1,
            cv=RepeatedStratifiedKFold,
        )
    """
    optimum_path = "{}optimal_params.csv".format(path_dict["exp_dir"])
    all_param_path = "{}hyperparam_results.csv".format(path_dict["exp_dir"])
    hyper_score_path = "{}hyperparam_scores.csv".format(path_dict["exp_dir"])
    if True and os.path.isfile(optimum_path):
        optimal_params = (
            pd.read_csv(optimum_path, index_col=0, na_filter=False).squeeze().to_dict()
        )
        for k, v in optimal_params.items():
            print(v)
            if v == "True":
                optimal_params[k] = True
            elif v == "False":
                optimal_params[k] = False
            elif v is np.nan or v == "":
                optimal_params[k] = None
            elif isinstance(v, str) and "." in v:
                optimal_params[k] = float(v)
            elif isinstance(v, str) and v.isnumeric():
                optimal_params[k] = int(v)
            else:
                optimal_params[k] = v
    elif True:
        param_grid = {
            # "max_depth": [None, 25, 15],
            "n_estimators": [50, 250, 1000],
            "min_impurity_decrease": [0, 0.0025],
            "max_leaf_nodes": [150, 250],
            "class_weight": ["balanced", "balanced_subsample"],
            "bootstrap": [True, False],
        }
        tree_search = GridSearchCV(
            estimator=clone(model),
            param_grid=ParameterGrid(param_grid).param_grid,
            scoring=scoring,
            n_jobs=-1,
            cv=cv,
            return_train_score=True,
            error_score="raise",
        )
        tree_search.fit(X=feature_df, y=labels)
        optimal_params = tree_search.best_params_
        pd.Series(optimal_params).to_csv(optimum_path)
        hyper_rankings = pd.Series(tree_search.cv_results_["rank_test_score"]).tolist()
        pd.DataFrame.from_records(tree_search.cv_results_["params"]).to_csv(
            all_param_path
        )
        hyper_scores = (
            pd.Series(tree_search.cv_results_["mean_test_score"])
            .reindex(index=hyper_rankings)
            .sort_index()
        )
        hyper_scores.to_csv(hyper_score_path)
        try:
            pd.DataFrame.from_records(tree_search.cv_results_["params"]).sort_index(
                key=lambda x: pd.Series(hyper_rankings)
            ).to_csv(all_param_path, index_label="Ranking")
        except:
            pd.DataFrame.from_records(tree_search.cv_results_["params"]).sort_index(
                key=lambda x: pd.Series(hyper_rankings)
            ).to_csv(all_param_path, index_label="Ranking")
        finally:
            hyper_scores = (
                pd.Series(tree_search.cv_results_["mean_test_score"])
                .reindex(index=hyper_rankings)
                .sort_index()
            )
            hyper_scores.to_csv(hyper_score_path)
    estimator = RandomForestClassifier().set_params(**optimal_params)
    return estimator


def main():
    classification = True
    dataset = "all_dmso"
    frac_enamine = 0.5
    rus = False
    plot_all_feature_displays = True
    estimator_name = "error_weighted_score"
    n_subsets = 2
    n_epochs = 2
    (
        data_tuple,
        select_params,
        path_dict,
        estimator,
    ) = setup_params(
        dataset, frac_enamine, classification, rus=rus, estimator_name=estimator_name
    )
    select_params["n_subsets"] = n_subsets
    select_params["n_epochs"] = n_epochs
    train_df, train_labels, test_df, test_labels = data_tuple
    model_dir = path_dict["exp_dir"]
    os.makedirs(model_dir, exist_ok=True)
    # Plot all features model.
    path_dict["base_dir"] = "{}base_model/".format(path_dict["exp_dir"])
    path_dict["base_model"] = "{}frozen_base_model.pkl".format(path_dict["base_dir"])
    path_dict["base_weights"] = "{}/booster_weights.pkl".format(path_dict["base_dir"])
    path_dict["base_probs"] = "{}/booster_probs.pkl".format(path_dict["base_dir"])
    (
        train_df,
        train_labels,
        test_df,
        test_labels,
        preprocessor,
        best_corrs,
        cross_corr,
    ) = preprocess_data(
        (train_df, train_labels), (test_df, test_labels), select_params, path_dict
    )
    logger.debug(
        "Discretized Labels: Value Counts:\n{}".format(
            pprint.pformat(train_labels.value_counts())
        )
    )
    print("Running model in directory: {}".format(path_dict["parent_dir"]))
    search_features = train_df.columns.tolist()
    logger.debug("Selecting from {} features.".format(len(search_features)))

    if (
        not os.path.isfile(path_dict["base_model"])
        or not os.path.isfile(path_dict["base_weights"])
        or not os.path.isfile(path_dict["base_probs"])
    ):
        booster, booster_probs, booster_weights = train_base_model(
            train_df=train_df,
            train_labels=train_labels,
            select_params=select_params,
            path_dict=path_dict,
            plot_all_feature_displays=plot_all_feature_displays,
        )
        with open(path_dict["base_model"], "wb") as f:
            pickle.dump(FrozenEstimator(booster), f)
        booster_weights.to_pickle(path_dict["base_weights"])
        booster_probs.to_pickle(path_dict["base_probs"])
    else:
        with open(path_dict["base_model"], "rb") as f:
            booster = pickle.load(f)
        booster_weights = pd.read_pickle(path_dict["base_weights"])
        booster_probs = pd.read_pickle(path_dict["base_probs"])
    model_list = list()
    if os.path.isdir("{}final/".format(path_dict["exp_dir"])):
        model_list = fp_regressor.assemble_models(
            final_path="{}final/".format(path_dict["exp_dir"]),
            base_path=path_dict["base_dir"],
        )
    if len(model_list) <= n_subsets:
        print(len(model_list))
        fuzzy_list, ffh_list = create_submodels(
            estimator=estimator,
            feature_df=train_df,
            labels=train_labels,
            estimator_name=estimator_name,
            select_params=select_params,
            path_dict=path_dict,
            save_dir=model_dir,
        )
        features_list, pred_list, prob_list, score_list, weights_list = (
            save_plot_submodel_results(
                fuzzy_list=fuzzy_list,
                ffh_list=ffh_list,
                feature_df=train_df,
                labels=train_labels,
                estimator_name=estimator_name,
                select_params=select_params,
                save_dir=model_dir,
            )
        )
    for dev_X, dev_y, eval_X, eval_y in cv_tools.split_df(train_df, train_labels):
        class_weights = compute_sample_weight(
            class_weight="balanced",
            y=train_labels,
        )
        # booster_weights = base_weights[train_labels.index]
        sample_weight = booster_weights.loc[train_labels.index] * class_weights
        gating_estimtor = fp_regressor.train_predict(
            model_path=model_dir,
            train_data=(dev_X, dev_y),
            test_data=(eval_X, eval_y),
            sample_weight=sample_weight,
        )
        eval_pred_proba = gating_estimtor.predict_proba(eval_X)
        eval_pred = eval_pred_proba.where(lambda x: x < 0.5, [0, 1])
        brier_loss = brier_score_loss(
            eval_y, eval_pred_proba, sample_weight=sample_weight, pos_label=1
        )
        class_brier_loss = brier_score_loss(
            eval_y, eval_pred_proba, sample_weight=class_weights, pos_label=1
        )
        boost_bal_acc = balanced_accuracy_score(
            test_labels, eval_pred, sample_weight=booster_weights
        )
        boost_mcc = matthews_corrcoef(test_labels, eval_pred)
        bal_acc = balanced_accuracy_score(test_labels, eval_pred)
        mcc = matthews_corrcoef(test_labels, eval_pred)

        print("Unweighted Balanced Accuracy: {:.5f}".format(bal_acc))
        print("Unweighted MCC: {:.5f}".format(mcc))
        print("Class-Weighted Brier Loss: {:.5f}".format(class_brier_loss))
        print("Full-Weighted Brier Loss: {:.5f}".format(brier_loss))
        print("Booster-weighted Balanced Accuracy: {:.5f}".format(boost_bal_acc))
        print("Booster-weighted MCC: {:.5f}".format(boost_mcc))

    exit()
    """
    if False and (
        len(subsets) < 2
        or isinstance(subsets, str)
        or any([len(subset) < 2 for subset in subsets])
    ):
        print("Size of feature set to be plotted: {}".format(len(subsets)))
        plot_models(
            feature_df=train_df,
            labels=train_labels,
            select_params=select_params,
            estimator_name=estimator_name,
            estimator=estimator,
            path_dict=path_dict,
            score_tups=score_tups,
            subsets_list=subsets,
            save_path="{}subset_scores.png".format(path_dict["exp_dir"]),
        )
    return (
        (train_df, test_df),
        (train_labels, test_labels),
        subsets,
        scores,)
        """


def train_base_model(
    train_df,
    train_labels,
    select_params,
    path_dict,
    plot_all_feature_displays,
    weighted_booster=True,
):
    """

    Parameters
    ----------
    weighted_booster
    train_df : pd.DataFrame
    train_labels : pd.Series
    select_params : dict
    path_dict : dict
    plot_all_feature_displays : bool

    Returns
    -------
    booster : estimator
    Gradient Booster with parameters used. Untrained due to cross-validation.
    probs : pd.Series | pd.DataFrame
    CV test output predictions from trained booster
    booster_weights : pd.Series
    Sample weights assigned based on booster output predictions
    """

    """
    best_estimator = optimize_tree(
        train_df,
        train_labels,
        estimator,
        select_params["scoring"],
        select_params["cv"],
        path_dict,
    )
    """
    from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

    # Set initial sample weights to SAMME weights from boosting.
    custom = False
    booster_name = "extra"
    base_dir = "{}/base_model/".format(path_dict["exp_dir"])
    # path_dict["base_weights"] = "{}/booster_weights.pkl".format(base_dir)
    # path_dict["base_probs"] = "{}/booster_probs.pkl".format(base_dir)
    # path_dict["base_model"] = "{}/booster_model.pkl".format(base_dir)
    if booster_name == "hist_grad":
        booster_params = {
            "max_iter": 100,
            "max_depth": 5,
            "max_bins": 100,
            "min_samples_leaf": 2,
            # "max_features": 0.5,
            "class_weight": "balanced",
            "n_iter_no_change": 10,
            "tol": 1e-6,
            "learning_rate": 0.05,
        }
        booster = HistGradientBoostingClassifier(
            # l2_regularization=0.025,
            # max_depth=7,
            #  verbose=1,
        )
        booster.set_params(**booster_params)
    elif booster_name == "ada_extra":
        extra = ExtraTreesClassifier()
        extra_params = {
            # "min_samples_split": 10,
            "max_depth": 3,
            # "min_impurity_decrease": 0.05,
            "max_leaf_nodes": 25,
            "class_weight": "balanced",
            "n_jobs": -4,
        }
        extra.set_params(**extra_params)
        booster = AdaBoostClassifier(extra, n_estimators=5, learning_rate=0.025)
    elif booster_name == "extra":
        extra_params = {
            "n_estimators": 5000,
            "max_depth": 6,
            # "min_impurity_decrease": 0.05,
            "max_leaf_nodes": 35,
            # "max_features": "log2",
            "class_weight": "balanced",
            "n_jobs": -2,
        }
        extra = ExtraTreesClassifier()
        extra.set_params(**extra_params)
        booster = extra
    else:
        booster = None
        booster_weights = None
    if (
        os.path.isfile(path_dict["base_weights"])
        and os.path.isfile(path_dict["base_probs"])
        and os.path.isfile(path_dict["base_model"])
    ):
        probs = pd.read_pickle(path_dict["base_probs"]).squeeze()
        booster_weights = pd.read_pickle(path_dict["base_weights"]).squeeze()
        with open(path_dict["base_model"], "rb") as f:
            booster = pickle.load(f)
        # samples.weight_by_proba(train_labels, probs, prob_thresholds=select_params["brier_clips"])
    elif True:
        if weighted_booster and (booster is not None) and booster_name != "extra":
            onehot = pd.concat([1 - train_labels, train_labels], axis=1)
            cv_list, staged_df_list = list(), list()
            for dev_X, dev_y, eval_X, eval_y in cv_tools.split_df(
                train_df, train_labels
            ):
                staged_list, con_list = list(), list()
                onehot_test = onehot.loc[eval_y.index]
                booster.fit(X=dev_X, y=dev_y)
                if booster_name != "extra":
                    for stage_i, stage_est in enumerate(booster.estimators_):
                        booster_probs = stage_est.predict_proba(X=eval_X)
                        pred_df = pd.DataFrame(booster_probs, index=eval_X.index)
                        staged_list.append(pred_df.mul(onehot_test.sum(axis=1)))
                    # print(booster_probs)
                    # print(staged_list)
                    staged_df_list.append(pd.concat(staged_list))
                    con_list.append(
                        pd.DataFrame(
                            booster.predict_proba(X=eval_X), index=eval_X.index
                        )
                    )
                else:
                    staged_df_list.append(
                        pd.DataFrame(
                            booster.predict_proba(X=eval_X), index=eval_X.index
                        )
                    )
            staged_df = pd.concat(staged_df_list)
            prob_dist_list = [a[1] for a in staged_df.items()]
            fig_path = "{}base_model_prob_distance_heatmap.png".format(
                path_dict["exp_dir"]
            )
            htmap = plot_prob_dist_heatmap(
                prob_dist_list=prob_dist_list,
                save_path=fig_path,
                include_thresholds=(0.0, None),
            )

            i_list = np.arange(staged_df.shape[1])
            if booster_name == "extra":
                probs = staged_df
            else:
                weight_ser = 1 / pd.Series(
                    [booster.learning_rate * np.log2(w + 3) for w in i_list]
                )
                probs = staged_df.mul(weight_ser).sum(axis=1).div(weight_ser.sum())
                assert staged_df.shape[1] == weight_ser.shape[0]
                assert staged_df.shape[0] == train_labels.shape[0]
        elif custom:
            results, score_dict, long_form, test_idx_list, score_dict = (
                scoring.cv_model_generalized(
                    booster,
                    train_df,
                    train_labels,
                    v=select_params["cv"],
                )
            )
            # booster = FrozenEstimator(booster.fit(train_df, train_labels))
            # probs = pd.DataFrame(booster.predict_proba(train_df), index=train_df.index)
            probs = pd.concat(results["predict_proba"]["test"])
        else:
            probs = cross_val_predict(
                estimator=booster,
                X=train_df,
                y=train_labels,
                method="predict_proba",
                # cv=select_params["cv"],
                # n_jobs=-1,
                params=None,
            )
            probs = pd.DataFrame(probs, index=train_labels.index)
        # print("HGB Probabilities: {}".format(probs))
        assert probs.shape[0] == train_labels.shape[0]
        booster_weights = samples.weight_by_proba(
            y_true=train_labels,
            probs=probs,
            prob_thresholds=select_params["brier_clips"],
        )
        # probs.to_csv(path_dict["base_probs"], index_label="INCHI_KEY")
        probs.to_pickle(path_dict["base_probs"])
        booster_weights.to_pickle(path_dict["base_weights"])
    if plot_all_feature_displays:
        display_dir = "{}base_model/".format(path_dict["exp_dir"])
        os.makedirs(display_dir, exist_ok=True)

        f_stats, f_pvals = f_classif(X=train_df, y=1 - train_labels)
        f_logp = (
            pd.Series(f_pvals, index=train_df.columns)
            .sort_values()
            .map(np.log)
            .mul(-1)
            .clip(upper=100.0)
        )
        subsets = tuple(
            [
                train_df.columns.to_series().sample(
                    n=select_params["max_features_out"], weights=f_logp
                )
                for a in range(2)
            ]
            + [train_df.columns.tolist()]
        )
        subsets = tuple(
            [
                train_df.columns.tolist(),
            ]
        )
        score_results, score_plot = plot_model_scores(
            feature_df=train_df,
            train_labels=train_labels,
            score_tups=select_params["score_tups"],
            estimator_list=[booster],
            subsets=subsets,
            cv=select_params["cv"],
        )
        score_plot.savefig("{}base_model_scores.png".format(display_dir))
        all_feat_displays = plot_clf_model_displays(
            estimator=booster,
            estimator_name=booster_name,
            train_df=train_df,
            train_labels=train_labels,
            select_params=select_params,
            subset_dir=display_dir,
            display_labels=["Insoluble", "Soluble"],
            # sample_weight=booster_weights,
            probs=probs,
        )
        plt.close()
    return booster, probs, booster_weights


def plot_prob_dist_heatmap(
    prob_dist_list,
    save_path=None,
    symmetric=False,
    sample_weight=None,
    dist_metric=root_mean_squared_log_error,
    include_thresholds=None,
):
    """

    Parameters
    ----------
    include_thresholds
    dist_metric
    prob_dist_list : list[pd.Series]
    List of correct-class probabilities.
    save_path : str
    symmetric : bool
    sample_weight : pd.Series
    """
    prob_distances = scoring_metrics.model_prediction_distance(
        predicts_list=prob_dist_list,
        metric=dist_metric,
        symmetric=symmetric,
        sample_weight=sample_weight,
        include_thresholds=include_thresholds,
    )
    plt.figure(figsize=(12, 8), dpi=300)
    dist_plot = sns.heatmap(
        prob_distances,
        square=True,
        robust=True,
        yticklabels=False,
        cmap="coolwarm",
    )
    plt.savefig(save_path)
    plt.close()
    return prob_distances


def setup_params(dataset_name, frac_enamine, estimator, rus, estimator_name):
    if True or is_classifier(estimator):
        # Keep score_func at predict_proba. This only affects how selection weight samples.
        # Plotting, etc is handled by scoring function signature.
        select_params, estimator = _set_params(
            score_func=balanced_accuracy_score, score_name="balanced_accuracy"
        )
        # Model scoring metrics.
        select_params["cv"] = partial(
            StratifiedKFold,
            shuffle=True,
            # RepeatedStratifiedKFold,
            # n_splits=3,
            # n_repeats=3,
            random_state=0,
        )
        select_params["score_tups"] = (
            ("MCC", matthews_corrcoef),
            ("Balanced Acc", balanced_accuracy_score),
            # ("Brier Loss", brier_score_loss),
        )
        print(estimator.__repr__())
    else:
        select_params, estimator = _set_params(
            score_func=partial(
                root_mean_squared_log_error,
                multioutput="raw_values",
            ),
            score_name="RMSLE",
            model_output="predict",
            greater_is_better=False,
        )
        select_params["cv"] = partial(KFold, shuffle=True, random_state=0)
        # estimator = LinearRegression()
        # estimator_name = "linear_reg"
        select_params["score_tups"] = (
            ("r2", r2_score),
            ("mape", mean_absolute_percentage_error),
        )
        # Labels are scikit-learn compatible.
    if "epa" in dataset_name or "enamine" in dataset_name or "dmso" in dataset_name:
        if rus:
            parent_dir = "{}{}_rus/".format(os.environ.get("MODEL_DIR"), dataset_name)
        else:
            parent_dir = "{}{}/".format(os.environ.get("MODEL_DIR"), dataset_name)
        data_dir = parent_dir
        path_dict = _set_paths(parent_dir, data_dir, select_params)
        train_df, train_labels, test_df, test_labels = assemble_dmso_dataset(
            dataset_name, select_params, path_dict, frac_enamine=frac_enamine, rus=True
        )
    else:
        train_data, test_data, preprocessor, data_dir, parent_dir = assemble_dataset(
            dataset_name
        )
        train_df, train_labels = train_data
        test_df, test_labels = test_data
        path_dict = _set_paths(parent_dir, data_dir, select_params)
    path_dict["exp_dir"] = "{}{}/".format(parent_dir, estimator_name)
    os.makedirs(path_dict["exp_dir"], exist_ok=True)
    train_labels = pd.Series(
        _check_y(train_labels.copy().squeeze(), estimator=estimator),
        index=train_labels.index,
        name="Labels",
    )
    test_labels = pd.Series(
        _check_y(test_labels.copy().squeeze(), estimator=estimator),
        index=test_labels.index,
        name="Labels",
    )
    train_df.dropna(axis="columns", inplace=True)
    train_df.dropna(axis="index", inplace=True)
    test_df = test_df[train_df.columns.copy()]
    output_data = (
        train_df,
        train_labels,
        test_df,
        test_labels,
    )
    return (
        output_data,
        select_params,
        path_dict,
        estimator,
    )


def _set_params(
    score_func=matthews_corrcoef,
    score_name=None,
    model_output="predict_proba",
    loss_func=mean_absolute_percentage_error,
    greater_is_better=True,
):
    """
    Set parameters for feature selection

    Parameters
    ----------
    score_func : callable
    score_name : str
    model_output : str
    loss_func : callab;e
    greater_is_better : bool

    Returns
    -------
    select_params : dict
    """
    select_params = {
        "thresh_xc": 0.95,
        "fails_min_vif": 100,
        "fails_min_perm": 0,
        "fails_min_sfs": 0,
        "W_confusion": get_confusion_weights(),
        "loss_func": loss_func,
        "lang_lambda": 0.1,
        # Features In Use
        "max_trials": 10,
        "randomize_init_weights": True,
        "max_features_out": 30,
        "min_features_out": 10,
        "tol": 0.01,
        "n_iter_no_change": 3,
        "corr_method": "spearman",
        "xc_method": "pearson",
        "thresh_reset": 0.05,
        "n_vif_choices": 5,
        "add_n_feats": 3,
        "features_min_vif": 8,
        "features_min_perm": 12,
        "features_min_sfs": 15,
        "thresh_vif": 30,
        "thresh_perm": 0.025,
        "thresh_sfs": 0,
        "thresh_sfs_cleanup": 0,
        "cv": partial(StratifiedKFold, shuffle=True, random_state=14),
        "importance": True,
        # "scoring": make_scorer(three_class_solubility),
        "scoring": score_func,
        "scorer": make_scorer(score_func, greater_is_better=greater_is_better),
        "model_output": model_output,
        "sample_weight": None,
        "base_weight": None,
        "pos_label": 0,
        "brier_clips": (0.1, 1.0),
        "prob_thresholds": (0.15, 0.85),
        "pred_combine": "best",
        "best_k": 0.05,
    }
    if score_name is None:
        select_params["score_name"] = str(score_func.__repr__())
    else:
        select_params["score_name"] = score_name
    estimator = RandomForestClassifier().set_params(
        **{
            "n_estimators": 50,
            "class_weight": "balanced",
            # "max_depth": 15,
            "max_leaf_nodes": 150,
            # "min_impurity_decrease": 0.001,
            "n_jobs": -3,
        }
    )
    return select_params, estimator


def _make_proba_residuals(data, labels=None, combine=True):
    resid = dict()
    for col_a in np.arange(data.shape[1]):
        for col_b in np.arange(col_a + 1, data.shape[1]):
            resid[(col_a, col_b)] = data.iloc[:, col_a] - data.iloc[:, col_b]
    if combine:
        resid = pd.DataFrame.from_dict(resid)
    return resid


if __name__ == "__main__":
    from sklearn import set_config

    sklearn.set_config(transform_output="pandas")
    logger = logging.getLogger(name="selection")
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        # warnings.simplefilter("error")
        main_dfs, main_labels, subset_dict, scores_dict = main()
