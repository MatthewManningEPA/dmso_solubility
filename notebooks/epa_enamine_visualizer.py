import copy
import itertools
import os
import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    cross_val_predict,
    KFold,
    LearningCurveDisplay,
    StratifiedKFold,
)
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.class_weight import compute_sample_weight

import scoring
import scoring_metrics


# from model_selection_epa_multiclass import _make_proba_residuals


def main():
    pass


if __name__ == "__main__":
    main()


def plot_model_scores(
    feature_df,
    train_labels,
    score_tups,
    estimator_list,
    subsets,
    cv=None,
    sample_weight=None,
    **kwargs
):
    assert len(subsets) > 0
    group_cols = ["Subset", "Labels", "Split", "CV_Fold"]
    long_cols = copy.deepcopy(group_cols)
    long_cols.extend(["Score", "Metric", "subset"])
    results_list = list()
    kwargs.update({"pos_label": 0})
    long_method_dict_lists = dict()
    subset_df_list = list()
    for i, (best_features, zipped_estimator) in enumerate(zip(subsets, estimator_list)):
        # name_scorer_tups = dict((k, make_scorer(v)) for k, v in score_tups)
        score_dict = dict.fromkeys([name for name, f in score_tups])
        if len(best_features) == 0:
            continue
            print("No features included for subset {}!!!".format(i))
        print("Sample weights for model score plots:\n{}".format(sample_weight))
        if sample_weight is None:
            weights = None
        elif isinstance(sample_weight, dict):
            weights = list(sample_weight.values())[i]
        else:
            weights = sample_weight
        all_results, long_form, test_idx_tuple = scoring.cv_model_generalized(
            estimator=zipped_estimator,
            feature_df=feature_df[list(best_features)],
            labels=train_labels,
            cv=cv,
            methods=["predict"],
            return_train=True,
            sample_weight=weights,
            randomize_classes="both",
            **kwargs,
        )
        results_list.append(all_results)
        for s_name, s_func in score_tups:
            method_name = scoring.get_input_for_scorer(s_func)
            if method_name not in long_method_dict_lists.keys():
                long_method_dict_lists[method_name] = list()
            long_df = long_form[method_name]
            if "Subset" not in long_df.columns:
                long_df.insert(loc=0, column="Subset", value=i)
            long_method_dict_lists[method_name].append(long_df)
        # results_list.append(all_results)

        long_method_dict = dict()
        subset_ser_list = list()
        for method_name, method_df in long_method_dict_lists.items():
            long_method_dict[method_name] = pd.concat(method_df).reset_index(drop=True)
            for s_name, s_func in score_tups:
                method_name = scoring.get_input_for_scorer(s_func)
                score_input_cols = copy.deepcopy(group_cols)
                score_input_cols.append(method_name)
                score_input_cols.append("True")
                # print(long_method_dict[load_cv_results].columns)
                # print(long_method_dict[load_cv_results])
                grouper = (
                    long_method_dict[method_name]
                    .drop(
                        columns=[
                            a
                            for a in long_method_dict[method_name]
                            if a not in score_input_cols
                        ]
                    )
                    .groupby(group_cols, as_index=False)
                )
                for g_name, g_df in grouper:
                    """
                    base_info_df = (
                        long_method_dict[load_cv_results][group_cols]
                        .drop_duplicates()
                        .squeeze()
                    )
                    """
                    score_ser = pd.Series(data=g_name, index=group_cols)
                    long_score = scoring.score_long_form(
                        func=s_func,
                        x=g_df[["True", method_name]],
                        data_cols=method_name,
                        true_col="True",
                    )
                    score_ser["score"] = long_score
                    score_ser["Metric"] = s_name
                    score_ser["Subset"] = i
                    # print("score_ser")
                    subset_ser_list.append(score_ser)
        subset_df_list.append(pd.concat(subset_ser_list, axis=1).T)
        # print("Concatenated subset_ser_list")
        # print(pd.concat(subset_ser_list, axis=1).T)
    all_scores_long = pd.concat(subset_df_list, ignore_index=True)
    pprint.pp(all_scores_long)
    sns.set_theme(
        "notebook", palette=sns.color_palette("colorblind"), style="whitegrid"
    )
    plot = sns.catplot(
        all_scores_long,
        x="Split",
        y="score",
        hue="Labels",
        col="Subset",
        row="Metric",
        errorbar="se",
        aspect=0.5,
        margin_titles=True,
        kind="swarm",
        sharey="row",
    )
    plot.despine(left=True, bottom=False)
    plot.figure.subplots_adjust(wspace=0.0, hspace=0.15)
    return results_list, plot


def plot_dmso_model_displays():
    pass


def multi_model_clf_displays(
    estimator_tuples,
    train_df,
    train_labels,
    select_params,
    preds_list=None,
    probs_list=None,
    subset_dir=None,
    sample_weights_list=None,
    display_labels=None,
    frozen=False,
):
    preds, probs, weights = None, None, None
    axes_list = None
    save_dir = None
    for i_est, (est, est_name) in enumerate(estimator_tuples):
        preds, probs, weights = None, None, None
        if preds_list is not None:
            preds = preds_list[i_est]
        if probs_list is not None:
            probs = probs_list[i_est]
        if sample_weights_list is not None:
            weights = sample_weights_list[i_est]
        elif isinstance(sample_weights_list, (pd.Series)):
            weights = sample_weights_list
        if i_est == len(estimator_tuples) - 1:
            save_dir = subset_dir
        axes_list = plot_clf_model_displays(
            estimator=est,
            estimator_name=est_name,
            train_df=train_df,
            train_labels=train_labels,
            select_params=select_params,
            preds=preds,
            probs=probs,
            subset_dir=save_dir,
            sample_weight=weights,
            frozen=frozen,
            axes_list=axes_list,
        )
    return axes_list


def plot_clf_model_displays(
    estimator,
    estimator_name,
    train_df,
    train_labels,
    select_params,
    preds=None,
    probs=None,
    subset_dir=None,
    sample_weight=None,
    display_labels=None,
    frozen=False,
    axes_list=(None, None, None, None),
):
    """

    Parameters
    ----------
    select_params
    estimator : pd.BaseEstimator, ClassifierMixin
        Estimator to be used when constructing
    estimator_name : str
    train_df : pd.DataFrame
        Descriptors for training
    train_labels : pd.Series | pd.DataFrame
        Ground truth labels, DataFrame format for multioutput classifiers
    preds : pd.Series | pd.DataFrame
        Class predictions from estimator
    probs : pd.DataFrame
        Class-wise probability predictions from estimator
    subset_dir : str | Path-like
        Location to save figures
    sample_weight : pd.Series
        Previously calculated sample weights
    display_labels : Iterable[str]
    Returns
    -------

    """
    if all([a is None for a in axes_list]):
        rcd, det, lcd, cmd = axes_list

    class_weights = compute_sample_weight("balanced", y=train_labels)
    if estimator is None:
        return False
    if display_labels is None:
        display_labels = [str(s) for s in train_labels.unique()]
    if is_regressor(estimator):
        cv = KFold(shuffle=True, random_state=0)
    else:
        cv = StratifiedKFold(shuffle=True, random_state=0)
    if HasMethods("n_jobs").is_satisfied_by(estimator):
        estimator = estimator.set_params({"n_jobs": 1})
    else:
        estimator = estimator
    if False and (probs is None and not frozen):
        cv_results = cross_val_predict(
            estimator=clone(estimator),
            X=train_df,
            y=train_labels,
            cv=cv,
            n_jobs=-2,
            method="predict_proba",
            params={"sample_weight": sample_weight},
        )
        probs = pd.DataFrame(
            cv_results, index=train_labels.index, columns=display_labels
        )[display_labels[0]]
        if subset_dir is not None:
            os.makedirs(subset_dir, exist_ok=True)
            probs.to_csv(
                "{}{}_{}.csv".format(
                    subset_dir, estimator_name, select_params["model_output"]
                )
            )
    if False and (preds is None and not frozen):
        cv_results = cross_val_predict(
            estimator=estimator,
            X=train_df,
            y=train_labels,
            cv=cv,
            n_jobs=-2,
            method="predict",
            params={"sample_weight": sample_weight},
        )
        preds = pd.Series(cv_results, index=train_labels.index)
    # if sample_weight is not None and isinstance(sample_weight, pd.Series):
    roc_ax = None
    if preds is not None:
        roc_ax = RocCurveDisplay.from_predictions(
            y_true=train_labels,
            y_pred=preds[train_labels.index],
            pos_label=select_params["pos_label"],
            name="DMSO Insolubles",
            plot_chance_level=True,
            despine=True,
            ax=roc_ax,
        )
    else:

        roc_ax = RocCurveDisplay.from_estimator(
            estimator=estimator.fit(train_df, train_labels),
            X=train_df,
            y=train_labels,
            sample_weight=sample_weight,
            pos_label=select_params["pos_label"],
            name="DMSO Insolubles",
            plot_chance_level=True,
            despine=True,
            ax=roc_ax,
        )
    fig = roc_ax.figure_
    fig.set_dpi(300)
    # rcd.ax_.set(ylim=[0, 1.0])
    # rcd.ax_.set(xlim=[0, 1.0])
    if subset_dir is not None:
        fig.savefig("{}RocCurve_{}.png".format(subset_dir, estimator_name))
    if preds is not None:
        det = DetCurveDisplay.from_predictions(
            y_true=train_labels,
            y_pred=preds,
            pos_label=select_params["pos_label"],
            name="DMSO Insolubles",
            sample_weight=sample_weight,
        )
    else:
        det = DetCurveDisplay.from_estimator(
            estimator=estimator,
            X=train_df,
            y=train_labels,
            sample_weight=sample_weight,
            name="DMSO_Insolubles",
        )
    # det.figure_.set_dpi(300)
    # det.ax_.set(ylim=[-0.01, 1.01])
    # det.ax_.set(xlim=[-0.01, 1.01])
    if subset_dir is not None:
        det.figure_.savefig("{}DetCurve_{}.png".format(subset_dir, estimator_name))
    if is_classifier(estimator):
        cmd_fig, cmd_ax = plt.subplots()
        if preds is None or frozen:
            preds = cross_val_predict(
                estimator=estimator,
                X=train_df,
                y=train_labels,
                cv=cv,
                n_jobs=-2,
                method="predict",
                params={"sample_weight": sample_weight},
            )
        if isinstance(preds, (pd.Series, np.ndarray)):
            cmd = ConfusionMatrixDisplay.from_predictions(
                y_true=train_labels,
                y_pred=preds,
                display_labels=display_labels,
                normalize="true",
                cmap="Blues",
                ax=cmd_ax,
            )
        else:
            cmd = ConfusionMatrixDisplay.from_estimator(
                estimator=estimator,
                X=train_df,
                y=train_labels,
                normalize="true",
                cmap="Blues",
                display_labels=display_labels,
                ax=cmd_ax,
            )
        if subset_dir is not None:
            cmd_fig.savefig(
                "{}confusion_matrix_{}.png".format(subset_dir, estimator_name)
            )
    else:
        cmd = None
    if not frozen:
        lcd = LearningCurveDisplay.from_estimator(
            estimator=clone(estimator),
            X=train_df,
            y=train_labels,
            train_sizes=np.linspace(0.025, 0.95, num=20),
            scoring=select_params["scorer"],
            n_jobs=-4,
            shuffle=True,
            # random_state=0,
            score_name=select_params["score_name"],
        )
        lcd.figure_.set_dpi(300)
        # lcd.ax_.set(ylim=[-0.01, 1.01])
        if subset_dir is not None:
            lcd.figure_.savefig("{}lcd_{}.png".format(subset_dir, estimator_name))
    else:
        lcd = None
    return rcd, det, lcd, cmd


def _plot_proba_pairs(labels, subset_predicts, select_params):
    """

    Parameters
    ----------
    labels : pd.Series
    subset_predicts : list[dict[str[ strpd.Series | pd.DataFrame]
    select_params : dict

    Returns
    -------

    """
    pred_df, pred_long = _prep_for_proba_pairs(labels, subset_predicts, select_params)
    fg = sns.FacetGrid(
        pred_long, row="variable", col="variable", margin_titles=True, sharey=False
    )
    true_labels = pred_df["True"].unique()
    print(pred_df.loc[pred_df["True"] == true_labels[0]])
    for i, j in itertools.combinations(pred_df.drop(columns="True").columns, r=2):
        # print(col[0]*(pred_df.shape[1] - 1) + col[1] - 1)
        ax1 = fg.facet_axis(pred_df.columns.get_loc(i), pred_df.columns.get_loc(j))
        ax1.set(xlim=(-0.01, 1.01))
        ax1.set(ylim=(-0.01, 1.01))
        ax1 = sns.scatterplot(
            x=pred_df[pred_df["True"] == true_labels[0]][i],
            y=pred_df[pred_df["True"] == true_labels[0]][j],
            ax=ax1,
            size=0.01,
            legend=False,
            color="red",
        )
        ax2 = fg.facet_axis(pred_df.columns.get_loc(j), pred_df.columns.get_loc(i))
        ax2.set(xlim=(-0.01, 1.01))
        ax2.set(ylim=(-0.01, 1.01))
        sns.scatterplot(
            x=pred_df[pred_df["True"] == true_labels[1]][i],
            y=pred_df[pred_df["True"] == true_labels[1]][j],
            ax=ax2,
            size=0.01,
            legend=False,
            color="blue",
        )
    for i, clr in zip(pred_df.drop(columns="True").columns.tolist(), ("red", "blue")):
        sns.histplot(
            pred_long[pred_long["variable"] == i].drop(columns="variable"),
            x="value",
            hue="True",
            stat="density",
            common_norm=False,
            common_bins=True,
            color=clr,
            ax=fg.facet_axis(pred_df.columns.get_loc(i), pred_df.columns.get_loc(i)),
            legend=False,
        )
        # fg.fig.axes[col[0]+ col[1] - 1].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.fig.axes[col[0]*(resid_df.shape[1] + 1) + col[1]].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.map_upper(sns.scatterplot, data=flat_resids, size=0.75, alpha=.6)
    plt.close()
    return fg


def _prep_for_proba_pairs(labels, subset_predicts, select_params):
    assert isinstance(subset_predicts, (list, tuple))
    for m in subset_predicts:
        assert isinstance(m, dict)
        for v in m.values():
            assert isinstance(v, dict)
            for d in v.values():
                assert isinstance(d, (pd.Series, pd.DataFrame))
    marker_style = dict(
        color="tab:blue",
        linestyle=":",
        marker="o",
        #  markersize=15,
        markerfacecoloralt="tab:red",
    )
    size = (1 - labels.copy()).clip(lower=0.5) ** 4
    alpha = (1 - labels.copy()).clip(lower=0.1, upper=1.00)
    true_named = labels.copy()
    true_named.name = "True"
    data_list = list()
    for s_i, sers in enumerate(subset_predicts):
        # print(pd.concat(sers["predict_proba"]["test"]).iloc[:, 0], flush=True)
        if isinstance(sers[select_params["model_output"]]["test"], (list, tuple)):
            df = pd.concat(sers[select_params["model_output"]]["test"])
            if len(df.shape) > 1:
                print(df)
                df = df.iloc[:, 0]
        if isinstance(
            sers[select_params["model_output"]]["test"], (pd.Series, pd.DataFrame)
        ):
            df = sers[select_params["model_output"]]["test"]
            # df = pd.concat([s.iloc[:, 0] for s in sers["predict_proba"]["train"]])
            # df.columns = ["{}_{}".format(s_i, f_i) for f_i in np.arange(df.shape[1])]
            # print(df.head(), df.shape)
        else:
            try:
                print("Exception caught!\n\n\n")
                print(sers)
                df = pd.concat(
                    [
                        pd.concat(
                            [a[select_params["model_output"]]["train"] for a in sers]
                        )
                    ],
                    axis=1,
                )
            except TypeError:
                df = pd.concat(
                    [a for a in sers[select_params["model_output"]]["train"]]
                )
        df = scoring.correct_class_probs(y_true=labels, y_proba=df).squeeze()
        df.name = "Subset_{}".format(s_i)
        data_list.append(df.sort_index())
    data_list.append(true_named.sort_index())
    pred_df = pd.concat(data_list, axis=1)
    # , left_index=True, right_index=True, how="inner")
    # Rework this. Made to avoid circular import.
    # resid_df = _make_proba_residuals(pred_df, labels=labels.loc[pred_df.index])
    resid = dict()
    for col_a in np.arange(pred_df.shape[1]):
        for col_b in np.arange(col_a + 1, pred_df.shape[1]):
            resid[(col_a, col_b)] = pred_df.iloc[:, col_a] - pred_df.iloc[:, col_b]
    resid_df = pd.DataFrame.from_dict(resid)
    flat_resids = resid_df.copy()
    flat_resids.columns = resid_df.columns.map(
        dict(enumerate(pred_df.drop(columns="True").columns))
    ).to_flat_index()
    together_df = pd.concat([resid_df, pred_df.drop(columns="True")], axis=1)
    # flat_resids = flat_resids.merge(pred_df["True"], right_index=True, left_index=True)
    # fig = plt.figure(figsize=(8, 8), dpi=500)
    # pg = sns.PairGrid(data=together_df, x_vars=pred_df.drop(columns="True").columns.tolist(), y_vars=resid_df.columns.tolist(), diag_sharey=False, corner=True)
    # pg = pg.map_lower(sns.kdeplot, data=pred_df, hue="True", common_norm=False, levels=5, legend=False)
    # pg.map_lower(sns.kdeplot, data=flat_resids[flat_resids["True"] == 1].drop(columns="True"), sizes=0.1, alpha=0.01)
    # pg.map_lower(sns.kdeplot, data=flat_resids[flat_resids["True"] == 0].drop(columns="True"), size=2.0, alpha=1.0)
    # print(pred_df)
    pred_long = pred_df.melt(id_vars="True")
    print(pred_df)
    return pred_df, pred_long


def plot_proba_distances(
    feature_df, labels, model_subsets_dict, name_model_dict, path_dict=None
):
    triple_tup = list()
    for i, name in enumerate(model_subsets_dict.keys()):
        triple_tup = [
            (name, name_model_dict[name], subset) for subset in model_subsets_dict[name]
        ]
    model_dist_plots = scoring_metrics.cv_model_prediction_distance(
        feature_df, labels, triple_tup
    )
    if path_dict is not None:
        for i, (name, plot) in enumerate(model_dist_plots.items()):
            plot.savefig(
                "{}subset{}/{}_prediction_distance_plot.png".format(
                    path_dict["exp_dir"], i, name
                )
            )
    return model_dist_plots


def plot_models(
    feature_df,
    labels,
    select_params,
    estimator_name,
    estimator_list,
    best_features_list,
    save_dir,
    save_path,
    preds_list=None,
    probs_list=None,
    sample_weight_list=None,
):
    """

    Parameters
    ----------
    feature_df : pd.DataFrame
    labels: pd.Series
    select_params : dict
    estimator_name : str
    estimator_list : list of estimators
    best_features_list : list of lists or tuples
    save_dir : str
    save_path : str
    preds_list : list of pd.Series or None
    probs_list : list of pd.Series | pd.DataFrames or None
    sample_weight_list : list [pd.Series] | None
    """
    print(preds_list)
    for i, (estimator, best_features, weights) in enumerate(
        zip(estimator_list, best_features_list, sample_weight_list)
    ):
        submodel_name = "{}_{}".format(estimator_name, i)
        if preds_list is None or len(preds_list) == 0 or preds_list[i] is None:
            preds = None
        elif isinstance(preds_list[i], (pd.Series, pd.DataFrame)):
            preds = preds_list[i]
        else:
            preds = pd.concat(preds_list[i])
        if probs_list is None or len(probs_list) == 0 or len(probs_list[i]) == 0:
            probs = None
        elif isinstance(probs_list[i], (pd.Series, pd.DataFrame)):
            probs = probs_list[i]
        else:
            probs = pd.concat(probs_list[i])
        plots = plot_clf_model_displays(
            estimator=estimator,
            estimator_name=submodel_name,
            train_df=feature_df[list(best_features)],
            train_labels=labels,
            select_params=select_params,
            preds=preds,
            probs=probs,
            subset_dir=save_dir,
            sample_weight=weights,
        )
        plt.close()

    results_list, scores_plot = plot_model_scores(
        feature_df=feature_df,
        train_labels=labels,
        score_tups=select_params["score_tups"],
        estimator_list=estimator_list,
        subsets=best_features_list,
        cv=select_params["cv"],
    )
    # score_plot.figure.set(title="{}".format(model_name), ylabel="Score")
    scores_plot.savefig(save_path)
    plt.close()
    # pd.concat(results_list).to_csv("{}{}results_long-form.csv".format(path_dict["exp_dir"], estimator_name))
