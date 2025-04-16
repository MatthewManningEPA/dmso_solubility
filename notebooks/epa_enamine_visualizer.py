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

import samples
import scoring


# from model_selection_epa_multiclass import _make_proba_residuals


def main():
    pass


if __name__ == "__main__":
    main()


def plot_model_scores(
    feature_df,
    train_labels,
    score_tups,
    estimator,
    subsets,
    cv=None,
    sample_weight=None,
    **kwargs,
):
    assert len(subsets) > 0
    results_list = list()
    long_list = list()
    kwargs.update({"pos_label": 0})
    subset_df_list = list()
    for i, best_features in enumerate(subsets):
        # name_scorer_tups = dict((k, make_scorer(v)) for k, v in score_tups)
        score_dict = dict.fromkeys([name for name, f in score_tups])
        if len(best_features) == 0:
            continue
        else:
            print("Sample weights for model score plots:\n{}".format(sample_weight))
        if isinstance(sample_weight, dict):
            weights = list(sample_weight.values())[i]
        else:
            weights = sample_weight
        all_results, long_form, test_idx_tuple = scoring.cv_model_generalized(
            estimator=estimator,
            feature_df=feature_df[pd.Index(best_features)],
            labels=train_labels,
            cv=cv,
            return_train=True,
            score_list=score_tups,
            sample_weight=weights,
            randomize_classes="both",
            **kwargs,
        )
        long_df = long_form["predict_proba"]
        long_df.insert(loc=0, column="Subset", value=i)
        long_list.append(long_form["predict_proba"])
        results_list.append(all_results)
        print(all_results)
        print("Plot input columns:")
        print(long_list[0].columns)
        base_info_df = long_form[
            ["predict"]["Subset", "Labels", "Split", "CV_Fold"]
        ].drop_duplicates()
        group_cols = ["Subset", "Labels", "Split", "CV_Fold"]
        score_input_cols = copy.deepcopy(group_cols)
        score_input_cols.append("predict")
        score_input_cols.append("True")
        grouper = long_form[score_input_cols].groupby(
            group_cols, as_index=False, group_keys=False
        )
        for s_name, s_func in score_tups:
            for g_name, g_df in grouper:
                score_ser = g_df[group_cols].copy()
                long_score = scoring.score_long_form(
                    s_func, long_form["predict"], data_cols="predict", true_col="True"
                )
                score_ser[s_name] = long_score
                score_ser["Subset"] = i
                subset_df_list.append(score_ser)
    all_scores_long = pd.concat(subset_df_list)
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
    if display_labels is None:
        display_labels = [str(s) for s in train_labels.unique()]
    if is_regressor(estimator):
        cv = KFold(shuffle=True, random_state=0)
    else:
        cv = StratifiedKFold(shuffle=True, random_state=0)
    try:
        estimator = estimator.set_params({"n_jobs": 1})
    except:
        estimator = estimator
    if probs is None and preds is None:
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
                    subset_dir, estimator_name, select_params["score_func"]
                )
            )
    # if sample_weight is not None and isinstance(sample_weight, pd.Series):

    rcd = RocCurveDisplay.from_predictions(
        train_labels,
        probs,
        pos_label=0,
        name="DMSO Insolubles",
        plot_chance_level=True,
        sample_weight=sample_weight,
    )
    rcd.ax_.set(ylim=[0, 1.0])
    rcd.ax_.set(xlim=[0, 1.0])
    rcd.figure_.set_dpi(300)
    if subset_dir is not None:
        rcd.figure_.set_dpi(300)
        rcd.figure_.savefig("{}RocCurve_{}.png".format(subset_dir, estimator_name))
    det = DetCurveDisplay.from_predictions(
        train_labels,
        probs,
        pos_label=0,
        name="DMSO Insolubles",
        sample_weight=sample_weight,
    )
    # det.ax_.set(ylim=[0, 1.0])
    # det.ax_.set(xlim=[0, 1.0])
    det.figure_.set_dpi(300)
    # det.ax_.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    if subset_dir is not None:
        det.figure_.savefig("{}DetCurve_{}.png".format(subset_dir, estimator_name))
    if is_classifier(estimator):
        if preds is None:
            preds = cross_val_predict(
                estimator=clone(estimator),
                X=train_df,
                y=train_labels,
                cv=cv,
                n_jobs=-2,
                method="predict",
                params={"sample_weight": sample_weight},
            )
        cmd = ConfusionMatrixDisplay.from_predictions(
            y_true=train_labels,
            y_pred=preds,
            display_labels=display_labels,
            normalize="true",
            cmap="Blues",
        )
        cmd.figure_.set_dpi(300)
        if subset_dir is not None:
            cmd.figure_.savefig(
                "{}confusion_matrix_{}.png".format(subset_dir, estimator_name)
            )
    lcd = LearningCurveDisplay.from_estimator(
        estimator=clone(estimator),
        X=train_df,
        y=train_labels,
        train_sizes=np.linspace(0.05, 0.9, num=15),
        scoring=select_params["scorer"],
        n_jobs=-2,
        shuffle=True,
        random_state=0,
        score_name=select_params["score_name"],
    )
    lcd.figure_.set_dpi(300)
    lcd.ax_.set(ylim=[0, 1.0])
    if subset_dir is not None:
        lcd.figure_.savefig("{}lcd_{}.png".format(subset_dir, estimator_name))
    return rcd, det, lcd, cmd


def _plot_proba_pairs(labels, n, subset_predicts, select_params):
    marker_style = dict(
        color="tab:blue",
        linestyle=":",
        marker="o",
        #  markersize=15,
        markerfacecoloralt="tab:red",
    )
    size = (1 - labels.copy()).clip(lower=0.5) ** 4
    alpha = (1 - labels.copy()).clip(lower=0.1, upper=1.00)
    pred_df = labels.copy()
    pred_df.name = "True"
    data_list = list()
    for s_i, sers in enumerate(subset_predicts[n]):
        # print(pd.concat(sers["predict_proba"]["test"]).iloc[:, 0], flush=True)
        try:
            df = pd.concat(sers[select_params["score_func"]]["test"])
            if len(df.shape) > 1:
                print(df)
                df = df.iloc[:, 0]
            # df = pd.concat([s.iloc[:, 0] for s in sers["predict_proba"]["train"]])
            # df.columns = ["{}_{}".format(s_i, f_i) for f_i in np.arange(df.shape[1])]
            # print(df.head(), df.shape)
        except TypeError:
            try:
                print("Exception caught!\n\n\n")
                print(sers)
                df = pd.concat(
                    [
                        pd.concat(
                            [a[select_params["score_func"]]["train"] for a in sers]
                        )
                    ],
                    axis=1,
                )
            except TypeError:
                df = pd.concat([a for a in sers[select_params["score_func"]]["train"]])
        df.name = "Subset_{}".format(s_i)
        data_list.append(df.sort_index())
    data_list.append(pred_df.sort_index())
    pred_df = pd.concat(
        data_list, axis=1
    )  # , left_index=True, right_index=True, how="inner")
    ### Rework this. Made to avoid circular import.
    # resid_df = _make_proba_residuals(pred_df, labels=labels.loc[pred_df.index])
    resid = dict()
    for col_a in np.arange(pred_df.shape[1]):
        for col_b in np.arange(col_a + 1, pred_df.shape[1]):
            resid[(col_a, col_b)] = pred_df.iloc[:, col_a] - pred_df.iloc[:, col_b]
    resid_df = pd.DataFrame.from_dict(resid)

    # pprint.pp(resid_df)
    # full_kde = True
    # full_pred = pred_df.copy()
    sample_fraction = 0.5
    # pred_df = pred_df.groupby(["True"], observed=True).sample(frac=sample_fraction, replace=True)
    # print(pred_df["True"])
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
    # print(pred_long)
    fg = sns.FacetGrid(
        pred_long, row="variable", col="variable", margin_titles=True, sharey=False
    )
    for i, j in itertools.combinations(pred_df.drop(columns="True").columns, r=2):
        for dg in pred_df["True"].unique():
            data_group = pred_df.loc[pred_df["True"] == dg]
            # print(col[0]*(pred_df.shape[1] - 1) + col[1] - 1)
            ax1 = sns.scatterplot(
                x=data_group[i],
                y=data_group[i] - data_group[j],
                ax=fg.facet_axis(
                    data_group.columns.get_loc(i), data_group.columns.get_loc(j)
                ),
                size=0.01,
                legend=False,
            )
            ax1.set(xlim=(0.0, 1.0))
            ax1.set(ylim=(-1.0, 1.0))
            ax2 = sns.scatterplot(
                x=data_group[i],
                y=data_group[j],
                ax=fg.facet_axis(
                    data_group.columns.get_loc(j), data_group.columns.get_loc(i)
                ),
                size=0.01,
                legend=False,
            )
            ax2.set(xlim=(0.0, 1.0))
            ax2.set(ylim=(0.0, 1.0))
    for i in pred_df.drop(columns="True").columns.tolist()[::-1]:
        sns.histplot(
            pred_long[pred_long["variable"] == i].drop(columns="variable"),
            x="value",
            hue="True",
            stat="density",
            common_norm=False,
            common_bins=True,
            ax=fg.facet_axis(pred_df.columns.get_loc(i), pred_df.columns.get_loc(i)),
            legend=False,
        )
        # fg.fig.axes[col[0]+ col[1] - 1].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.fig.axes[col[0]*(resid_df.shape[1] + 1) + col[1]].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.map_upper(sns.scatterplot, data=flat_resids, size=0.75, alpha=.6)
    plt.close()
    return fg


def plot_proba_distances(
    feature_df, labels, model_subsets_dict, name_model_dict, path_dict=None
):
    triple_tup = list()
    for i, name in enumerate(model_subsets_dict.keys()):
        triple_tup = [
            (name, name_model_dict[name], subset) for subset in model_subsets_dict[name]
        ]
    model_dist_plots = samples.cv_model_prediction_distance(
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
