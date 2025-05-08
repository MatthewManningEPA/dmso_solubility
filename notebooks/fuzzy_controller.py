import copy
import logging
import os

import numpy as np
import pandas as pd
from sklearn import linear_model

import padel_categorization
from epa_enamine_visualizer import _plot_proba_pairs, plot_models
from FuzzyAnnealer import FuzzyAnnealer
from FuzzyFileHandler import FuzzyFileHandler


def control_loop():
    pass


def main():
    pass


if __name__ == "__main__":
    main()


def select_subsets_from_model(
    feature_df,
    labels,
    select_params,
    estimator,
    estimator_name,
    model_dir,
    path_dict,
    prior_probs=None,
):
    """
    Selects best scoring feature subsets from specified scikit-learn estimators.
    Parameters for feature selection are specified in select_params.

    Parameters
    ----------
    path_dict
    feature_df: DataFrame
    Feature descriptors
    labels: Series
    Target labels
    Max number of feature subsets to return
    Dictionary with name and unfitted estimator
    Path of directory to save files
    select_params: dict
    Describes parameters for feature selection function
    model_dir : str
    estimator_name : str
    estimator : estimator
    prior_probs : list[pd.DataFrame]

    Returns
    -------
    model_scores_dict: dict[str, list[float]]
    CV scores of each estimator, refit using chosen subsets.
    model_subsets_dict: dict[str, list[str]]
    List of names of chosen features for each estimator.
    subset_predicts: dict[str, Iterable[pd.Series | pd.DataFrame]]
    Outputs from each model using its best-scoring feature subset.
    name_weight_dict: dict[str, list[pd.Series]]
    Dictionary of lists of sample weights used for each subset selection and scoring.
    """

    # Set default weighting values.
    assert (
        select_params["base_weight"] is not None
        and select_params["base_weight"].sum() != 0
    )
    """
    if (
        select_params["sample_weight"] is None
        or select_params["sample_weight"].sum() == 0
    ):
        select_params["sample_weight"] = pd.Series(
            np.ones_like(labels.to_numpy()),
            dtype=np.float32,
            index=labels.index,
            name="sample_weight",
        )
        """
    # Start subset selection loop

    weights_list = list()
    subsets_list = list()
    score_list = list()
    predicts_list = list()
    if prior_probs is None:
        prior_probs = list()
        subset_paths = dict()
        # TODO: Compare i to len(blah_blah_dict[estimator_name]) to see if a fit is needed.

    """
    best_probs = get_best_probs(labels, prior_probs)
    if best_probs.sum() == 0:
        print(best_probs.head())
        print("\n\n\nBest probs sums to zero!\n\n\n")
        best_probs = pd.Series(
            1 / (1 + labels.nunique()), index=labels.index
        )
    subset_weights = weights_from_predicts(
        y_true=labels,
        y_predict=predicts_list,
        predict_model=estimator,
        select_params=select_params,
        score_func=select_params["model_output"],
    )
    # select_params["sample_weights"] = subset_weights
    label_corr, cross_corr = get_weighted_correlations(
        feature_df,
        labels,
        select_params,
        subset_dir,
        weights=subset_weights,
    )
    """
    # assert np.shape(label_corr)[0] == cross_corr.shape[1]
    # label_corr.to_csv(subset_paths["weighted_label"])
    # cross_corr.to_csv(subset_paths["weighted_cross"])
    """
    if isinstance(prior_probs, list) and len(prior_probs) > 1:

        prob_list = list()
        one_hot, one_hot_normed = samples.one_hot_conversion(labels)
        if select_params["sample_weight"].shape[0] != one_hot.shape[0]:
            print("OneHot")
            print(one_hot)
            print("Sample weight")
            print(select_params["sample_weight"])
            # raise ValueError
        for prob_out in prior_probs:
            if len(prob_out.shape) == 2:
                prob_list.append(
                    select_params["sample_weight"].mul(one_hot).sum(axis=1)
                )
        for a, b in itertools.combinations(np.arange(len(prob_list)), r=2):
            sq_diff = (prob_list[a] - prob_list[b]) ** 2
    """
    ###################
    # Run iteration of feature selection on model architecture.
    # cv_predicts, cv_scores, best_features, weights_list =
    create_submodels(
        estimator=estimator,
        feature_df=feature_df,
        labels=labels,
        estimator_name=estimator_name,
        select_params=select_params,
        path_dict=None,
        save_dir=model_dir,
    )
    # return weights_list, best_features, cv_scores, cv_predicts


def create_submodels(
    estimator,
    feature_df,
    labels,
    estimator_name,
    select_params,
    path_dict,
    save_dir,
    **kwargs
):

    logger = logging.Logger("controller")

    """
    Returns a set of features that maximizes the score function of the estimator. Iteratively adds features, sampled based on label correlation and correlation with other features.
    Uses VIF score to reduce multicollinearity amongst current feature set and SFS to eliminate uninformative or confounding features.

    Parameters
    ----------
    file_handler
    feature_df: pd.DataFrame
    labels: pd.Series
    estimator: pd.BaseEstimator
    estimator_name: str
    select_params: dict
    save_dir: str
    prior_probs

    Returns
    -------
    results_dict : dict[str, dict[str, list[pd.Series]]]
        Contains sample-wise results for each CV fold
        {load_cv_results: {split_name: list[pd.Series] | pd.Series}}
    cv_scores : dict[str, dict[str, list[pd.Series]]]
    best_features : list, Highest scoring set of features
    """
    # if "n_jobs" in model.get_params():
    #    model.set_params(**{"n_jobs": 1})

    ffh_list, prior_probs = list(), list()
    ###################
    # Retrieve previous modeling runs.
    """
    if os.path.isfile(subset_paths["best_features"]):
        cv_results = ffh.load_cv_results(i)
        predicts_list.append(cv_results["original"])
        subset_weights = weights_from_predicts(
            y_true=labels,
            y_predict=predicts_list,
            predict_model=estimator,
            select_params=select_params,
            score_func=select_params["model_output"],
        )
        weights_list.append(subset_weights)
    """
    selection_models = {
        "predict": estimator,
        "permutation": estimator,
        "importance": estimator,
        "vif": linear_model.LinearRegression(),
    }
    fuzzy_list = list()
    epoch_dir = "{}epoch_{}/".format(save_dir, 0)
    os.makedirs(epoch_dir, exist_ok=True)
    for _ in np.arange(select_params["n_subsets"]):
        fuzz = FuzzyAnnealer(
            params=select_params,
            models=selection_models,
            save_dir=save_dir,
        )
        fuzz.fit(feature_df=feature_df, labels=labels)
        fuzzy_list.append(fuzz)
        ffh = FuzzyFileHandler(
            working_dir=save_dir,
            estimator_name=estimator_name,
            estimator_type="clf",
            primary_output_type="predict_proba",
        )
        ffh.working_dir = epoch_dir
        ffh.save_model_results(
            save_dir=ffh.working_dir,
            best_features=fuzz.best_subset,
            label_corr=fuzz.label_corr,
            pair_corr=fuzz.cross_corr,
        )
        ffh_list.append(ffh)
        prior_probs.append(fuzz.best_preds)
    for i_epoch in np.arange(select_params["n_epochs"]):
        print("Running {} epoch".format(i_epoch))
        epoch_dir = "{}epoch_{}/".format(save_dir, i_epoch + 1)
        os.makedirs(epoch_dir, exist_ok=True)
        updated_prior_probs = list()
        for i_subset, (ffh, submodel) in enumerate(zip(ffh_list, fuzzy_list)):
            ffh.working_dir = epoch_dir
            subset_dir = ffh.get_subset_dir(i_subset)
            os.makedirs(subset_dir, exist_ok=True)
            submodel.save_dir = subset_dir
            submodel_others = copy.deepcopy(prior_probs)
            submodel_others.pop(i_subset)
            submodel.partial_fit(
                feature_df=feature_df, labels=labels, other_probs=submodel_others
            )
            ffh.save_model_results(
                i=i_subset,
                best_features=report_best_features(submodel.best_subset),
                label_corr=submodel.label_corr,
                pair_corr=submodel.cross_corr,
            )
            updated_prior_probs.append(submodel.best_preds)
        prior_probs = copy.deepcopy(updated_prior_probs)

    return fuzzy_list, ffh_list
    # TODO: Fix this plotting function.
    # name_distplot_dict = plot_proba_distances(feature_df, labels, model_subsets_dict, name_model_dict)


def save_plot_submodel_results(
    fuzzy_list, ffh_list, feature_df, labels, estimator_name, select_params, save_dir
):
    predicts_list, subsets_list, score_list, weights_list = (
        list(),
        list(),
        list(),
        list(),
    )
    estimator_list = list()
    score_df_list = list()
    pred_concat_list, prob_concat_list = list(), list()
    cv_models_list = list()
    for i_submodel, (ffh, submodel) in enumerate(zip(ffh_list, fuzzy_list)):
        ffh.working_dir = "{}final/".format(save_dir)
        subset_scores = submodel.subset_scores
        score_list.append(submodel.best_score)
        weights_list.append(submodel.sample_weight)
        best_features = submodel.best_subset
        subsets_list.append(best_features)
        # best_estimator = submodel.best_model
        # if best_estimator is None:
        best_estimator = submodel.freeze_best()
        estimator_list.append(best_estimator)
        predicts_list.append(
            {select_params["model_output"]: {"test": submodel.best_preds}}
        )
        model_results = submodel.freeze_cv_best(model="best")  # cv=select_params["cv"])
        print(list(zip(model_results)))
        print([len(m) for m in model_results])
        print(list(zip(model_results)))
        frozen, y_list, y_pred, y_prob = list(zip(*model_results))
        cv_models_list.append(frozen)
        pred_concat_list.append(pd.concat(y_pred))
        prob_concat_list.append(pd.concat(y_prob))
        ffh.save_model_results(
            save_dir=ffh.working_dir,
            i=i_submodel,
            best_features=best_features,
            frozen_model=best_estimator,
        )
        """
        cv_results, long_form_results, test_idx_splits = scoring.cv_model_generalized(
            estimator=estimator,
            feature_df=feature_df[list(best_features)],
            labels=labels,
            cv=select_params["cv"],
            return_train=True,
            sample_weight=None,
            random_state=None,
        )
        # for k, long_df in long_form_results.items():
        #    long_df.insert(loc=0, column="Subset", value=model_name)
        filtered_results = ffh.save_cv_results(
            i=i_submodel,
            cv_predicts=cv_results,
            output_type=("predict", "predict_proba"),
            split_type=("test",),
            label_names=("original", "randomized"),
        )
        score_tuples = [(select_params["score_name"], select_params["scoring"])]
        group_cols = ["CV_Fold", "Split", "Labels"]
        predict_df = long_form_results["predict"]
        predict_groups = predict_df.groupby(
            group_cols, as_index=False, group_keys=False
        )
        # score_dict = dict([(k, v) for k, v in (s_name, s_func)])
        score_dict = dict()
        for score_name, score_func in score_tuples:
            scores_list = list()
            for g in predict_groups:
                long_df = g[1].drop(columns=group_cols)
                score_ser = pd.Series(g[0], index=group_cols)
                long_score = scoring.score_long_form(
                    score_func, x=long_df, true_col="True", remove_cols="INCHI_KEY"
                )
                score_ser["Metric"] = score_name
                score_ser["score"] = long_score
                scores_list.append(score_ser)
            score_dict[score_name] = pd.concat(scores_list)
        score_df = pd.concat(score_dict, ignore_index=True)
        score_df_list.append(score_df)
        print([filtered_results.keys()])
        priors = filtered_results["original"][select_params["model_output"]]["test"]
        if isinstance(priors, (pd.Series, pd.DataFrame)):
            priors_df = priors
        else:
            priors_df = pd.concat(priors)
        prior_probs.append(priors_df)
        # score_list.append(cv_scores)
        """
    plot_models(
        feature_df=feature_df,
        labels=labels,
        select_params=select_params,
        estimator_name=estimator_name,
        estimator_list=estimator_list,
        preds_list=pred_concat_list,
        probs_list=prob_concat_list,
        best_features_list=subsets_list,
        save_dir="{}final/".format(save_dir),
        save_path="{}subset_scores.png".format(save_dir),
        sample_weight_list=weights_list,
    )
    pg = _plot_proba_pairs(labels, predicts_list, select_params)
    pg.savefig("{}pair_proba_grid.png".format(save_dir), dpi=300)
    return subsets_list, pred_concat_list, prob_concat_list, score_list, weights_list


"""
def select_feature_subset(
    train_df,
    labels,
    target_corr,
    cross_corr,
    select_params,
    initial_subset=None,
    save_dir=None,
    selection_models=None,
    prior_best_probs=None,
):

    Parameters
    ----------
    train_df : pd.DataFrame
    labels : pd.Series
    target_corr : pd.Series
    cross_corr : pd.DataFrame
    select_params : dict
    initial_subset
    save_dir : str
    selection_models : dict
    prior_best_probs : pd.DataFrame

    Returns
    -------

    clean_up, selection_state, sqcc_df = selection_setup(
        train_df,
        labels,
        initial_subset,
        target_corr,
        cross_corr,
        select_params,
        selection_models,
        prior_best_probs,
        save_dir,
    )
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        print("\n\nSelection step {} out of {}.".format(i, select_params["max_trials"]))
        if i > 0:
            selection_state["temp"] = math_tools.acf_temp(
                [
                    selection_state["subset_scores"][tuple(sorted(s))]
                    for s in selection_state["chosen_subsets"]
                ]
            )
        print("New temperature: {}".format(selection_state["temp"]))
        # print("Feature_df shape: {}".format(train_df.shape))
        # print(len(selection_state["current_features"]))
        # TODO: Does this check need to be here? Especially with the new weighting for reused features
        maxed_out = (
            len(selection_state["current_features"])
            >= select_params["max_features_out"]
        )
        above_min = (
            len(selection_state["current_features"])
            >= select_params["features_min_sfs"]
        )
        if False and (maxed_out or (above_min and clean_up)):
            # or train_df.shape[1]
            # - len([selection_state["rejected_features"].keys()])
            # - len(selection_state["current_features"])
            # < select_params["n_vif_choices"]
            print(selection_state["current_features"])
            # Keep eliminating until sequential elimination is within range of going under score_exceeded limit.
            while _over_sfs_thresh(select_params, selection_state):
                print(current_score(selection_state), selection_state["best_score_adj"])
                original_size = len(selection_state["current_features"])
                selection_state, subset_scores, score_improve = sequential_elimination(
                    train_df=train_df,
                    labels=labels,
                    select_params=select_params,
                    selection_state=selection_state,
                    selection_models=selection_models,
                    clean_up=True,
                    save_dir=save_dir,
                    randomize=True,
                )
                if original_size == len(selection_state["current_features"]):
                    clean_up = False
                    break
            continue
        else:
            clean_up = False
        try:
            sqcc_df_choices = sqcc_df.drop(
                index=selection_state["rejected_features"]
            ).drop(columns=selection_state["rejected_features"])
        except KeyError:
            print("Key Error in sqcc_df")
            raise KeyError
            sqcc_df_choices = (
                sqcc_df[train_df.columns.intersection(sqcc_df.index)]
                .loc[train_df.columns.intersection(sqcc_df.index)]
                .drop(index=selection_state["rejected_features"])
                .drop(columns=selection_state["rejected_features"])
            )        

        new_feat, selection_state = choose_next_feature(
            feature_df=train_df,
            feature_list=selection_state["current_features"],
            target_corr=target_corr,
            sq_xcorr=sqcc_df,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
        )
        if new_feat is None:
            break
        else:
            selection_state, subset_scores, score_improve = score_subset(
                feature_df=train_df,
                labels=labels,
                selection_models=selection_models,
                selection_state=selection_state,
                select_params=select_params,
                save_dir=save_dir,
                record_results=True,
            )
        if score_improve is not None:
            subset_metric = score_improve
        else:
            subset_metric = subset_scores
        # Check if score has dropped too much.
        exceeded, selection_state = score_drop_exceeded(
            subset_metric,
            selection_params=select_params,
            selection_state=selection_state,
            set_size=len(selection_state["current_features"]),
            replace_current=False,
        )
        if (
            exceeded
            and len(selection_state["current_features"])
            > select_params["features_min_sfs"]
        ):
            while exceeded:
                if len(selection_state["current_features"]) <= select_params[
                    "features_min_sfs"
                ] or random.random() > math_tools.zwangzig(
                    selection_state["subset_scores"],
                    current_score(selection_state["current_features"]),
                    select_params["lang_lambda"],
                    math_tools.size_factor(
                        len(selection_state["current_features"]), select_params
                    ),
                ):
                    selection_state["current_features"] = copy.deepcopy(
                        selection_state["best_subset"]
                    )
                    break
                selection_state, subset_scores, score_improve = sequential_elimination(
                    train_df,
                    labels,
                    select_params,
                    selection_state,
                    selection_models,
                    clean_up=False,
                    save_dir=save_dir,
                )
                exceeded, selection_state = score_drop_exceeded(
                    score_improve,
                    selection_params=select_params,
                    selection_state=selection_state,
                    set_size=len(selection_state["current_features"]) - 1,
                )
            continue
        # Variance Inflation Factor: VIF check implemented in "new feature" selection function.
        if True and (
            len(selection_state["current_features"])
            >= select_params["features_min_vif"]
        ):
            selection_state = _vif_elimination(
                train_df=train_df,
                best_corrs=target_corr,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_state=selection_state,
            )
        # Feature Importance Elimination
        if False and select_params["features_min_perm"] < len(
            selection_state["current_features"]
        ):
            if select_params["importance"] == "permutate":
                selection_state = _permutation_removal(
                    train_df=train_df,
                    labels=labels,
                    estimator=clone(selection_models["permutation"]),
                    select_params=select_params,
                    selection_state=selection_state,
                )
            elif (
                select_params["importance"] != "permutate"
                and select_params["importance"] is not False
            ):
                # rfe = RFECV(estimator=grove_model, min_features_to_select=len(selection_state["current_features"])-1, n_jobs=-1).set_output(transform="pandas")
                rfe = SelectFromModel(
                    estimator=clone(selection_models["importance"]),
                    max_features=len(selection_state["current_features"]) - 1,
                ).set_output(transform="pandas")
                rfe.fit(train_df[selection_state["current_features"]], labels)
                if True:  # any(~rfe.support_):
                    dropped = [
                        c
                        for c in selection_state["current_features"]
                        if c
                        not in rfe.get_feature_names_out(
                            selection_state["current_features"]
                        )
                    ][0]
                    # dropped = train_df[selection_state["current_features"]].columns[~rfe.support_][0]
                    print("Dropped: {}".format(dropped))
                    selection_state["rejected_features"].update((dropped, "importance"))
                    selection_state["current_features"].remove(dropped)
                    continue
                else:
                    # _get_fails(selection_state)
                    break
        while (
            len(selection_state["current_features"])
            >= select_params["features_min_sfs"]
        ) or clean_up:
            if (
                len(selection_state["current_features"])
                >= select_params["max_features_out"]
            ):
                clean_up = True
            # DEBUG
            if (
                len(selection_state["current_features"])
                <= select_params["min_features_out"]
            ):
                break
            n_features_in = copy.deepcopy(len(selection_state["current_features"]))
            selection_state, subset_scores, score_improve = sequential_elimination(
                train_df,
                labels,
                select_params=select_params,
                selection_state=selection_state,
                selection_models=selection_models,
                clean_up=clean_up,
                save_dir=save_dir,
            )
            # SFS fails to eliminate a feature.
            # DEBUG
            print(
                "Features in: {}. Features out: {}".format(
                    n_features_in, len(selection_state["current_features"])
                )
            )
            too_much, selection_state = score_drop_exceeded(
                new_scores=score_improve,
                selection_params=select_params,
                selection_state=selection_state,
                set_size=len(selection_state["current_features"]),
            )
            if too_much or n_features_in == len(selection_state["current_features"]):
                clean_up = False
                break
    print(
        "Best adjusted score of {} with feature set: {}".format(
            selection_state["best_score_adj"],
            selection_state["best_subset"],
        )
    )
    # TODO: Check if this is even possible to fail.
    if len(selection_state["best_subset"]) > 0:
        best_fit_model = FrozenEstimator(
            selection_models["predict"].fit(
                X=train_df[list(selection_state["best_subset"])], y=labels
            )
        )
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(best_fit_model, f)
    else:
        print(selection_state["best_subset"])
        raise RuntimeError

    print("Rejects: {}".format(selection_state["rejected_features"]))
    pd.Series(selection_state["rejected_features"], name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )

    return (
        selection_models["predict"],
        selection_state["subset_scores"],
        selection_state["rejected_features"],
        selection_state["best_subset"],
    )
    """


def report_best_features(best_features):
    print("Best features!")
    short_to_long = padel_categorization.padel_convert_length()
    best_features_long = short_to_long[
        [f for f in best_features if f in short_to_long.index]
    ]
    if len(best_features_long) > 0:
        missing_long = [f for f in best_features if f not in short_to_long.index]
        best_features_long = pd.concat(
            [best_features_long, pd.Series(missing_long, index=missing_long)]
        )
    else:
        best_features_long = pd.Series(best_features_long, index=best_features)
    print("\n".join(best_features_long.tolist()))
    return best_features_long
