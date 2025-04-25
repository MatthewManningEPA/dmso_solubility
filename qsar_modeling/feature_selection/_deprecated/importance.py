import copy
import logging
import os
import pickle
import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import check_X_y as checker

pjobs = -1

logger = logging.getLogger(__name__)


# TODO: Chose best predictors among multiple trees. Use CV to avoid overfitting.
def extra_trees_gini_importances(
    data_df, labels, sample_wts, n_feats_out, save_dir, num_bins=50, save_model=False
):
    # Selects highest performing features using the Gini importance from an ensemble of extremely random decision trees.

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # print('Lowest cardinality features above variance cut:\n')

    low_cardinality = data_df[data_df.columns[data_df.nunique(axis=0) <= num_bins]]
    high_cardinality = data_df.drop(columns=low_cardinality.columns)
    binned_feats = KBinsDiscretizer(
        n_bins=num_bins, encode="ordinal", strategy="kmeans"
    ).fit_transform(high_cardinality)
    var_bin_feats = pd.concat([binned_feats, low_cardinality], axis=1)
    num_trees = int(np.ceil(var_bin_feats.shape[1] * np.log(var_bin_feats.shape[1])))
    xtclf = ExtraTreesClassifier(
        n_estimators=num_trees, max_features=2, n_jobs=-1, verbose=1, random_state=0
    )
    train_X, train_y = checker(var_bin_feats, labels)
    xtclf.fit(X=train_X, y=train_y, sample_weight=sample_wts.loc[binned_feats.index])
    feature_importance_normalized = pd.Series(
        data=np.mean([tree.feature_importances_ for tree in xtclf.estimators_], axis=0),
        index=var_bin_feats.columns,
    ).sort_values(ascending=False, kind="stable")
    important_feats = feature_importance_normalized.index
    with open("{}extra_trees_feature_importances_std.pkl".format(save_dir), "w+b") as f:
        pickle.dump(important_feats, f)
    if save_model:
        with open("{}extra_trees_final_model.pkl".format(save_dir), "w+b") as f:
            pickle.dump(xtclf, f)
    feats_df = data_df[important_feats[:n_feats_out]]
    logger.info("Extra trees features:")
    logger.info(pprint.pformat(important_feats, compact=True))
    return feats_df


def dummy_score_cv(
    feature_df,
    labels,
    importance_model,
    feature,
    score_func,
    cv=None,
    use_noninformative=True,
    n_jobs=-2,
    sample_kde=True,
    **fit_kwargs
):
    # Calcalates the score of model if one of the features is replaced with a randomly valued feature. Used to benchmark model results.
    dummy_df = feature_df.copy(deep=True)
    if use_noninformative:
        # if sample_kde:
        #    if KernelDensity(bandwith='silverman', kernel='epanechnikov', metric='manhattan', )
        dummy_data = np.random.choice(
            dummy_df[feature].squeeze(), size=dummy_df.shape[0]
        )
        dummy_df[feature + "_dummy"] = dummy_data
        input_df = dummy_df.drop(columns=feature)
    else:
        input_df = dummy_df.drop(columns=feature)
    cv_scores = cross_val_score(
        importance_model,
        input_df,
        labels,
        scoring=score_func,
        cv=cv,
        n_jobs=n_jobs,
        params=fit_kwargs,
        error_score="raise",
    )
    return cv_scores


def dummy_score_elimination(
    feature_df,
    labels,
    estimator,
    min_feats,
    step_size=1,
    cv=5,
    subset=None,
    score_func=None,
    use_noninformative=True,
    sample_kde=True,
    verbose=0,
    n_jobs=-2,
    importance="auto",
    **fit_kwargs
):
    dummy_df = feature_df.copy(deep=True)
    drop_list = list()
    if subset is None:
        subset = dummy_df.columns
    subset = list(subset)
    if type(cv) is int:
        cv = StratifiedKFold(n_splits=cv)
    mean_scores = None
    while dummy_df.shape[1] > min_feats:
        score_dict = dict()
        test_splits = list()
        for i, (dev, eva) in enumerate(cv.split(dummy_df, y=labels)):
            # print("CV split counts: {}".format(labels.iloc[eva].value_counts()))
            test_splits.append((dev, eva))
        # test_splits_ser = pd.concat([pd.Series(data=np.ones(shape=s.shape) * i, index=s) for i, s in enumerate(test_splits)], axis="index")
        # cv = PredefinedSplit(test_fold=test_splits_ser)
        feat_score = cross_val_score(
            estimator,
            X=dummy_df,
            y=labels,
            scoring=score_func,
            cv=test_splits,
            n_jobs=n_jobs,
            params=fit_kwargs,
            error_score="raise",
        )
        print("All feature score: {}".format(feat_score))
        for feat in subset:
            if type(feat) is list():
                feat = feat[0]
            if feat not in dummy_df.columns:
                print("Feature not in columns! {}".format(feat))
                continue
            dummy_scores = dummy_score_cv(
                dummy_df,
                labels,
                estimator,
                feat,
                score_func,
                cv=cv,
                use_noninformative=use_noninformative,
                n_jobs=n_jobs,
                sample_kde=sample_kde,
                **fit_kwargs
            )
            score_dict[feat] = [a - b for a, b in zip(feat_score, dummy_scores)]
        mean_scores = (
            pd.DataFrame.from_dict(score_dict, orient="index")
            .mean(axis=1)
            .sort_values()
        )
        drop_size = max(1, min(step_size, dummy_df.shape[1] - min_feats))
        dropped_feats = mean_scores.index[:drop_size].to_list()
        drop_list.extend(dropped_feats)
        pprint.pp(mean_scores.iloc[: max(5, drop_size * 2)])
        dummy_df.drop(columns=dropped_feats, inplace=True)
        [subset.pop(subset.index(f)) for f in dropped_feats]
    return mean_scores


def _get_importance_model(model, target_type, **model_kwargs):
    # Helper function to pick which ensemble method to use in subsequent importance selection task.
    if model == "extra_trees_gini_importances" and "reg" in target_type:
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor
    elif model == "random_forest" and "reg" in target_type:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor
    elif model == "extra_trees_gini_importances" and "class" in target_type:
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier
    elif model == "random_forest" and "reg" in target_type:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier
    elif "balance" in model:
        from imblearn.ensemble import BalancedRandomForestClassifier

        model = BalancedRandomForestClassifier()
    return model


def get_feature_gini_importances(
    target_type,
    feature_df,
    labels,
    model,
    feature_subset,
    sample_weights,
    **model_kwargs
):
    # Gets importance of every feature in a feature set by random permutation of feature values and comparing the resulting model scores.
    importance_dict = dict()
    if feature_subset is None:
        feature_subset = copy.deepcopy(feature_df.columns)
    if type(model) is str:
        model = _get_importance_model(model, target_type=target_type, **model_kwargs)
    for feature in feature_subset:
        importance_dict[feature] = get_feature_gini_importances(
            feature_df, labels, model, feature, sample_weights=sample_weights
        )
    return importance_dict


def get_regression_importances(
    feature_df,
    labels,
    model="extra_trees_gini_importances",
    feature_subset=None,
    sample_weights=None,
    **model_kwargs
):
    # Helper function to get feature importance for regression models.
    return get_feature_gini_importances(
        "regression",
        feature_df,
        labels,
        model=model,
        feature_subset=feature_subset,
        samples_weights=sample_weights,
        **model_kwargs
    )


def get_classifier_importances(
    feature_df,
    labels,
    model="extra_trees_gini_importances",
    feature_subset=None,
    sample_weights=None,
    **model_kwargs
):
    # Helper function to get feature importances for classification tasks.
    return get_feature_gini_importances(
        "classifier",
        feature_df,
        labels,
        model=model,
        feature_subset=feature_subset,
        samples_weights=sample_weights,
        **model_kwargs
    )


def brute_force_selection(
    estimator,
    feature_df,
    labels,
    n_features_out,
    save_path=None,
    est_rfe_kwargs_list=({}, {}),
):
    # Selects features by buliding models from remaining features (with one withheld) and comparing the results of each feature and dropping the worst performing feature.
    print("Brute force feature selection started.")
    if estimator == "rfc" or "forest" in estimator:
        clf = RandomForestClassifier(random_state=0, oob_score=True, n_jobs=-2)
    elif estimator == "lr" or "logistic" in estimator:
        clf = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=-2,
            random_state=0,
            cv=5,
            l1_ratios=[0.25],
        )
    else:
        clf = estimator(**est_rfe_kwargs_list[0])
    ranked_features, rfe_model = brute_force_importance_rf_clf(
        feature_df,
        labels,
        clf=clf,
        n_features_out=n_features_out,
        **est_rfe_kwargs_list[1]
    )
    if save_path is not None:
        ranked_features.to_csv(
            save_path, index_label="Features"
        )  # , float_format='%.4f')
    selected_feature_df = rfe_model.transform(feature_df)
    return ranked_features, selected_feature_df


def brute_force_importance_rf_clf(
    feature_df, labels, clf, n_features_out, step_size=1, **fit_kwargs
):
    # Helper function for brute force feature selection.
    eliminator = RFE(estimator=clf, n_features_to_select=n_features_out, step=step_size)
    brute_features_rankings = pd.Series(
        eliminator.ranking_, index=feature_df.columns.tolist()
    ).sort_values()
    return brute_features_rankings, eliminator
