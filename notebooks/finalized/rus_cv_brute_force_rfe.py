import datetime
import itertools
import os
import pickle
import pprint

import multicollinear
import numpy as np
import pandas as pd
import qsar_modeling.feature_selection.multicollinear
from imblearn.under_sampling import RandomUnderSampler
from sklearn._config import set_config as sk_config
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import *
from sklearn.utils import check_X_y

from data import feature_name_lists
from data_cleaning import stepwise_duplicate
from dmso_utils.data_tools import (
    get_interpretable,
    load_metadata,
    load_training_data,
)

# from utils import cv_tools

"""
This script uses scikit-learn's RecursiveFeatureElimination (with RandomForest feature_importance_) to iteratively
remove features in the descriptor set of PaDeL descriptors for DMSO solubilty data. The purpose of this is to reduce
the dimensionality of the training data to make results more accurate and interpretable.
"""

pd.set_option("display.precision", 4)
pd.set_option("format.precision", 4)
np.set_printoptions(formatter={"float": "{: 0.4f}".format})
sk_config(transform_output="pandas", display="text", print_changed_only=True)

start = datetime.datetime.now()


def load_sample_keys(filepath):
    sample_list = list()
    with open(filepath, "r", encoding="utf-8") as f:
        [sample_list.extend(s.split(",")) for s in f.readlines()]
        sample_list = [s.partition("\\")[0] for s in sample_list if s.endswith("-N")]
        # csv.reader(f)
    return sample_list


def brute_force_importance_rf_clf(
    feature_df,
    labels,
    clf,
    n_feats_out,
    step_size=1,
    n_jobs=-2,
    verbose=0,
    **model_kwargs
):
    eliminator = RFE(
        estimator=clf, n_features_to_select=n_feats_out, step=step_size, verbose=verbose
    ).fit(feature_df, y=labels)
    brute_features = pd.Series(
        eliminator.ranking_, index=feature_df.columns.tolist()
    ).sort_values()
    return brute_features, eliminator


def multistep_rfe_importance(
    feature_df, labels, clf, steps_feats_tup, n_jobs=-2, verbose=0, **model_kwargs
):
    feats = feature_df.columns
    for step, n_feats_out in steps_feats_tup:
        fts, elim = brute_force_importance_rf_clf(
            feature_df[feats],
            labels,
            clf,
            n_feats_out,
            step_size=step,
            n_jobs=n_jobs,
            verbose=verbose,
            **model_kwargs
        )
        feats = fts.index
        if verbose:
            pprint.pp(fts)
    return fts, elim


def get_subsample_from_meta(
    field, include=None, exclude=None, feature_df=None, meta_df=None
):
    """
    Retrieves a single selection from metadata as INCHI keys.
    Parameters
    ----------
    meta_df
    feature_df
    field
    include
    exclude
    """
    subsample = list()
    if feature_df is None or meta_df is None:
        x, y, m = load_training_data()
        if feature_df is None:
            feature_df = x
        if meta_df is None:
            meta_df = load_metadata()
    for ind, val in meta_df[field].items:
        if all([i.lower() in val.lower() for i in include]) and not any(
            [i.lower() in val.lower() for i in exclude]
        ):
            subsample.append(ind)
    return subsample


def get_epa_sol_all_insol(feature_df, labels, tups):
    # insol_samples = pd.concat([tups['epa_in'][0], tups['en_in'][0]]).index.intersection(train_insols)
    # train_sols = labels[labels == 1].index
    # sol_samples = tups['epa_sol'][0].index.intersection(train_sols)
    en_in = tups["en_in"][0][
        [c for c in tups["en_in"][0].index if c not in tups["epa_in"][0].index]
    ]
    all_samples_ind = pd.concat(
        [tups["epa_in"][0], en_in, tups["epa_sol"][0]]
    ).index.intersection(feature_df.index)
    all_samples = labels[all_samples_ind]
    all_samples[all_samples == "Insoluble"] = 0
    all_samples[all_samples == "Soluble"] = 1
    select_y = labels[all_samples.index]
    select_X = feature_df.loc[all_samples.index]
    print(select_X.head())
    # assert not select_X.isna().any()
    return select_X, select_y


def sample_observations_by_source(
    groups_observations, freq_list, replace=False, random_state=None
):
    if all([0.0 <= p <= 1.0 for p in freq_list]):
        freq_list = [int(p * len(n)) for n, p in zip(groups_observations, freq_list)]
    yield itertools.repeat(
        [
            pd.concat(
                [
                    obs.sample(n=freq, random_state=random_state, replace=replace)
                    for obs, freq in zip(groups_observations, freq_list)
                ]
            )
        ]
    )


def mix_insol_all_insol(feature_df, labels, tups, epa_ratio=0.75):
    train_insols = labels[labels == 0].index
    insol_samples = pd.concat([tups["epa_in"][0], tups["en_in"][0]]).index.intersection(
        train_insols
    )
    train_sols = labels[labels == 1].index
    epa_sol_samples = tups["epa_sol"][0].index.intersection(train_sols)
    en_sol_samples = tups["en_sol"][0]
    select_y = labels[all_samples]
    select_X = feature_df.loc[all_samples][feature_df.columns]
    return select_X, select_y


def data_by_groups(labels, group_dict):
    ind_dict = dict()
    for k, v in group_dict.items():
        ind_dict[k] = v[0].index.intersection(labels.index)
    return ind_dict


n_features_out = 40
project_dir = "//"
# exp_dir = '{}models/hyperopted-training_curve/'.format(project_dir)
feature_dir = "{}models/epa_solubles_hyperparam/".format(project_dir)
exp_dir = "{}models/epa_solubles_more_selection/".format(project_dir)
train_data_dir = "{}epa_solubles_atom_type_count/".format(project_dir)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
# Select EPA soluble/combined insoluble dataset.
meta = load_metadata()
interp_X, interp_y = [stepwise_duplicate(a) for a in get_interpretable()]
# Drop duplicates
print("Original size: {}".format(interp_X.shape))
g_dict = data_by_groups(interp_y, meta)
full_groups_list = dict(
    [
        (k, pd.Series(data=k, index=v, name=k))
        for k, v in g_dict.items()
        if k in ["epa_sol", "epa_in", "en_in"]
    ]
)
epa_enamine_overlap = full_groups_list["en_in"].index.intersection(
    full_groups_list["epa_in"].index
)
print("EPA-Enamine overlap: \n{}".format(epa_enamine_overlap))
full_groups_list["en_in"].drop(index=epa_enamine_overlap, inplace=True)
combined_y = interp_y[
    stepwise_duplicate(pd.concat(full_groups_list.values(), sort=True)).squeeze().index
]

combined_X = interp_X.loc[combined_y.index]
combined_X = combined_X[combined_X.var(axis=1) > 1e-4]
check_X_y(combined_X, combined_y)
print(combined_y.size)
print(combined_y.value_counts())


def averaged_spearman_mutlicollinear(
    feature_df, labels, save_dir, cv=5, nan_policy="omit"
):
    spear_list, ward_list, fig_list = list(), list(), list()
    for i in np.arange(cv):
        feature_rus, labels_rus = RandomUnderSampler(
            random_state=1000 * i
        ).fit_resample(feature_df, labels)
        feature_rus = feature_rus[feature_rus.var(axis=1) > 1e-4]
        feat_names, spear, dendro, spearfig1, spearfig2 = (
            qsar_modeling.feature_selection.multicollinear.spearman_rank_multicollinear(
                feature_rus, nan_policy="omit"
            )
        )
        spear_list.append(spear)
        ward_list.append(dendro)
        fig_list.append(spearfig1)
    col_list = spear_list[0].index
    avg_spear = spear_list[0]
    for i, sfig in enumerate(fig_list):
        # sfig.savefig('{}spearman_hierarchical_{}.svg'.format(exp_dir, i), transparent=True)
        col_list = spear_list[i].index.intersection(col_list)
        if i > 0:
            avg_spear = avg_spear.add(other=spear_list[i])
    avg_spear /= len(spear_list)
    if avg_spear.isna().count().count() > 0:
        print("Average correlation matrix contains NaNs!!!")
        print(avg_spear)
    feat_names, avg_corr, avg_dendro, avg_fig1, avg_fig2 = (
        multicollinear.spearman_rank_multicollinear(
            feature_df=feature_df[col_list],
            corr=avg_spear,
            nan_policy="raise",
            cluster_feats=True,
        )
    )
    print([x for x in feat_names])
    spearman_feature_path = "{}average_spearman_clustered_features.csv".format(exp_dir)
    feat_names.to_series().to_csv(spearman_feature_path)
    avg_fig2.dpi = 1000
    avg_fig2.savefig(
        "{}spearman_hierarchical_avg_250_extra_trees.svg".format(exp_dir),
        transparent=True,
    )
    dendro_path = "{}average_dendro_obj_250_extra_trees.pkl".format(exp_dir)
    with open(dendro_path, "wb") as fn:
        pickle.dump(avg_dendro, fn)
    return feat_names, avg_dendro


# full_groups = full_groups[full_groups.index.drop_duplicates()]
# Determine feature set.
# ranked_features_path = '{}feature_rankings.csv'.format(feature_dir)
# ranked_features_path = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/models/epa_solubles_hyperparam/feature_rankings.csv"

ranked_features_path = "{}1000_extra_trees_500-100_250-25.csv".format(exp_dir)
ranked_features = pd.DataFrame([])
# if os.path.isfile(ranked_features_path):
if False:
    ranked_features = load_feature_rankings(ranked_features_path, threshold=1)
    print("Ranked features loaded: \n{}".format(len(ranked_features)))
    selected_train_X = combined_X[ranked_features]
elif True:
    feature_method = "brute"
    if feature_method == "brute":
        print("Brute force feature selection started.")
        # train_X, train_y = get_epa_sol_all_insol(combined_X, combined_y, meta)
        step_tups = ((100, 500), (25, 250))
        xt = ExtraTreesClassifier(
            n_estimators=1000, random_state=0, n_jobs=-1, class_weight="balanced"
        )
        rfc = RandomForestClassifier(
            random_state=0, oob_score=True, n_jobs=-1, class_weight="balanced"
        )
        lr = LogisticRegressionCV(
            Cs=np.geomspace(1e-5, 1.0, 10),
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=-1,
            random_state=0,
            cv=5,
            tol=1e-3,
            l1_ratios=[0.25],
        )
        ranked_features, rfe_model = multistep_rfe_importance(
            combined_X, combined_y, clf=xt, steps_feats_tup=step_tups, verbose=2
        )
        ranked_features.to_csv(
            ranked_features_path, index_label="Features"
        )  # , float_format='%.4f')
        ranked_X = combined_X[ranked_features.index]
        indie_features, indie_dendro = averaged_spearman_mutlicollinear(
            ranked_X, combined_y, exp_dir
        )

        selected_train_X = ranked_X[indie_features].copy()
        # print([c for c in selected_train_X.columns if 'e-state' in c.lower()])
    elif feature_method == "list":
        print("Using E-count descriptors")
        ecounts = feature_name_lists.get_estate_counts(combined_X.columns.tolist())
        selected_train_X = combined_X[ecounts]
# Get split indices.
exit(67)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
log_mcc_list, log_bac_list, log_auc_list = list(), list(), list()
for i in np.arange(5):
    i = i + 1
    rus_dir = "{}rus{}/".format(exp_dir, i)
    os.makedirs(rus_dir, exist_ok=True)
    if not os.path.isdir(rus_dir) and not os.path.exists(rus_dir):
        print("Not here.")
    else:
        print("{} already exists.".format(rus_dir))
    check_X_y(selected_train_X, combined_y)
    """
    with open('{}X_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(selected_train_X, f)
    with open('{}y_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(combined_y, f)
    with open('{}grouped_by_source.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(full_groups_list, f)
        """
    rand_idx = interp_y.index.tolist()
    random.shuffle(rand_idx)
    shuffle_X = interp_X[ranked_features].loc[rand_idx]
    shuffle_y = interp_y[rand_idx]
    epa_rand = combined_y.index.tolist()
    shuffle_epa_X = selected_train_X.loc[epa_rand]
    shuffle_epa_y = combined_y[epa_rand]
    rus_data, rus_labels = RandomUnderSampler(random_state=1000 * i).fit_resample(
        shuffle_X, shuffle_y
    )
    epa_rus, epa_labels = RandomUnderSampler(random_state=1000 * i).fit_resample(
        shuffle_epa_X, shuffle_epa_y
    )

    # insol_ser = pd.concat([full_groups_list['epa_in'], full_groups_list['en_in']], verify_integrity=True).squeeze()
    # full_groups_list['epa_sol'].drop(index=full_groups_list['epa_sol'].index.intersection(insol_ser.index), inplace=True)
    # shared_ind = rus_y.index.drop_duplicates().intersection(full_groups.index.drop_duplicates())
    # shared_ind = [c for c in rus_y.index.drop_duplicates() if c in full_groups.index.drop_duplicates()]
    print("Input size: {}".format(combined_y.size))
    print("RUS size: {}".format(rus_labels.size))
    rus_y = stepwise_duplicate(rus_labels)
    rus_X = stepwise_duplicate(rus_data)
    scramble_y = pd.Series(
        data=np.random.randint(0, 2, size=rus_y.size), index=rus_y.index
    )
    # Even group sampling
    rus_groups = [
        g[g.index.intersection(rus_labels.index)] for g in full_groups_list.values()
    ]
    group_folds = stepwise_duplicate(get_quadratic_test_folds(rus_groups))
    # rus_y = remove_duplicate_idx(rus_labels)[group_folds.index]
    # rus_X = remove_duplicate_idx(shuffle_X.loc[rus_y.index]).loc[group_folds.index]
    # cv_method = PredefinedSplit(group_folds)

    # rus_X = shuffle_X
    # rus_y = shuffle_y

    cv_method = None
    import cv_tools
    from sklearn.metrics import (
        matthews_corrcoef,
        balanced_accuracy_score,
        roc_auc_score,
    )

    splitter_kw = {"shuffle": True, "random_state": 0}
    # Name of model for saving files!
    # model_name = "rf_hyperopt_strat_shuffled"
    hyper_path = "{}grid_search/".format(rus_dir)
    if not os.path.isdir(hyper_path):
        os.mkdir(hyper_path)
    model_name = "rf_alldata_trained_epa_scored_strat_shuffled"
    # Permuted
    #
    #
    # rus_y = scramble_y
    model = RandomForestClassifier
    forest_params = ParameterGrid(
        {
            "n_estimators": [10, 50, 100],
            "min_impurity_decrease": [0, 0.00005, 0.0001, 0.00025],
            "max_features": [3, 5],
            "max_leaf_nodes": [100, 150, 200, 250],
            "bootstrap": [False],
        }
    ).param_grid
    search_party = GridSearchCV(
        model(),
        forest_params,
        scoring=make_scorer(matthews_corrcoef),
        cv=cv_method,
        n_jobs=-1,
        error_score="raise",
        return_train_score=True,
        verbose=0,
    ).fit(,
    pprint.pp(search_party.best_params_)
    pprint.pp(search_party.best_score_)
    clf = search_party.best_estimator_
    """
    try:
        with open('{}gridsearch_obj.pkl'.format(hyper_path), 'wb') as f:
            pickle.dump(search_party, f)
    except:
        with open('{}best_model.pkl'.format(hyper_path), 'wb') as f:
            pickle.dump(search_party.best_estimator_, f)
    """
    print("CV Optimized Random Forest Scoring")
    for train_X, train_y, test_X, test_y in cv_tools.split_df(
        epa_rus, epa_labels, **splitter_kw
    ):
        print("Train value counts: \n{}".format(train_y.value_counts()))
        """        
        scale = RobustScaler().fit(train_X)
        train_X = scale.transform(train_X)
        test_X = scale.transform(test_X)
        clf = LogisticRegressionCV(max_iter=5000, n_jobs=-1, class_weight='balanced',
                                   scoring=make_scorer(matthews_corrcoef),
                                   l1_ratios=[0.5, 0.6, 0.7], solver='saga', penalty='elasticnet')
        print(clf.l1_ratio_)
        print(clf.C_)        
        """
        # alldata_X = interp_X[ranked_features].drop(index=train_X.index)
        alldata_X = selected_train_X.drop(index=train_X.index)
        alldata_y = combined_y.drop(index=train_X.index)
        clf.fit(train_X, train_y)
        train_pred = clf.predict(train_X)
        test_pred = clf.predict(test_X)
        interp_pred = clf.predict(alldata_X)
        log_mcc_list.append(
            (
                matthews_corrcoef(train_y, train_pred),
                matthews_corrcoef(test_y, test_pred),
                matthews_corrcoef(alldata_y, interp_pred),
            )
        )
        log_bac_list.append(
            (
                balanced_accuracy_score(train_y, train_pred),
                balanced_accuracy_score(test_y, test_pred),
                balanced_accuracy_score(alldata_y, interp_pred),
            )
        )
        log_auc_list.append(
            (
                roc_auc_score(train_y, train_pred),
                roc_auc_score(test_y, test_pred),
                roc_auc_score(alldata_y, interp_pred),
            )
        )
summary = dict()
for scorename, scorelist in zip(
    ["MCC", "BAC", "ROC-AUC"], [log_mcc_list, log_bac_list, log_auc_list]
):
    trains = [a[0] for a in scorelist]
    tests = [a[1] for a in scorelist]
    alldata = [a[2] for a in scorelist]
    # print(trains)
    # print(tests)
    for setname, scoreset in zip(
        ["Train", "Test", "AllData"], [trains, tests, alldata]
    ):
        sumname = "{}({})".format(scorename, setname)
        summary[sumname] = (
            np.mean(scoreset),
            np.std(scoreset),
            np.min(scoreset),
            np.median(scoreset),
            np.max(scoreset),
        )
[print("{}: {}".format(k, v)) for k, v in summary.items()]
summary_df = pd.DataFrame.from_dict(
    summary, orient="index", columns=["Mean", "StDev", "Min", "Median", "Max"]
)
summary_df.to_csv("{}{}_alldata_score_summary.csv".format(exp_dir, model_name))
exit()
# dev_summary, eva_summary, dev_dict, eva_dict = scoring.cv_model_generalized(clf, rus_X, rus_y, score_dir="{}logit/".format(rus_dir), scaler=RobustScaler())
# print(dev_summary)
# print(eva_summary)
# check_consistent_length(rus_y, rus_X)
# check_consistent_length(rus_y, group_folds)
# check_consistent_length(rus_X, group_folds)
# Hyperparameterize
for i in [0]:

    print(rus_X.head())
    print(rus_y.head())

    #  {'n_estimators': [100, 500], 'max_leaf_nodes': [250, 300], 'max_features': [3, 5, 10]}])
    # ,{'n_estimators': [10000], 'bootstrap': [False], 'ccp_alpha': [0.0001, 0.001, 0.005]}])

    for train_X, train_y, test_X, test_y in cv_tools.split_df(rus_X, rus_y):
        clf = model().set_params(**search_party.best_params_)
        clf.fit(train_X, train_y)
        train_pred = clf.predict(train_X)
        test_pred = clf.predict(test_X)
        print(
            matthews_corrcoef(train_y, train_pred), matthews_corrcoef(test_y, test_pred)
        )
        print(
            balanced_accuracy_score(train_y, train_pred),
            balanced_accuracy_score(test_y, test_pred),
        )
        print(
            roc_auc_score(train_y, train_pred, average="weighted"),
            roc_auc_score(test_y, test_pred, average="weighted"),
        )
    fitted = search_party.best_estimator_
    cv_dev_score_dict = dict()
    cv_eval_score_dict = dict()
    depth_list, node_list = list(), list()
    for t in search_party.best_estimator_.estimators_:
        depth_list.append(t.get_depth())
        node_list.append(t.get_n_leaves())
        # print(search_party.best_estimator_.tree_.max_depth())
    with open("{}cv_results_headline.txt", "w") as f:
        f.write(
            "Best Parameters: {}\nResults: {}".format(
                search_party.best_params_, search_party.best_index_
            )
        )
        f.write(
            "Tree depths: \nMean: {} \nStd Dev: {} \nMin: {} \nMedian: {} \nMax: {}".format(
                np.mean(depth_list),
                np.std(depth_list),
                np.min(depth_list),
                np.median(depth_list),
                np.max(depth_list),
            )
        )
        f.write(
            "Number of leaves:: \nMean {}: \nStd Dev: {} \nMin: {} \nMedian: {} \nMax: {}".format(
                np.mean(node_list),
                np.std(node_list),
                np.min(node_list),
                np.median(node_list),
                np.max(node_list),
            )
        )
    # Needs lots more coding for full score summary.
    # dev_score_dict, eva_score_dict = score_model(fitted, shuffle_X, fit_eva_X, rus_dev_y, rus_eva_y)
    """    
    with open('{}hyper-dev_scores.csv'.format(hyper_path), 'w') as f:
        csv.writer(f).writerow(cv_dev_score_dict.items())
    with open('{}hyper-eval_scores.csv'.format(hyper_path), 'w') as f:
        csv.writer(f).writerow(cv_eval_score_dict.items())    
    lcurve = learning_curve(RandomForestClassifier(), rus_X, rus_y,
                            n_jobs=-1, #  scoring=make_scorer(matthews_corrcoef),
                            fit_params=best.get_params())[0]
    print(lcurve)    
    best = RandomForestClassifier(max_features=5, max_leaf_nodes=250, min_impurity_decrease=5e-05, n_estimators=100, bootstrap=False, n_jobs=-1)
    fitted = best.fit(rus_X, rus_y)    
    """
    import matplotlib.pyplot as plt

    det_path = "det_curve_rus".format(rus_dir)
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    det = DetCurveDisplay.from_estimator(
        model(),
        X=rus_X,
        y=rus_y,
        pos_label=0,
        name="Detection Error Curve: Random Forest",
        ax=ax,
        fit_params=search_party.best_params_,
    )
    det.plot(name="Detection Error Curve: RF (Undersampled)", ax=ax)
    print(det.fnr, det.fpr)
    plt.savefig(fname="{}.svg".format(det_path), transparent=True)
    whole_stub = "{}learning_curve_not_rus".format(rus_dir)
    # whole_lcd, _, __ = scoringlearn_curve(best, 'Random Forest (All Data)', shuffle_X, shuffle_y, fname_stub='{}all_data_lc'.format(rus_dir))
    continue
    # cv_model_generalized(best, X=rus_X, y=rus_y, cv_inst=PredefinedSplit(group_folds), score_dir=rus_dir)
    std_scaler = StandardScaler().fit(rus_X, )
    fit_dev_X = std_scaler.fit_transform(rus_X)
    fit_eva_X = std_scaler.transform(rus_eva_X)
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    # train_curve_ser = pd.Series(lcurve[1], index=lcurve[0], name='Test Scores')
    # test_curve_ser = pd.Series(lcurve[2], index=lcurve[0], name='Train Scores')
    # test_curve_ser = pd.Series(lcurve.test_scores, index=train_sizes_abs, name='Train Scores')
    continue

    import seaborn as sns

    fig, ax = plt.subplots()
    train_plot = sns.lineplot(train_curve_ser)
    test_plot = sns.lineplot(test_curve_ser)
    fig.savefig(lc_path)
    [print(c) for c in lcurve]
    continue

    dev_keys, eva_keys = list(), list()
    for j in np.arange(5):
        cv_dir = "{}cv{}/".format(rus_dir, j + 1)
        dev_key_path = "{}dev_keys.csv".format(cv_dir, i)
        eval_key_path = "{}dev_keys.csv".format(cv_dir, i)
        if not os.path.isdir(cv_dir):
            os.mkdir(cv_dir)
        if os.path.isfile(dev_key_path) and os.path.isfile(eval_key_path):
            dev_keys.append(load_sample_keys(dev_key_path))
            eva_keys.append(load_sample_keys(eval_key_path))
        else:
            dev_ind, eva_ind = [
                (d, e) for d, e in StratifiedKFold().split(rus_X, rus_y)
            ][j]
            dev_keys.append(rus_y.iloc[dev_ind].index)
            eva_keys.append(rus_y.iloc[eva_ind].index)
            with open("{}dev_keys.csv".format(cv_dir), "w") as f:
                csv.writer(f).writerow(rus_y.iloc[dev_ind].index)
            with open("{}eval_keys.csv".format(cv_dir), "w") as f:
                csv.writer(f).writerow(rus_y.iloc[eva_ind].index)
    continue
    # Manual Cross-val
    for i, (dev, eva) in enumerate(zip(dev_keys, eva_keys)):
        cv = i + 1
        cv_dir = "{}cv{}/".format(rus_dir, cv)
        print("Starting CV run at {}".format(datetime.datetime.now()))
        rus_dev_X, rus_dev_y = selected_train_X.loc[dev], train_y[dev]
        rus_eva_X, rus_eva_y = selected_train_X.loc[eva], train_y[eva]
        print(
            "Number of insoluble:soluble training compounds selected: {}:{}".format(
                rus_dev_y[rus_dev_y == 0].shape, rus_dev_y[rus_dev_y == 1].shape
            )
        )

        std_scaler = StandardScaler().fit(rus_dev_X, )
        continue
        fit_dev_X = std_scaler.transform(rus_dev_X)
        fit_eva_X = std_scaler.transform(rus_eva_X)
        logistic_cv = partial(LogisticRegressionCV, max_iter=1000, solver="newton-cg")
        positive_lab = rus_dev_y.unique()[0]
        # ('logistic-cv', logistic_cv),
        for model_name, model in [("optimized_forest", fitted.params)]:
            model_path = "{}{}/".format(cv_dir, model_name)
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            if model_name == "logistic-cv":
                fitted = model(n_jobs=-1).fit(fit_dev_X, rus_dev_y)
            # if model_name != 'logistic-cv':
            # fitted = model(bootstrap=False, max_leaf_nodes=200, n_estimators=10000, n_jobs=-1).fit(fit_dev_X, rus_dev_y)
            dev_scores, eva_scores = scoring.score_model(
                fitted, fit_dev_X, fit_eva_X, rus_dev_y, rus_eva_y
            )

            """
            for score_name, score_obj in prob_scores.items():
                cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_pred)
                cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_pred)
                # cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_prob)
                # cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_prob)
            """

            """
            with open('{}dev_proba.csv'.format(model_path), 'w') as f:
                csv.writer(f).writerow(eliminator.estimator_.predict_proba(rus_dev_y))
            with open('{}eval_proba.csv'.format(model_path), 'w') as f:
                csv.writer(f).writerow(eliminator.estimator_.predict_proba(rus_eva_y))
            """
    end = datetime.datetime.now()
    exec_time = end - start
    print("One CV run took {}".format(exec_time))
    exit()
