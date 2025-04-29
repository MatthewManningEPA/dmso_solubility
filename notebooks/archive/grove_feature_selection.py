import copy
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import clone
from sklearn.preprocessing import RobustScaler
from sklearn.tree import (
    DecisionTreeClassifier,
)
from sklearn.utils import compute_sample_weight

import sample_clusters
from correlation_filter import cross_corr_filter
from dataset_creation import padel_candidate_features
from vif import repeated_stochastic_vif

mcc = make_scorer(balanced_accuracy_score)


def _permutation_removal(
    train_df, labels, feature_list, grove_model, dropped_dict, fails, select_params
):
    n_repeats = select_params["perm_n_repeats"]
    while len(feature_list) > select_params["features_min_perm"]:
        perm_model = clone(grove_model).fit(train_df[feature_list], labels)
        perm_results = permutation_importance(
            perm_model,
            train_df[feature_list],
            labels,
            n_repeats=n_repeats,
            scoring=mcc,
            n_jobs=-1,
        )
        import_mean = pd.Series(
            data=perm_results["importances_mean"],
            index=feature_list,
            name="Mean",
        ).sort_values()
        import_std = pd.Series(
            data=perm_results["importances_std"],
            index=feature_list,
            name="StD",
        )
        adj_importance = import_mean.iloc[0] + import_std[import_mean.index].iloc[0]
        if fails < 2 or len(feature_list) < 15:
            import_thresh = min(select_params["thresh_perm"] + 0.05, adj_importance)
        else:
            import_thresh = min(select_params["thresh_perm"], adj_importance)
        unimportant = import_mean[import_mean <= import_thresh].index.tolist()
        if len(unimportant) > 1 and n_repeats <= 50:
            n_repeats = n_repeats + 25
            continue
        elif len(unimportant) == 1 or (n_repeats >= 50 and len(unimportant) > 0):
            low_feats = unimportant[0]
        else:
            fails -= 1
            break
            # [feature_list.remove(c) for c in low_feats]
        feature_list.remove(low_feats)
        dropped_dict.update([(low_feats, "Importance")])
        print(
            "Permutation drop: \n{}: {} {}".format(
                low_feats,
                import_mean[low_feats],
                import_std[low_feats],
            )
        )
        fails -= 1
    return feature_list, dropped_dict, fails


def _vif_elimination(
    train_df,
    feature_list,
    best_corrs,
    cross_corr,
    dropped_dict,
    fails,
    select_params,
    verbose=False,
):
    model_size = 10
    vif_rounds = int(len(feature_list) * (len(feature_list) + 1) / model_size)
    # fweights = 0.5 - pd.Series(scipy.linalg.norm(cross_corr.loc[feature_list, feature_list], axis=1), index=feature_list).squeeze().sort_values(ascending=False)
    vif_results = repeated_stochastic_vif(
        feature_df=train_df[feature_list],
        importance_ser=best_corrs[feature_list],
        threshold=select_params["thresh_vif"],
        model_size=model_size,
        feat_wts=cross_corr[feature_list].loc[feature_list],
        min_feat_out=len(feature_list) - 1,
        rounds=vif_rounds,
    )
    if len(vif_results["vif_stats_list"]) > 1:
        all_vif_mean = (
            pd.concat([df["Max"] for df in vif_results["vif_stats_list"]], axis=1)
            .max()
            .sort_values(ascending=False)
        )
        all_vif_sd = pd.concat(
            [df["StdDev"] for df in vif_results["vif_stats_list"]], axis=1
        ).std(ddof=len(feature_list) - model_size)
    else:
        all_vif_mean = vif_results["vif_stats_list"][0]["Max"].sort_values(
            ascending=False
        )
        all_vif_sd = vif_results["vif_stats_list"][0]["SD"]
    vif_dropped = all_vif_mean.index[0]
    # [print(c, [d.loc[c] for d in vif_results["vif_stats_list"] if c in d.index]) for c in vif]
    if (
        vif_dropped in feature_list
        and all_vif_mean[vif_dropped] > select_params["thresh_vif"]
    ):
        feature_list.remove(vif_dropped)
        dropped_dict.update([(vif_dropped, "VIF")])
        fails -= 1
        if False:
            print("VIF Scores")
            pprint.pp(
                pd.concat(
                    [
                        all_vif_mean,
                        all_vif_sd[all_vif_mean.index],
                        vif_results["votes_list"][-1].loc[all_vif_mean.index],
                    ],
                    axis=1,
                ).head(n=3),
                width=120,
            )
    else:
        fails -= 1
        if verbose:
            print("Nothing above threshold.")

    return feature_list, dropped_dict, fails


def grove_features_loop(
    feature_df, labels, grove_model, member_col=None, cv=None, save_dir=None
):
    select_params = {
        "max_features_out": 15,
        "fails_min_vif": 3,
        "fails_min_perm": 3,
        "fails_min_sfs": 4,
        "features_min_vif": 5,
        "features_min_perm": 20,
        "features_min_sfs": 15,
        "thresh_vif": 10,
        "thresh_perm": 0,
        "thresh_sfs": -0.005,
        "permutate": False,
    }
    with open("{}selection_params.txt".format(save_dir), "w") as f:
        for k, v in select_params.items():
            f.write("{}: {}\n".format(k, v))
    score_dict, best_features, dropped_dict = dict(), list(), dict()
    top_score, last_best, fails = 0.5, 0.5, 0
    if cv is None:
        cv = RepeatedStratifiedKFold(random_state=0, n_repeats=5)
    best_corrs, cross_corr, train_df = process_selection_data(
        dropped_dict, feature_df, labels, save_dir
    )
    assert train_df.index.equals(labels.index)
    train_df.to_pickle("{}preprocessed_feature_df.pkl".format(save_dir))
    labels.to_csv("{}member_labels.csv".format(save_dir))
    print("Training Size: {}".format(train_df.shape))
    for n in labels.unique():
        if labels[labels == n].shape[0] < 10:
            return None
    """    
    pc = PCA(n_components="mle").fit(train_df)
    print("PCA Results: Noise Variance and Number of Components to Reach 0.9 EVR:")
    print(
        pc.noise_variance_,
        len([v for v in np.cumsum(pc.explained_variance_ratio_) if v < 0.9]),
    )
    """
    if member_col is None or member_col not in train_df:
        print("\nFeature selection for category: {}".format(member_col))
        feature_list = [best_corrs.abs().sort_values(ascending=False).index[0]]
    else:
        feature_list = [member_col]
    pprint.pp(dropped_dict.items())
    sqcc_df = cross_corr * cross_corr
    # Start feature loop
    while (
        len(feature_list) < select_params["max_features_out"] - 1
        and best_corrs.shape[0] - len([dropped_dict.keys()]) - len(feature_list) - 1 > 0
    ):
        clean_up = False
        if fails > 5:
            print(dropped_dict.items())
            print(feature_list)
        feat_corrs = best_corrs.drop(
            [
                c
                for c in best_corrs.index
                if c in dropped_dict.keys() or c in feature_list
            ]
        )

        sum_sqcc = (1 - sqcc_df).loc[feat_corrs.index][feature_list].sum(axis=1)
        feat_probs = scipy.special.softmax(np.abs(feat_corrs) * sum_sqcc)
        feature_list = list(set(feature_list))
        # noinspection PyTypeChecker
        new_feat = np.random.choice(
            a=feat_corrs.index.to_numpy(), replace=False, p=feat_probs
        )
        if new_feat not in cross_corr.columns:
            print("{} not in cross_corr".format(new_feat))
            continue
        feature_list.append(new_feat)
        scores = cross_val_score(
            grove_model,
            X=train_df[feature_list],
            y=labels,
            scoring=mcc,
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        best_features, score_dict, top_score, last_best, fails = record_score(
            best_features,
            fails,
            feature_list,
            save_dir,
            score_dict,
            scores,
            top_score,
            last_best,
        )
        if len(feature_list) < 10:
            continue
        if (
            fails >= select_params["fails_min_vif"]
            and len(feature_list) >= select_params["features_min_vif"]
        ):
            feature_list, dropped_dict, fails = _vif_elimination(
                train_df,
                feature_list,
                best_corrs,
                cross_corr,
                dropped_dict,
                fails,
                select_params,
            )
            while (
                fails >= select_params["features_min_perm"]
                and select_params["features_min_perm"]
                < len(feature_list)
                < select_params["max_features_out"]
            ):

                if select_params["permutate"]:
                    # print("Starting permutation importance")
                    feature_list, dropped_dict, fails = _permutation_removal(
                        train_df,
                        labels,
                        feature_list,
                        grove_model,
                        dropped_dict,
                        fails,
                        select_params,
                    )
        if len(feature_list) == select_params["max_features_out"]:
            clean_up = True
        while (
            fails >= select_params["fails_min_sfs"]
            and select_params["features_min_sfs"] < len(feature_list)
            or clean_up
        ):
            sfs_tol = select_params["thresh_sfs"]
            if clean_up:
                sfs_tol = 2 * select_params["thresh_sfs"]
            sfs = SequentialFeatureSelector(
                grove_model,
                direction="backward",
                tol=sfs_tol,
                n_features_to_select=len(feature_list) - 1,
                scoring=mcc,
                n_jobs=-1,
            ).fit(train_df[feature_list], y=labels)
            new_features = list(sfs.get_feature_names_out(feature_list))
            sfs_drops = [c for c in feature_list if c not in new_features]
            print("Dropped by SFS:")
            [print(c) for c in sfs_drops]
            dropped_dict.update([(c, "SFS") for c in sfs_drops])
            fails -= len(sfs_drops)
            sfs_score = cross_val_score(
                grove_model,
                train_df[new_features],
                labels,
                scoring=mcc,
                cv=cv,
                n_jobs=-1,
                error_score="raise",
            )
            if np.mean(sfs_score) - np.std(sfs_score) > top_score + sfs_tol:
                best_features, score_dict, top_score, last_best, fails = record_score(
                    best_features,
                    fails,
                    feature_list,
                    save_dir,
                    score_dict,
                    sfs_score,
                    top_score,
                    last_best,
                )
            [feature_list.remove(d) for d in sfs_drops]
            if len(feature_list) == select_params["max_features_out"]:
                fails = 0
                break
            else:
                fails -= 1
                clean_up = False
                continue
    print(
        "Best score of {} for {} with feature set: {}".format(
            top_score, member_col, best_features
        )
    )
    if len(best_features) > 0:
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(grove_model.fit(train_df[best_features], labels), f)
        pd.Series(best_features).to_csv(
            "{}best_features.csv".format(save_dir),
            index_label="Index",
            encoding="utf-8",
            header=member_col,
        )
    else:
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(grove_model, f)
    pd.Series(dropped_dict, name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    return grove_model, score_dict, dropped_dict, best_features


def process_selection_data(dropped_dict, feature_df, labels, save_dir):
    sample_wts = compute_sample_weight(class_weight="balanced", y=labels)
    scaler = (
        RobustScaler(unit_variance=True)
        .set_output(transform="pandas")
        .fit(
            feature_df,
        )
    )
    if save_dir is not None:
        with open("{}scaler.pkl".format(save_dir), "wb") as f:
            pickle.dump(scaler, f)
    train_df = scaler.transform(feature_df)
    cross_corr = train_df.corr(method="kendall")
    del_ser = cross_corr_filter(
        cross_corr, cutoff=0.95, n_drop=max(1, train_df.shape[1] - 30)
    )
    print([c for c in del_ser.index if type(c) is not str])
    train_df.drop(columns=del_ser.index, inplace=True)
    cross_corr = cross_corr.drop(columns=del_ser.index).drop(index=del_ser.index)
    dropped_dict.update([(c, "Cross-correlation") for c in del_ser.index])
    best_corrs = train_df.corrwith(labels, method="kendall").sort_values(
        ascending=False
    )
    na_corrs = best_corrs.index[best_corrs.isna()]
    [dropped_dict.update([(c, "NA Correlation")]) for c in na_corrs]
    best_corrs.drop(na_corrs, inplace=True)
    cross_corr = cross_corr.drop(index=na_corrs).drop(columns=na_corrs)
    train_df = train_df[cross_corr.index]
    best_corrs.sort_values(ascending=False, inplace=True, key=lambda x: np.abs(x))
    train_df.to_pickle("{}feature_df.pkl".format(save_dir))
    return best_corrs, cross_corr, train_df


def record_score(
    best_features,
    fails,
    feature_list,
    save_dir,
    score_dict,
    scores,
    top_score,
    last_best,
):
    with open("{}feature_score_path.csv".format(save_dir), "a", encoding="utf-8") as f:
        f.write(
            "{}\t{}\n".format(
                "\t".join([str(sc) for sc in scores]), "\t".join(feature_list)
            )
        )
    score_dict[tuple(feature_list)] = scores
    if np.mean(scores) - np.std(scores) > top_score:
        print(
            "New top results for {} feature model: {:.4f}, {:.4f}".format(
                len(feature_list), np.mean(scores), np.std(scores)
            )
        )
        top_score = np.mean(scores) - np.std(scores)
        last_best = copy.deepcopy(best_features)
        best_features = copy.deepcopy(feature_list)
        fails = max(0, fails - 3)
    else:
        fails += 1
    return best_features, score_dict, top_score, last_best, fails


def get_clf_model(model_name):
    if "log" in model_name:
        grove_model = LogisticRegressionCV(
            scoring=mcc,
            solver="newton-cholesky",
            tol=2e-4,
            cv=5,
            max_iter=10000,
            class_weight="balanced",
        )
    elif "xtra" in model_name:
        grove_model = ExtraTreesClassifier(
            max_leaf_nodes=250,
            min_impurity_decrease=0.05,
            max_depth=16,
            class_weight="balanced",
            bootstrap=False,
        )
    elif "tree" in model_name and "xtra" not in model_name:
        grove_model = DecisionTreeClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=15,
            class_weight="balanced",
        )
    else:
        grove_model = RandomForestClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    return grove_model


def membership(feature_df, labels, grove_cols, search_dir):
    grove_features = feature_df[grove_cols]
    member_dict, accepted_features = dict(), dict()
    i = 0
    k_one_tree = 20
    one_tree_df = sample_clusters.find_enriching_splits(feature_df[grove_cols], labels)
    top_k_splits = one_tree_df.sort_values(by="Impurity").iloc[:k_one_tree]
    print("Top {} one node splits".format(k_one_tree))
    pprint.pp(top_k_splits, width=120, compact=True)
    for col in grove_cols:
        members = grove_features[col][
            grove_features[col] != grove_features[col].mode().iloc[0]
        ].index
        insol = labels[members][labels[members] == 0]
        sol = labels[members][labels[members] == 1]
        if insol.shape[0] > 15 and sol.shape[0] > 15:
            print("Feature Selected: {}".format(col))
            print(
                "Dataset: Insoluble: {} Soluble: {}".format(
                    insol.shape[0], sol.shape[0]
                )
            )
            member_dict[col] = members
            os.makedirs("{}{}/".format(search_dir, i), exist_ok=True)
            # labels[members].to_csv("{}{}/feature_members.csv".format(search_dir, i))
            accepted_features[col] = (insol.shape[0], sol.shape[0])
            i += 1
        else:
            print("Feature: {}".format(col))
            print(
                "Did not meet requirements Insoluble: {} Soluble: {}".format(
                    insol.shape[0], sol.shape[0]
                )
            )
    pd.DataFrame.from_dict(
        accepted_features, orient="index", columns=["Insoluble", "Soluble"]
    ).to_csv("{}feature_names.csv".format(search_dir), sep="\t")
    return member_dict


def main():
    feature_df, labels, grove_cols, kurtosis_stats = sample_clusters.main()
    # kurtosis_stats.columns = ["Kurtosis", "Counts"]
    # split_df = sample_clusters.find_enriching_splits(feature_df[grove_cols], labels)
    # pprint.pp(split_df.sort_values(by="Impurity"), width=120, compact=True)
    # pos_samples, neg_samples = labels[labels == 0].index, labels[labels == 1].index
    model_name = "rfc"
    search_dir = "{}{}_all_samples_2/".format(os.environ.get("MODEL_DIR"), model_name)
    os.makedirs(search_dir, exist_ok=True)
    # members_dict = membership(feature_df, labels, grove_cols, search_dir)
    search_features = padel_candidate_features()
    model_dict, score_dict, dropped_dict, best_features = grove_features_loop(
        feature_df[search_features],
        labels,
        grove_model=get_clf_model(model_name),
        member_col="All Data",
        save_dir=search_dir,
    )
    return model_dict, score_dict, dropped_dict, best_features
    for i, (col, members) in enumerate(list(members_dict.items())):
        if (
            labels[members][labels[members] == 0].size < 15
            or labels[members][labels[members] == 1].size < 15
        ):
            continue
        col_dir = "{}/{}/".format(search_dir, i)
        os.makedirs(col_dir, exist_ok=True)
        with sklearn.config_context(transform_output="pandas"):
            model_dict[col], score_dict[col], dropped_dict, best_features = (
                grove_features_loop(
                    feature_df.loc[members][search_features],
                    labels[members],
                    grove_model=get_clf_model(model_name),
                    member_col=col,
                    save_dir=col_dir,
                )
            )


if __name__ == "__main__":
    main()
