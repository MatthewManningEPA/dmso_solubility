import itertools
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import clone
from sklearn.model_selection import StratifiedKFold


def separate_floats_and_strs(mixed_list):
    float_list, str_list = list(), list()
    for n in mixed_list:
        try:
            float_list.append(float(n))
        except ValueError:
            str_list.append(str(n))
    return float_list, str_list


def get_score_features(feature_dir, score_path):
    if os.path.isdir(feature_dir) and os.path.isfile(score_path):
        with open(score_path, encoding="utf-8") as f:
            best_list = f.readlines()
    else:
        print("Score file not found: {}".format(score_path))
        return None, None
    scores, feature_lists = list(), list()
    for line in best_list:
        score_feats = line.split("\t")
        line_scores, line_feats = separate_floats_and_strs(score_feats)
        scores.append(tuple(line_scores))
        feature_lists.append([s.strip() for s in line_feats])
        if len(feature_lists[-1]) == 0:
            print("No features found in feature list: {}".format(score_feats))
    score_feat_df = pd.DataFrame(
        data=(list(zip(scores, [len(f) for f in feature_lists]))),
        index=list(range(len(scores))),
        columns=["Scores", "n_features"],
    )
    return score_feat_df, feature_lists


def heirarchical_predictions(
    feature_df, labels, model, features_lists, cv=(5, 5), binary=False, save_dir=None
):
    # feat_cv_preds_dict: stores CV predictions for all sets
    # cv_members = pd.DataFrame(index=labels.index, columns=["Split", "Repeat"])
    # print(feature_df.head(), labels.head())
    # Feature-set dictionary of lists of CV split predictions
    n_sets = list(range(len(features_lists)))
    prediction_path = "{}clustering/cv_{}x{}_predictions.pkl".format(
        save_dir, cv[0], cv[1]
    )
    if os.path.isfile(prediction_path):
        with open(prediction_path, "rb") as f:
            all_repeats_df = pickle.load(f)
        interrater_feats = dict()
        for set_ix in list(range(all_repeats_df.shape[1])):
            iterrater_path = "{}clustering/{}th_set_cv_{}x{}_predict_wide.csv".format(
                save_dir, set_ix, cv[0], cv[1]
            )
            if os.path.isfile(
                "{}clustering/{}th_set_cv_{}x{}_predict_wide.csv".format(
                    save_dir, set_ix, cv[0], cv[1]
                )
            ):
                interrater_feats[set_ix] = pd.read_csv(
                    iterrater_path, skiprows=[0, 1], index_col=0
                )
    else:
        repeat_keys = ["Repeat_{}".format(r) for r in np.arange(cv[0])]
        feature_repeats_predicts = dict([(i, dict()) for i in n_sets])
        interrater_feats = dict([(i, list()) for i in n_sets])
        set_sers_dict = dict([(i, list()) for i in n_sets])
        all_repeats_feat_sers = dict()
        # Fit, predict and organize results.
        for repeat_i in list(range(cv[0])):
            feat_cv_preds_dict = dict([(i, list()) for i in n_sets])
            rand_state = 4 * (repeat_i + 1) + 2
            for split_i, (train, test) in enumerate(
                StratifiedKFold(
                    n_splits=cv[1], shuffle=True, random_state=rand_state
                ).split(feature_df, labels)
            ):
                train_df, train_y = feature_df.iloc[train], labels.iloc[train]
                test_df, test_y = feature_df.iloc[test], labels.iloc[test]
                # cv_members.iloc[test] = (split, repeat)
                new_model = clone(estimator=model)
                for set_ix, feature_list in enumerate(features_lists):
                    retrained_model = new_model.fit(X=train_df[feature_list], y=train_y)
                    if binary:
                        new_predicts = retrained_model.predict(X=test_df[feature_list])
                    else:
                        new_predicts = retrained_model.predict_proba(
                            X=test_df[feature_list]
                        )
                    feat_cv_preds_dict[set_ix].append(
                        pd.Series(data=new_predicts[:, 0], index=test_df.index)
                    )
            # feature_repeats_predicts = concatenate predictions from previous KFold
            # ser = ser = rename indices with repeat num postfix for later stacking.
            for set_ix in n_sets:
                feature_repeats_predicts[set_ix][repeat_keys[repeat_i]] = (
                    pd.concat(feat_cv_preds_dict[set_ix]).squeeze().copy()
                )
                feature_repeats_predicts[set_ix][repeat_keys[repeat_i]] = (
                    feature_repeats_predicts[set_ix][repeat_keys[repeat_i]]
                    .reset_index()
                    .drop_duplicates(ignore_index=True)
                )
                feature_repeats_predicts[set_ix][repeat_keys[repeat_i]].set_index(
                    keys=feature_repeats_predicts[set_ix][
                        repeat_keys[repeat_i]
                    ].columns[0],
                    inplace=True,
                )
                # feature_repeats_predicts[set_ix][repeat_keys[repeat_i]].name = repeat_keys[repeat_i]
                # pd.DataFrame.set_index()
                # print(feature_repeats_predicts[set_ix][-1].iloc[0])
                ser = pd.concat(feat_cv_preds_dict[set_ix]).copy()
                ser.index = ["{}_{}".format(x, str(repeat_i)) for x in ser.index]
                set_sers_dict[set_ix].append(ser)
        # interrater_feats = dictionary of whole set predictions stacked horizontally
        os.makedirs("{}clustering/".format(save_dir), exist_ok=True)
        for set_ix in n_sets:
            all_repeats_feat_sers[set_ix] = pd.concat(set_sers_dict[set_ix], sort=True)
            # print(all_repeats_feat_sers[set_ix])
            all_repeats_feat_sers[set_ix].index.name = "INCHI"
            all_repeats = all_repeats_feat_sers[set_ix].reset_index()
            all_repeats.drop_duplicates(inplace=True, ignore_index=True)
            # print(all_repeats_feat_sers[set_ix])
            # all_repeats_feat_sers[set_ix] = all_repeats.set_index(keys="Repeat")
            # set_idx = all_repeats_feat_sers[set_ix].index.tolist()
            # dropped_idx = [i for i in set_idx if type(i) is str and i[-2] != "_"]
            # all_repeats_feat_sers[set_ix].drop(index=dropped_idx, inplace=True)
            all_repeats_feat_sers[set_ix] = all_repeats.set_index(
                keys="INCHI"
            ).squeeze()
            """
            print(
                [
                    feature_repeats_predicts[set_ix][i].set_index(keys="index")
                    for i in np.arange(cv[0])
                ]
            )
            inter_feats = dict(
                [
                    (
                        "Repeat_{}".format(i),
                        feature_repeats_predicts[set_ix][i].set_index(keys="index"),
                    )
                    for i in np.arange(cv[0])
                ]
            )
            print("After set index.\n")
            print(inter_feats.values())
            print(inter_feats.keys())            
            try:
                interrater_feats[set_ix] = pd.DataFrame(
                    [c for c in inter_feats.values()],
                    columns=[c for c in inter_feats.keys()],
                )
            except ValueError:
                print("Trying concat on inter_feats")
                interrater_feats[set_ix] = pd.concat(inter_feats)
                except ValueError:
                print("Trying from_dict on inter_feats")
                interrater_feats[set_ix] = pd.DataFrame.from_dict(inter_feats)
            except ValueError:            
            """
            print("Trying concat on inter_feats")
            interrater_feats[set_ix] = pd.concat(
                feature_repeats_predicts[set_ix], axis=1
            )

            print(interrater_feats[set_ix])
            interrater_feats[set_ix].to_csv(
                "{}clustering/{}th_set_cv_{}x{}_predict_wide.csv".format(
                    save_dir, set_ix, cv[0], cv[1]
                )
            )
        try:
            all_repeats_df = pd.concat(list(all_repeats_feat_sers.values()), axis=1)
        except ValueError:
            # print(all_repeats_feat_sers)
            # print(all_repeats_feat_sers[0].shape)
            all_repeats_df = pd.DataFrame.from_dict(
                all_repeats_feat_sers,
                orient="index",
                columns=["Set_{}".format(i) for i in n_sets],
            )
        # print(all_repeats_df.shape)
        all_repeats_df.to_pickle(
            "{}clustering/cv_{}x{}_predictions.pkl".format(save_dir, cv[0], cv[1])
        )
    return all_repeats_df, interrater_feats


def average_subset_repeats(score_df_dict):
    # score_df_list: list of n_samples x n_repeats for each feature subsets
    mean_score_ser_dict = dict(
        (k, df.mean(axis=1, numeric_only=True)) for k, df in score_df_dict.items()
    )
    mean_scores_df = pd.DataFrame.from_dict(mean_score_ser_dict)
    print("Average CV Scores")
    print(mean_scores_df)
    return mean_scores_df


def grab_selection_results(feature_dir, best_only=True):
    score_path = "{}feature_score_path.csv".format(feature_dir)
    if not os.path.isdir(feature_dir) or not os.path.isfile(score_path):
        return None, None, None, None
    score_feat_df, feature_subsets = get_score_features(feature_dir, score_path)
    if score_feat_df is None:
        return None, None, None, None
    print("Scoring results retrieved.")
    score_feat_df["Mean"] = score_feat_df["Scores"].apply(np.mean)
    score_feat_df["Std"] = score_feat_df["Scores"].apply(np.std)
    score_feat_df["AdjMean"] = score_feat_df["Mean"] - score_feat_df["Std"]
    score_feat_df.sort_values(by="AdjMean", ascending=False).drop_duplicates(
        subset="n_features", inplace=True
    )
    print("Score Summary")
    pprint.pp(
        score_feat_df[["Mean", "Std", "AdjMean"]]
        .sort_values(by="AdjMean", ascending=False)
        .head(),
        compact=True,
        width=100,
    )
    best_adj = score_feat_df["AdjMean"].max()
    score_feat_df.drop(columns=["Mean", "Std", "AdjMean"], inplace=True)
    separate_scores_df = pd.DataFrame(
        score_feat_df["Scores"].to_list(), index=score_feat_df.index
    )
    separate_scores_df.columns = [
        "Repeat_{}".format(i) for i in np.arange(separate_scores_df.shape[1])
    ]
    separate_scores_df = pd.concat(
        [separate_scores_df, score_feat_df["n_features"]], copy=True, axis=1
    )
    # print(separate_scores_df, score_feat_df, feature_subsets, best_adj)
    # score_feat_df["Grove"] = grove_names[i].split("\t")[0]
    return separate_scores_df, score_feat_df, feature_subsets, best_adj


def main():
    model_nick, full_model_name = "logit", "Logistic Regression"
    model_dir = "{}{}_grove_features_3/".format(os.environ.get("MODEL_DIR"), model_nick)
    grove_names = pd.read_csv(
        "{}feature_names.csv".format(model_dir), index_col=0, sep="\t"
    ).squeeze()
    grove_names.name = "Grove"
    scores_df_dict, grove_df_dict, features_dict, best_adj_dict = (
        dict(),
        dict(),
        dict(),
        dict(),
    )
    # grove_nick = grove_names.index.map(mapper=lambda x: x[: min(len(x), 40)]).tolist()
    for i in list(range(0, 50)):
        feature_dir = "{}{}/".format(model_dir, i)
        s, g, f, b = grab_selection_results(feature_dir)
        if any([a is None for a in [s, g, f, b]]):
            break
        grove_df_dict[i], features_dict[i], best_adj_dict[i] = g, f, b
    print(s)
    grove_df_dict = dict(
        sorted(grove_df_dict.items(), key=lambda x: best_adj_dict[x[0]], reverse=True)
    )
    # sns.set_context("talk")
    grove_nick = [
        "{}   {} {}".format(x[:50], y[0], y[1]) for x, y in grove_names.iterrows()
    ]
    grove_list = list(grove_df_dict.values())
    sns.set_style("whitegrid")
    long_list = list()
    for g_i, (grove_df) in enumerate(grove_list):
        scores = pd.DataFrame(grove_df["Scores"].to_list())
        scores_names = ["CV{}".format(i) for i in np.arange(scores.shape[1])]
        scores.columns = scores_names
        grove_df = pd.concat([scores, grove_df["n_features"].astype(int)], axis=1)
        grove_df["Grove"] = grove_nick[g_i]
        long_list.append(grove_df.melt(
            id_vars=["n_features", "Grove"], var_name="fold", value_name="score"))
    for i, (grove_dfs) in enumerate(itertools.batched(long_list, n=4)):
        long_df = pd.concat(grove_dfs)
        score_grid = sns.FacetGrid(
            long_df,
            col="Grove",
            col_wrap=2,
            height=4,
            xlim=(0, 31),
            ylim=(0.5, 1.0),
        )
        print(long_df)
        # grove_df = pd.concat(grove_dfs).pivot(columns=["n_features"]) # .explode(column="Scores", ignore_index=True)
        score_grid.map(
            sns.boxplot,
            data=long_df,
            x="n_features",
            y="score",
            order=list(np.arange(2,31)),
            native_scale=True,
        )
        # score_grid.map(sns.boxplot, grove_df, "n_features", scores.columns, native_scale=True, ax=axs[i])
        score_grid.set_axis_labels("Num Features", "CV Scores")
        score_grid.set(xticks=[0, 10, 20, 30], yticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        score_grid.figure.subplots_adjust(wspace=0.02, hspace=0.2)
        score_grid.savefig("{}feature_selection_best_{}.png".format(model_dir, i))
        score_grid.fig.subplots_adjust(top=0.9)
        score_grid.fig.suptitle(full_model_name)
    # plt.show()


if __name__ == "__main__":
    main()
