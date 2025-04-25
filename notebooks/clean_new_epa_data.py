import copy
import difflib
import itertools
import os
import pickle
from functools import cache

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import descriptor_processing
import padel_categorization
from archive.epa_multiclass import get_conversions, safe_mapper, standardize_smiles
from DescriptorRequestor import QsarStdizer
from dmso_utils import data_tools
from dmso_utils.data_tools import get_epa_mapper, get_query_data, load_training_data


def bulk_rand(subset_list, jaccard=False):
    C = set(itertools.chain.from_iterable(subset_list))
    rand_arr = np.ones(shape=(len(subset_list), len(subset_list)))
    for i, j in itertools.pairwise(np.arange(len(subset_list))):
        x_i, x_j = set(subset_list[i]), set(subset_list[j])
        if jaccard:
            r_index = rand_index(x_i, x_j)
        else:
            r_index = rand_index(x_i, x_j, C)
        rand_arr[i, j] = rand_arr[j, i] = r_index
    return rand_arr


def rand_index(x_i, x_j, C=None):
    # Rand index = similarity between two subsets of C = conchordant / (conchordant + dischordant)
    a = x_i.intersection(x_j)
    b = x_i.difference(x_j)
    c = x_j.difference(x_i)
    if C is not None:
        d = C.difference(a)
    else:
        d = set()
    return (len(a) + len(d)) / (len(a) + len(b) + len(c) + len(d))


def report_overlaps(
    insol_df, maxed_sol_df, sol100_df, trim=True, report=False, save_dir=None
):
    for df in [maxed_sol_df, sol100_df]:
        df = df.astype(float)
    insol_100_overlaps = insol_df.index.intersection(sol100_df.index)
    max_insol_overlaps = maxed_sol_df.loc[
        maxed_sol_df.index.intersection(insol_df.index)
    ].infer_objects()
    maxed_insol_problematic = max_insol_overlaps[
        max_insol_overlaps["sample_concentration"].astype(float) > 4.95
    ].index.intersection(insol_df.index)
    if save_dir is not None:
        save_dir = "{}db_queries_12_2024/".format(os.environ.get("DATA_DIR"))
        insol_df.to_csv("{}insoluble_max_solubility_overlaps.csv".format(save_dir))
        maxed_insol_problematic["sample_concentration"].astype(
            np.float32
        ).sort_values().to_csv("{}insoluble_max_overlaps.csv".format(save_dir))
    if report:
        print(insol_df.shape[0], maxed_sol_df.shape[0], sol100_df.shape[0])
        print("INSOLUBLE and 100 mM OVERLAPS")
        print(insol_100_overlaps, flush=True)
        print("Insoluble/Soluble Overlaps: 4.95 or greater")
        print(
            maxed_insol_problematic.index.intersection(
                insol_df[insol_df["Insoluble"] == 1].index
            )
        )
        print(
            maxed_insol_problematic.index.intersection(
                insol_df[insol_df["Sparingly"] == 1].index
            )
        )
        print("Separated Insol/Maxed: Insoluble | Sparingly")
        print(
            insol_df.loc[insol_df.index.intersection(maxed_sol_df.index)][
                insol_df["Insoluble"] == 1
            ]
        )
        print(
            insol_df.loc[insol_df.index.intersection(maxed_sol_df.index)][
                insol_df["Sparingly"] == 1
            ],
            flush=True,
        )
        print("Soluble/Maxed Overlaps")
        print(
            maxed_sol_df.loc[
                maxed_sol_df["sample_concentration"] != 100
            ].index.intersection(sol100_df.index)
        )
        print("Soluble/Maxed Differences")
        print(sol100_df.index.symmetric_difference(maxed_sol_df.index))
    if trim:
        insol_df.drop(index=maxed_insol_problematic, inplace=True)
        insol_df.drop(index=insol_100_overlaps.intersection(insol_df), inplace=True)
        sol100_df.drop(index=insol_100_overlaps.intersection(sol100_df), inplace=True)
    return insol_df, maxed_sol_df, sol100_df


def get_models():
    exp_dir = "{}enamine_feat_selection_12-17-24/".format(os.environ.get("MODEL_DIR"))
    model_dir = "{}test_train_split/".format(exp_dir)
    """
    feature_df_path = "{}preprocessed_feature_df_backup.pkl".format(exp_dir)
    if not os.path.isfile(feature_df_path):
        print("No df")
        raise FileExistsError
    with open(feature_df_path, "rb") as f:
        feature_df = pickle.load(f)
    labels = pd.read_csv("{}member_labels.csv".format(exp_dir))
    labels.set_index(keys=labels.columns[0], inplace=True)
    """
    feature_df, labels = data_tools.load_training_data(clean=False)
    run_dirs = list(os.walk(model_dir))[0][1]
    results = list()
    for run in run_dirs:
        model_path = "{}/best_model.pkl".format(os.path.join(model_dir, run))
        subset_path = "{}/test_scores.csv".format(os.path.join(model_dir, run))
        print(run)
        if not os.path.isfile(model_path):
            print("No model: {}".format(run))
            continue
        with open(model_path, "rb") as f:
            base_model = pickle.load(f)
        test_scores, subset_list = list(), list()
        with open(subset_path) as f:
            subsets = f.readlines()
            [test_scores.append(float(s.split("\t")[0])) for s in subsets]
            [subset_list.append(s.split("\t")[1:]) for s in subsets]
        with open("{}/scaler.pkl".format(exp_dir), "rb") as f:
            scaler = pickle.load(f)
        results.append(
            [scaler, base_model, test_scores, subset_list, feature_df, labels]
        )
    return results


def get_subset_choices(score_subset, n_subsets=5):
    # Jaccard avoids high similarity metric for small, dissimilar sets.
    ranked_subsets = sorted(
        copy.deepcopy(score_subset), key=lambda x: x[0], reverse=True
    )
    print("Best recorded score: {}".format(ranked_subsets[0]))
    n_choices = min((n_subsets**2 - n_subsets), len(score_subset))
    top_choices = ranked_subsets[:n_choices]
    j_dists = 1 - bulk_rand([c[1] for c in top_choices], jaccard=True)
    j_weights = j_dists**2
    subset_choices = [
        np.argmax(np.sum(j_weights, axis=1) * np.array([c[0] for c in top_choices]))
    ]
    for i in np.arange(n_subsets - 1):
        free_choice = [i not in subset_choices for i, c in enumerate(top_choices)]
        subset_choices.append(
            (
                np.sum(j_weights[np.ix_(free_choice, subset_choices)], axis=1)
                * np.array(
                    [c[0] for i, c in enumerate(top_choices) if i not in subset_choices]
                )
            ).argmax()
        )
    return [top_choices[c] for c in subset_choices]


def score_epa_data(epa_data, train_data, model_score_subset, scoring, n_subsets=5):
    model = model_score_subset[0]
    score_subset = list(zip(model_score_subset[1], model_score_subset[2]))
    test_subsets = get_subset_choices(score_subset, n_subsets=n_subsets)
    subset_scores, subset_proba = list(), list()
    for score, subset in test_subsets:
        cols = [c.strip() for c in subset]
        for c in cols:
            for dataset in [epa_data[0], train_data[0]]:
                if c not in dataset.columns:
                    dataset.columns = find_most_similar_feat(c, tuple(dataset.columns))
        if any(
            [
                isinstance(model, m)
                for m in [
                    PassiveAggressiveClassifier,
                    RandomForestClassifier,
                    ExtraTreesClassifier,
                    SVC,
                ]
            ]
        ):
            fit_model = clone(model).set_params(class_weight="balanced")
        else:
            fit_model = clone(model).set_params(scoring=scoring)
        fit_model.fit(,
        train_score = balanced_accuracy_score(
            y_true=train_data[1][train_data[0].index].squeeze(),
            y_pred=fit_model.predict(X=train_data[0][cols]),
        )
        eval_score = cross_val_score(
            fit_model,
            train_data[0][cols],
            train_data[1],
            n_jobs=-1,
            scoring=scoring,
            error_score="raise",
        )
        # eval_pred = cross_val_predict(fit_model, train_data[0][cols], train_data[1], n_jobs=-1)  # , scoring=scoring))
        test_score = balanced_accuracy_score(
            y_true=epa_data[1][epa_data[0].index].squeeze(),
            y_pred=fit_model.predict(X=epa_data[0][cols]),
        )
        print(
            "Train: {:.4f}, Eval CV: {}, EPA: {:.4f}".format(
                train_score, ", ".join([str(s)[:4] for s in eval_score]), test_score
            )
        )
        # print(model.__repr__())
        subset_scores.append((train_score, eval_score, test_score))
        if not any(
            [
                isinstance(model, m)
                for m in [RidgeClassifierCV, PassiveAggressiveClassifier, SVC]
            ]
        ):
            subset_proba.append(fit_model.predict_proba(X=epa_data[0][cols]))
    return subset_scores, subset_proba


@cache
def find_most_similar_feat(feature_name, reference_index):
    diffcut = 0.91
    print(difflib.SequenceMatcher(a=feature_name, b=d).ratio() for d in reference_index)
    while True:
        if (
            len(
                difflib.get_close_matches(
                    feature_name, reference_index, n=1, cutoff=diffcut
                )
            )
            == 0
        ):
            diffcut -= 0.05
            if diffcut <= 0.4:
                print("No matching substring found! ", feature_name)
                closest = feature_name
                break
        else:
            closest = difflib.get_close_matches(
                word=feature_name,
                possibilities=reference_index,
                n=1,
                cutoff=diffcut,
            )[0]
            break
    print(
        feature_name,
        closest,
        [
            difflib.SequenceMatcher(a=feature_name, b=d).ratio()
            for d in reference_index[:20]
        ],
    )
    as_list = list(reference_index)
    idx = as_list.index(closest)
    as_list.insert(idx, feature_name)
    as_list.remove(closest)
    updated_index = pd.Index(as_list)
    return updated_index


def get_padel_descriptors(id_ser, desc_path, input_type="dtxsid", desc_set="padel"):
    # padel_grabber = DescriptorGrabber(desc_set="padel", input_type=input_type)
    desc_df = descriptor_processing.get_api_desc(
        desc_path=desc_path, id_ser=id_ser, d_set=desc_set, input_type=input_type
    )
    return desc_df


def main():
    # train_df, train_labels = load_training_data()
    exp_path = "{}epa_test_enamine_features/".format(os.environ.get("MODEL_DIR"))
    std_path = "{}stdizer_epa_query.csv".format(exp_path)
    desc_path = "{}padel_features_output".format(exp_path)
    epa_label_path = "{}epa_labels.csv".format(exp_path)
    transformer_path = "{}transformer.pkl".format(exp_path)
    if os.path.isfile("{}train_df.pkl".format(exp_path)) and os.path.isfile(
        epa_label_path
    ):
        epa_df = pd.read_pickle("{}train_df.pkl".format(exp_path))
        epa_labels = pd.read_csv(epa_label_path)
        epa_labels = epa_labels.set_index(keys=epa_labels.columns[0]).squeeze()
    else:
        train_df, epa_labels, insol_labels, sol100_labels, max_sol_labels = (
            get_new_epa_data(desc_path, epa_label_path, exp_path, std_path)
        )
        if not os.path.isfile("{}csv".format(desc_path)):
            train_df.to_csv("{}csv".format(desc_path), sep="\t")
        if not os.path.isfile("{}pkl".format(desc_path)):
            train_df.to_pickle("{}pkl".format(desc_path))
        labels_insol = pd.Series(
            np.zeros(insol_labels.shape[0]), index=insol_labels.index
        )
        labels_sol = pd.Series(
            np.ones(sol100_labels.shape[0]), index=sol100_labels.index
        )

        epa_labels = pd.concat([labels_insol, labels_sol])
        new_index = epa_labels.index.map(mapper=epa_map.to_dict())
        epa_labels.index = new_index
        epa_labels.drop(index=epa_labels.index[epa_labels.index.isna()], inplace=True)
        # TODO: Switch df to label and get descriptors by importing EPA data and indexing with label.
        df_diff = train_df.index.difference(epa_labels.index)
        train_df.drop(index=df_diff, inplace=True)
        label_diff = epa_labels.index.difference(train_df.index)
        epa_labels.drop(index=label_diff, inplace=True)
        epa_df = train_df.loc[epa_labels.index.intersection(train_df.index)]
        pd.Series(label_diff).to_csv("{}dropped_compounds.csv".format(exp_path))
        epa_labels.to_csv("{}epa_labels.csv".format(exp_path))
        epa_df.to_pickle("{}train_df.pkl".format(exp_path))
        print("Processing Done:")

    epa_df.columns = padel_categorization.get_full_padel_names(two_d=True)
    epa_df = epa_df.loc[~epa_df.index.duplicated()]
    epa_labels = epa_labels[~epa_labels.index.duplicated()]
    epa_df.drop(index=epa_df.index.difference(epa_labels.index), inplace=True)
    epa_labels.drop(index=epa_labels.index.difference(epa_df.index), inplace=True)
    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f).set_output(transform="pandas")
    score_df = transformer.transform(epa_df)
    feature_df, labels = load_training_data(clean=False)
    feature_df = transformer.transform(feature_df)
    # feature_df.columns = feature_df.columns._transform_index(lambda x: x.encode())
    print(epa_df.columns)
    print("Training")
    print(feature_df.columns)
    print(
        "Column differences: {}".format(
            score_df.columns.symmetric_difference(feature_df.columns)
        )
    )

    # model_subsets, train_data = model_tuple[:3], model_tuple[3:]
    insol_scores, sol100_scores, insol_proba, sol100_proba = (
        list(),
        list(),
        list(),
        list(),
    )
    score_dict = dict()
    proba_dict = dict()
    epa_score_dict = dict()
    epa_scorer = make_scorer(accuracy_score)
    # Scores list: [ insol/sol [ model_type [ subset ] ] ]
    model_tuple = get_models()
    for model_tup in model_tuple:
        mod_name = model_tup[1].__repr__().split("(")[0]
        print("\n{}".format(mod_name))
        # scaler = model_tup[0]
        # epa_df = scaler.transform(score_df)
        # print([c for c in model_tup[4].columns if not isinstance(c, str)])
        # model_tuple[4] = transformer.transform(model_tup[4].copy().astype(str))
        epa_results = score_epa_data(
            (score_df, epa_labels),
            (feature_df, labels),
            model_tup[1:4],
            scoring=epa_scorer,
            n_subsets=3,
        )
        epa_score_dict[mod_name] = epa_results[0][-1]
        score_dict[mod_name] = epa_results[0]
        proba_dict[mod_name] = epa_results[1]
        # proba_list.append(epa_results[1])
    print(pd.DataFrame.from_records(score_dict))


def max_min_epa_query(padel_path, desc_path, epa_label_path, std_path, label_list=None):
    if os.path.isfile(padel_path):
        descriptor_df = pd.read_pickle("{}.pkl".format(desc_path))
    else:
        descriptor_df, smiles_df = process_epa_data(label_list, desc_path, std_path)
    if os.path.isfile(epa_label_path):
        epa_labels = pd.read_csv(epa_label_path)
    else:
        epa_labels = None
    return descriptor_df, epa_labels


def get_new_epa_data(desc_path, epa_label_path, exp_path, std_path, exact_max=True):
    insol_labels, maxed_sol_labels, sol100_labels = get_query_data()
    insol_labels, maxed_sol_labels, sol100_labels = report_overlaps(
        insol_labels, maxed_sol_labels, sol100_labels
    )
    for labs in [insol_labels, maxed_sol_labels, sol100_labels]:
        if "dtxsid" in labs.index:
            labs.set_index(keys="dtxsid", inplace=True)
    padel_path = "{}pkl".format(desc_path)
    if exact_max:
        label_list = [maxed_sol_labels]
    else:
        label_list = [insol_labels, sol100_labels]
    train_df, epa_labels = max_min_epa_query(
        padel_path=padel_path,
        desc_path=desc_path,
        epa_label_path=epa_label_path,
        std_path=std_path,
        label_list=label_list,
    )
    return train_df, epa_labels, insol_labels, sol100_labels, maxed_sol_labels


def process_epa_data(label_list, desc_path, std_path):
    lookup_path = "{}multiclass_trial/lookup.csv".format(os.environ.get("MODEL_DIR"))
    std_pkl = "{}stdizer_epa_query.pkl".format(mc_path)
    stdizer = QsarStdizer()
    desc_list, smi_list = list(), list()
    epa_map = get_epa_mapper()  # Get data
    insol_labels, maxed_sol_labels, sol100_labels = get_query_data()
    combo_labels, convert_df = get_conversions(maxed_sol_labels, lookup_path)
    smiles_df, sid_to_key = standardize_smiles(
        convert_df, combo_labels, maxed_sol_labels, std_pkl
    )
    max_conc = combo_labels.set_index(keys="inchiKey", drop=True)[
        "sample_concentration"
    ].astype(np.float32)
    max_conc.index = max_conc.index.map(
        lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
    )
    if os.path.isfile(std_path):
        smi_df = pd.read_csv(std_path)
    else:
        for labels in label_list:
            api_ser = epa_map.loc[labels.index.intersection(epa_map.index.dropna())]
            results, returned_input = stdizer.grab_data(api_ser.to_list())
            print(results)
            results.drop(columns="mol", inplace=True)
            smi_list.append(results)
        smi_df = pd.concat(smi_list).dropna()
        smi_df.to_csv(std_path)
    print(smi_df)
    if os.path.isfile(desc_path):
        descriptor_df = pd.read_pickle(desc_path)
    else:
        descriptor_df = descriptor_processing.get_api_desc(
            desc_path="{}.csv".format(desc_path),
            id_ser=smi_df["smiles"],
            d_set="padel",
        )
        descriptor_df.index = smi_df["smiles"]
        with open(desc_path, "wb") as f:
            pickle.dump(descriptor_df, f)
    return descriptor_df, smi_df


if __name__ == "__main__":
    main()
