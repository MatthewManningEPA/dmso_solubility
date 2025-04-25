import itertools
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, make_scorer
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import build_preprocessor
import fuzzy_controller
import padel_categorization
from archive import vapor_pressure_selection
from data_cleaning import qsar_readiness
from dmso_utils.data_tools import get_query_data


def optimize_tree(feature_df, labels, model, scoring, cv):
    param_grid = {
        "max_depth": [None, 25, 20, 15],
        "min_impurity_decrease": [0, 0.005, 0.01],
        "max_leaf_nodes": [100, 150, 200, 250],
    }
    tree_optimized = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        return_train_score=True,
        error_score="raise",
    )
    tree_optimized.fit(feature_df, labels)
    print(tree_optimized.best_params_)
    return tree_optimized


def get_class_splits(labels):
    edges = [
        5.5,
        10.5,
        19,
    ]


def safe_mapper(x, map):
    if x in map.keys():
        return map[x]
    else:
        return x


def main():
    # Paths.
    epa_scorer = make_scorer(balanced_accuracy_score)
    mc_path = "{}multiclass_trial/".format(os.environ.get("MODEL_DIR"))
    std_path = "{}stdizer_epa_query.csv".format(mc_path)
    std_pkl = "{}stdizer_epa_query.pkl".format(mc_path)
    lookup_path = "{}lookup.csv".format(mc_path)
    desc_path = "{}padel_features_output_max_sol".format(mc_path)
    epa_label_path = "{}epa_labels.csv".format(mc_path)
    transformer_path = "{}transformer.pkl".format(mc_path)
    transformed_df_path = "{}transformed_epa_exact_sol_data.pkl".format(mc_path)
    # Get data
    insol_labels, maxed_sol_labels, sol100_labels = get_query_data()

    maxed_sol_labels = maxed_sol_labels[~maxed_sol_labels.index.duplicated()]
    """
    label_list = maxed_sol_labels.index.tolist()
    for i, label_i in enumerate(itertools.batched(label_list, n=1000)):
        with open("{}labels_ind_{}.csv".format(mc_path, i), "w") as f:
            f.write("DTXSID\n")
            f.write("\n".join(label_i))
    stdizer_list = list()
    for i in np.arange(27):
        smi_path = "{}labels_std_{}.csv".format(mc_path, i)
        stdizer_list.append(pd.read_csv(smi_path))
    stdized_df = pd.concat(stdizer_list)
    # Standardize Smiles.
    """
    if os.path.isfile(std_path):
        smiles_df = pd.read_csv(std_path, index_col="SMILES")
    elif os.path.isfile(std_pkl):
        smiles_df = pd.read_pickle(std_pkl)
    if os.path.isfile(transformed_df_path) and os.path.isfile(epa_label_path):
        train_df = pd.read_pickle(transformed_df_path)
        epa_labels = pd.read_csv(epa_label_path, index_col="index").squeeze()
    else:
        combo_labels, convert_df = get_conversions(maxed_sol_labels, lookup_path)
        # Filter completely invalid compounds.
        ok_compounds, forbidden_compounds = qsar_readiness(
            combo_labels["smiles"].tolist()
        )
        forbidden_df = combo_labels[combo_labels["smiles"].isin(forbidden_compounds)]
        combo_labels.drop(index=forbidden_df.index, inplace=True)
        forbidden_df.to_csv("{}not_qsar_ready.csv".format(mc_path))
        maxed_sol_labels.drop(
            index=pd.Index(forbidden_df["id"]), inplace=True, errors="ignore"
        )
        leftovers = forbidden_df[~forbidden_df["id"].isin(combo_labels.index)]
        combo_labels.drop(
            index=pd.Index(leftovers["sid"]).union(pd.Index(leftovers["id"])),
            inplace=True,
            errors="ignore",
        )
        smiles_df, sid_to_key = standardize_smiles(
            convert_df, combo_labels, maxed_sol_labels, std_pkl
        )
        epa_df = fetch_padel(smiles_df, desc_path, mc_path)
        max_conc = combo_labels.set_index(keys="inchiKey", drop=True)[
            "sample_concentration"
        ].astype(np.float32)
        max_conc.index = max_conc.index.map(
            lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
        )
        max_conc = max_conc[~max_conc.index.isna()]
        missing_desc = max_conc.index.difference(epa_df.index)
        print("Missing labelled compounds in descriptors: \n{}".format(missing_desc))
        pd.Series(missing_desc).to_csv(
            "{}compounds_with_missing_descriptors.csv".format(mc_path)
        )
        print(
            "Missing descriptor compounds in labels: \n{}".format(
                epa_df.index.difference(max_conc.index)
            )
        )
        intersect = epa_df.index.intersection(max_conc.index)
        assert intersect.size > 0

        epa_df.drop(index=epa_df.index.difference(max_conc.index))
        max_conc.drop(index=max_conc.index.difference(epa_df.index), inplace=True)
        if intersect.size > 0:
            epa_labels = cluster_labels(max_conc[intersect], mc_path)
            assert epa_labels.shape[0] > 0
        else:
            print("No clustering due to incompatible indices.")
            print([max_conc.index, epa_df.index], flush=True)
            raise ValueError
        pd.concat([epa_labels, max_conc]).to_csv(epa_label_path, index_label="index")
        """
        missing_desc = epa_labels.index.difference(epa_df.index)
        # print("Missing descriptors: \n{}".format(missing_desc))
        # print(new_smiles[new_smiles.isna()].index)
        if missing_desc.size > 0:
            missing_desc = missing_desc.dropna()
            new_smiles = (
                missing_desc.map(sid_to_smiles).to_series(name="SMILES").dropna()
            )
            if new_smiles.size == 0:
                new_smiles = (
                    missing_desc.map(smiles_to_sid).to_series(name="SMILES").dropna()
                )
                print("Trying alternate mapping.")
            print(new_smiles.size, flush=True)
            if not os.path.isfile("{}.pkl".format(desc_path)):
                missing_df = descriptor_processing.get_api_desc(
                    desc_path="{}.csv".format(desc_path),
                    id_ser=new_smiles,
                    d_set="padel",
                )
            else:
                missing_df = pd.read_csv("{}.pkl".format(desc_path))
            missing_df.index = missing_df.index.map(smiles_to_sid)
            filled_df = filled_df.dropna(axis="index").sort_index()
            filled_df = filled_df.loc[~filled_df.index.duplicated()]
            epa_labels = epa_labels[~epa_labels.index.duplicated()]
            print(filled_df.shape, epa_labels.shape)
            print(filled_df.index.symmetric_difference(epa_labels.index).size, flush=True)
            assert filled_df.index.symmetric_difference(epa_labels.index).size == 0
            filled_df.columns = padel_categorization.get_short_padel_names()
        else:
            filled_df = epa_df
        """

        preprocessor_params = {
            "xc_method": "kendall",
            "corr_method": "kendall",
            "thresh_xc": 0.95,
        }
        preprocessor, p = build_preprocessor.get_standard_preprocessor(
            transform_func="asinh", corr_params=preprocessor_params
        )
        train_df = preprocessor.fit_transform(epa_df, epa_labels)
        train_df.to_pickle(transformed_df_path)
        with open(transformer_path, "wb") as f:
            pickle.dump(preprocessor, f)
    """
    tree_search = optimize_tree(
        train_df,
        epa_labels,
        model=RandomForestClassifier(bootstrap=False, class_weight="balanced", random_state=0),
        scoring=epa_scorer,
        cv=RepeatedStratifiedKFold(n_repeats=3, random_state=0),
    )
    print(tree_search.best_params_)
    print(tree_search.best_score_)
    best_tree = tree_search.best_estimator_
    """
    cv_results = train_multilabel_models(train_df, epa_labels, epa_scorer, mc_path)
    print(cv_results)
    return train_df, epa_labels, cv_results


def fetch_padel(smiles_df, desc_path, mc_path):
    epa_df = pd.DataFrame([])
    if os.path.isfile("{}.pkl".format(desc_path)):
        epa_df = pd.read_pickle("{}.pkl".format(desc_path))
        # epa_df = epa_df[~epa_df.index.duplicated()]
    overage = epa_df.index.difference(smiles_df.index)
    epa_df.drop(overage, inplace=True)
    missing_smiles = smiles_df.index.difference(epa_df.index).dropna()
    new_smiles = smiles_df["smiles"].squeeze()[missing_smiles]
    """
            new_smiles = missing_smiles.intersection(
                smiles_df.index.map(sid_to_key)
            ).dropna()
            """
    # missing_smiles = different_smiles.difference(smiles_df.index)
    if new_smiles.size > 0:
        if not os.path.isfile("{}desc_api_out.pkl".format(mc_path)):
            desc_df, info_df, failed_list = get_descriptors(
                smi_list=new_smiles.tolist()
            )
            assert not desc_df.empty or len(failed_list) > 0
            desc_df.to_pickle("{}desc_api_out.pkl".format(mc_path))
        else:
            desc_df = pd.read_pickle("{}desc_api_out.pkl".format(mc_path))
        if not desc_df.empty and not epa_df.empty:
            if desc_df.shape[1] == epa_df.shape[1]:
                epa_df = pd.concat([epa_df, desc_df])
            else:
                print(
                    "epa_df and desc_df are different shapes[1]:\n{}\n{}".format(
                        epa_df.shape, desc_df.shape
                    )
                )
                epa_df = desc_df
            epa_df = epa_df[~epa_df.index.duplicated()]
        elif epa_df.empty and not desc_df.empty:
            epa_df = desc_df
        elif not epa_df.empty:
            print("desc_df is empty!!!")
            pass
        else:
            raise ValueError
        epa_df.columns = padel_categorization.get_short_padel_names().astype(str)
        epa_df.dropna(how="all", axis=1, inplace=True)
        epa_df.dropna(how="any", inplace=True)
        epa_df.to_pickle("{}.pkl".format(desc_path))
    else:
        epa_df.dropna(how="all", axis=1, inplace=True)
        epa_df.dropna(how="any", inplace=True)

        """            
        desc_dict = dict()
        with open("{}.csv".format(desc_path), "a") as fo:
            last_ind = 0
            for response, api_input in grabber.bulk_epa_call(
                new_smiles.tolist()[last_ind:]
            ):
                if isinstance(response, dict) and "descriptors" in response.keys():
                    desc_dict[response["inchiKey"]] = response["descriptors"]
                    fo.write(
                        "{}\n".format(
                            "\t".join([str(s) for s in response.values()])
                        )
                    )
                else:
                    print("Featurizer failed for: ", api_input, response)
                    fo.write(
                        "{}\n".format(
                            "\t".join([str(s) for s in response.values()])
                        )
                    )
        new_descriptors_df = pd.DataFrame.from_dict(
            desc_dict,
            orient="index",
            columns=padel_categorization.get_short_padel_names(),
        )          
        if isinstance(epa_df.index, pd.RangeIndex) and os.path.isfile(
            "{}_smiles.pkl".format(desc_path)
        ):
            with open("{}_smiles.pkl".format(desc_path), "rb") as f:
                smiles_df = pickle.load(f)
            epa_df.index = smiles_df["inchiKey"].to_list()
            smiles_df.set_index(keys="inchiKey", inplace=True)
            if "Unnamed: 0" in smiles_df.columns:
                smiles_df.drop(columns=["Unnamed: 0"], inplace=True)
        if (
            epa_df.shape[0] < 0.98 * epa_labels.shape[0]
            or isinstance(epa_df.index, pd.RangeIndex)
        ) and os.path.isfile("{}.csv".format(desc_path)):
            epa_df = pd.read_csv("{}.csv".format(desc_path), sep="\t")
        if (
            epa_df.shape[0] < 0.98 * epa_labels.shape[0]
            or isinstance(epa_df.index, pd.RangeIndex)
        ) and os.path.isfile("{}.pkl".format(desc_path)):
            print("EPA DF index is range...")
            print("Pickle loaded.")
            epa_df = pd.read_pickle("{}.pkl".format(desc_path))
        if epa_df.shape[0] < 0.98 * epa_labels.shape[0] or isinstance(
            epa_df.index, pd.RangeIndex
        ):
            epa_df, smiles_df = process_epa_data(
                [epa_labels.index.tolist], "{}.pkl".format(desc_path), std_path
            )
            if (
                smiles_df["inchiKey"][
                    smiles_df.index.intersection(epa_df.index)
                ].size
                > 0
            ):
                epa_df.index = smiles_df["inchiKey"][
                    smiles_df.index.intersection(epa_df.index)
                ]
            print("process_epa_data")
            print(smiles_df)
            with open("{}.pkl".format(desc_path), "wb") as f:
                pickle.dump(epa_df, f)
            smiles_df.set_index(keys="inchiKey", inplace=True)
            if "Unnamed: 0" in smiles_df.columns:
                smiles_df.drop(columns=["Unnamed: 0"], inplace=True)
            with open("{}_smiles.pkl".format(desc_path), "wb") as f:
                pickle.dump(smiles_df, f)
        else:
            print("Transformed loaded")
            """
    return epa_df


def get_conversions(maxed_sol_labels, lookup_path):
    if os.path.isfile(lookup_path):
        convert_df = pd.read_csv(lookup_path)
        if True:
            convert_df.drop(
                columns=[x for x in convert_df.columns if "Unnamed" in str(x)],
                inplace=True,
            )
            convert_df.drop(columns="mol", inplace=True, errors="ignore")
            convert_df.to_csv(lookup_path)
    else:
        convert_df = lookup_chem(comlist=maxed_sol_labels.index.tolist())
        convert_df.drop(columns="mol", inplace=True, errors="ignore")
        convert_df.to_csv(lookup_path)
        convert_df.drop(
            columns=["bingoId", "name", "casrn", "formula", "cid"],
            inplace=True,
            errors="ignore",
        )
    combo_labels = pd.concat(
        [convert_df, maxed_sol_labels.reset_index(names="label_id")], axis=1
    )
    failed_lookup = combo_labels[combo_labels["id"] != combo_labels["label_id"]]
    combo_labels.drop(index=failed_lookup.index, inplace=True)
    nosmiles = combo_labels[
        (combo_labels["smiles"] == "nan") | combo_labels["smiles"].isna()
    ]
    combo_labels.drop(index=nosmiles.index, inplace=True)
    return combo_labels, convert_df


def train_multilabel_models(epa_df, epa_labels, epa_scorer, mc_path):
    best_tree = RandomForestClassifier(
        bootstrap=False,
        max_leaf_nodes=100,
        min_impurity_decrease=0.005,
        class_weight="balanced",
        random_state=0,
    )
    lrcv = LogisticRegressionCV(
        scoring=epa_scorer,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    )
    select_params = {
        "corr_method": "kendall",
        "xc_method": "kendall",
        "max_features_out": 15,
        "fails_min_vif": 0,
        "fails_min_perm": 6,
        "fails_min_sfs": 0,
        "features_min_vif": 20,
        "features_min_perm": 20,
        "features_min_sfs": 15,
        "thresh_reset": 0.025,
        "thresh_vif": 10,
        "thresh_perm": 0.0025,
        "thresh_sfs": -0.005,
        "thresh_xc": 0.95,
        "max_trials": 50,
        "cv": 5,
        "importance": False,
        "scoring": make_scorer(balanced_accuracy_score),
    }
    from sklearn.ensemble import GradientBoostingClassifier

    gbc_default = GradientBoostingClassifier()
    ovo_tree = OneVsOneClassifier(estimator=best_tree)
    ovo_lr = OneVsOneClassifier(estimator=lrcv)
    ovr_tree = OneVsRestClassifier(estimator=best_tree)
    ovr_lr = OneVsRestClassifier(estimator=lrcv)
    model_list = [ovo_lr, gbc_default, ovr_lr, ovo_tree]
    # model_list = [best_tree, lrcv]
    name_list = ["ovo_logit", "gbc_default", "ovr_logit", "ovo_tree"]
    # name_list = ["Random_Forest", "Logistic"]
    intersect_final = epa_labels.index.intersection(epa_df.index)
    epa_labels = epa_labels[intersect_final]
    epa_df = epa_df.loc[intersect_final]
    print(epa_labels)
    print(epa_df.shape)

    train_labels, test_labels = train_test_split(
        epa_labels, test_size=0.2, random_state=0, shuffle=True, stratify=epa_labels
    )
    train_df = epa_df.loc[train_labels.index]
    test_df = epa_df.loc[test_labels.index]
    search_features = vapor_pressure_selection.padel_candidate_features()
    print("{} features to select from.".format(len(search_features)))
    short_to_long = padel_categorization.padel_convert_length().to_dict()
    # test_padel_conversion(search_features, short_to_long)
    pd.DataFrame.corrwith()
    best_corrs = train_df[search_features].corrwith(
        train_labels, method=select_params["corr_method"]
    )
    cross_corr = train_df[search_features].corr(method=select_params["xc_method"])
    cv_dict = dict()
    for m, n in zip(model_list, name_list):
        selection_models = {
            "predict": m,
            "permutation": m,
            "importance": m,
            "vif": linear_model.LinearRegression(),
        }
        model_dir = "{}feature_selection_metaestimator_trial/{}/".format(mc_path, n)
        os.makedirs(model_dir, exist_ok=True)
        model_dict, score_dict, dropped_dict, best_features = (
            fuzzy_controller.select_feature_subset(
                train_df[search_features],
                train_labels,
                target_corr=best_corrs,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_models=selection_models,
                save_dir=model_dir,
            )
        )
        print("Best features!")
        best_features_long = [short_to_long[f] for f in best_features]
        print("\n".join(best_features_long))
        pd.Series(best_features_long).to_csv(
            "{}best_features_{}.csv".format(model_dir, n)
        )
        cv_dict[n] = cross_validate(
            m,
            train_df[best_features],
            train_labels,
            scoring=epa_scorer,
            cv=RepeatedStratifiedKFold(n_repeats=3, random_state=0),
            n_jobs=-1,
            return_train_score=True,
            return_indices=True,
        )["test_score"]
        cmd = ConfusionMatrixDisplay.from_estimator(m, X=test_df, y=test_labels)
        cmd.plot()
        cmd.figure_.savefig("{}confusion_matrix.png".format(model_dir))
    cv_results = pd.DataFrame.from_dict(cv_dict)
    print(cv_results, flush=True)
    ax = sns.catplot(cv_results)
    ax.savefig("{}test_scores.png".format(mc_path))
    plt.show()
    return cv_results


def test_padel_conversion(feature_list, convert_dict):
    for f in feature_list:
        print(f, convert_dict[f])
    return


def cluster_labels(labels, mc_path):
    trimmed_labels = labels.clip(upper=101.0, lower=4.5).apply(np.asinh)
    trimmed_labels.to_csv("{}epa_transformed.csv".format(mc_path))
    print(
        "Asinh: {:.4f} {:.4f} {:.4f} {:.4f}".format(
            np.asinh(5), np.asinh(10), np.asinh(50), np.asinh(100)
        )
    )
    agg_clusters = AgglomerativeClustering(
        n_clusters=4, linkage="single", metric="cosine"
    ).fit_predict(trimmed_labels.to_frame())
    agg_clusters = pd.Series(agg_clusters, index=trimmed_labels.index)
    for n in agg_clusters.sort_values().unique():
        cluster = trimmed_labels.sort_values()[agg_clusters == n]
        print(np.sinh(cluster.min()), np.sinh(cluster.max()))
    """
    label_clusterer = (
        KMeans(n_clusters=6, random_state=0)
        .set_output(transform="pandas")
        .fit(trimmed_labels.to_frame())
    )
    epa_labels = pd.Series(label_clusterer.predict(trimmed_labels.to_frame()), index=trimmed_labels.index)
    """
    epa_labels = agg_clusters
    print("Cluster-assigned labels")
    print(np.sort(epa_labels.value_counts(normalize=True)), flush=True)
    # epa_df.index = epa_df.index.map(smiles_to_sid)
    # print("EPA DF Index:\n{}".format(epa_df.index))
    return epa_labels


def standardize_smiles(feature_df, combo_labels, maxed_sol_labels, std_pkl=None):
    if os.path.isfile(std_pkl):
        smiles_df = pd.read_pickle(std_pkl)
        if False:
            clean_list = [x for x in smiles_df.tolist() if "sid" in x.keys()]
            unregistered_list = [
                x
                for x in smiles_df.to_dict()
                if "sid" not in x.keys() and "smiles" in x.keys()
            ]
            failed_list = [x for x in smiles_df.tolist() if "smiles" not in x.keys()]
            clean_df = pd.json_normalize(clean_list)
            unregistered_df = pd.json_normalize(unregistered_list)
            smiles_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
            failed_df = pd.json_normalize(failed_list)
            if not failed_df.empty:
                failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
            assert not smiles_df.empty
            if std_pkl is not None:
                smiles_df.to_pickle(std_pkl)
            smiles_df.to_csv(std_path)
            if not os.path.isfile("{}failed_standardizer.csv".format(failed_df)):
                # TODO: Extract failed compounds by difference of inputs and outputs.
                pass

    else:
        smiles_df, failed_df = get_standardizer(comlist=feature_df["smiles"].tolist())
        if not failed_df.empty:
            failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
        assert not smiles_df.empty
        if std_pkl is not None:
            smiles_df.to_pickle(std_pkl)
    smiles_df.drop(
        columns=[
            "cid",
            "casrn",
            "name",
            "canonicalSmiles",
            "inchi",
            "mol",
            "formula",
            "id",
        ],
        inplace=True,
        errors="ignore",
    )
    assert not smiles_df.empty
    smiles_df = smiles_df[smiles_df["chemId"].isin(combo_labels["id"])]
    """
        if "smiles" in smiles_df.columns:
            smiles_df = smiles_df[~smiles_df.duplicated() & ~smiles_df["smiles"].isna()]
            # smiles_df.set_index(keys="sid", drop=True, inplace=True)
            print("Standardizer output:\n{}".format(smiles_df))
            print(smiles_df.columns)
            sid_to_smiles = smiles_df["smiles"].squeeze().to_dict()
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_ind_df = smiles_df.reset_index().set_index(keys="smiles")
            smiles_to_sid = smiles_ind_df["sid"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.values()
            )
        else:
            smiles_fails = maxed_sol_labels.index
        # Get Descriptors.
        print("Raw label to InchiKey failures:\n{}".format(smiles_fails))
        raw_retry_df, retry_failures_df = get_standardizer(smiles_fails.tolist())
        if not raw_retry_df.empty and "smiles" in raw_retry_df.columns:
            smiles_df = pd.concat([smiles_df, raw_retry_df[~raw_retry_df["smiles"].isna()]])
            smiles_df.drop(index=smiles_df[~smiles_df["error"].isna()].index, inplace=True)
            smiles_df = smiles_df[~smiles_df.index.duplicated()]
            smiles_df.to_csv("{}updated_std_output.csv".format(mc_path))
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.keys()
            )
            print("Retry label to InchiKey failures:\n{}".format(smiles_fails))
            # print(smiles_df.loc[smiles_fails])
        else:
            print("No return from retrying Standardizer on failed labels.")
            if "smiles" in smiles_df.columns:
                smiles_df.to_csv("{}updated_std_output.csv".format(mc_path))
        smiles_fails.difference(smiles_df.index).to_series().to_csv(
            "{}failed_standardization.csv".format(mc_path)
        )
        maxed_sol_labels.index = maxed_sol_labels.index.map(sid_to_key, na_action="ignore")
        """
    sid_to_key = (
        smiles_df[["inchiKey", "chemId"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    id_to_orig_smi = (
        smiles_df[["chemId", "smiles"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    smiles_df.set_index(keys="inchiKey", drop=True, inplace=True)
    return smiles_df, sid_to_key


def lookup_chem(comlist, batch_size=100, id_type="sid", prefix="DTXSID"):
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/search/download/properties"
    response_list = list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        req = requests.Request(method="POST", url=api_url)
        for c_batch in itertools.batched(comlist, n=batch_size):
            req_json = {"ids": [], "format": id_type}
            if prefix is not None:
                c_batch = [
                    c.removeprefix(prefix) for c in c_batch if prefix is not None
                ]
            for c in c_batch:
                req_json["ids"].append({"id": c, "sim": 0})
            response = r.request(
                method="POST", url=api_url, json=req_json, headers=auth_header
            )
            response_list.extend(response.json())
    response_df = pd.json_normalize(response_list)
    return response_df


def get_standardizer(comlist, batch_size=100, input_type="smiles"):
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/stdizer"
    response_list, failed_list = list(), list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        for c in comlist:
            params = {"workflow": "qsar-ready", input_type: c}
            response = r.request(
                method="GET", url=api_url, params=params, headers=auth_header
            )
            if isinstance(response.json(), list) and len(response.json()) > 0:
                response_list.append(response.json()[0])
            else:  # isinstance(response, (list, str)):
                failed_list.append(response.content)
    clean_list = [x for x in response_list if "sid" in x.keys()]
    unregistered_list = [
        x for x in response_list if "sid" not in x.keys() and "smiles" in x.keys()
    ]
    # failed_list = [x for x in response_list if "smiles" not in x.keys()]
    clean_df = pd.json_normalize(clean_list)
    unregistered_df = pd.json_normalize(unregistered_list)
    response_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
    failed_df = pd.json_normalize(failed_list)
    return response_df.drop(columns="mol"), failed_df


def get_descriptors(smi_list, batch_size=100, desc_type="padel", input_type="smiles"):
    # TODO: Add original SMILES/identifier to info_df to link original data and descriptor data through info_df.
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/padel"
    """
    {
        "chemicals": ["string"],
        "options": {
            "headers": true,
            "timeout": 0,
            "compute2D": true,
            "compute3D": true,
            "computeFingerprints": true,
            "descriptorTypes": ["string"],
            "removeSalt": true,
            "standardizeNitro": true,
            "standardizeTautomers": true,
        },
    }
   """
    info_list, desc_dict, failed_list = list(), dict(), list()
    with requests.session() as r:
        auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
        req = requests.Request(method="GET", url=api_url, headers=auth_header).prepare()
        for c in smi_list:
            params = {"smiles": c, "type": "padel"}
            req.prepare_url(url=api_url, params=params)
            response = r.send(req)
            if response.status_code == 200 and len(response.json()) > 0:
                single_info_dict = dict(
                    [
                        (k, v)
                        for k, v in response.json()["chemicals"][0].items()
                        if k != "descriptors"
                    ]
                )
                single_info_dict.update([("SMILES_QSAR", c)])
                info_list.append(single_info_dict)
                desc_dict[response.json()["chemicals"][0]["inchiKey"]] = (
                    response.json()["chemicals"][0]["descriptors"]
                )
            else:
                failed_list.append(response.content)
    padel_names = padel_categorization.get_short_padel_names()
    info_df = pd.json_normalize(info_list)
    info_df.set_index(keys="inchiKey", inplace=True)
    info_df.drop(columns=padel_names, inplace=True, errors="ignore")
    desc_df = pd.DataFrame.from_dict(
        data=desc_dict, orient="index", columns=padel_names
    )
    return desc_df, info_df, failed_list


if __name__ == "__main__":
    logger = logging.getLogger(name="selection")
    main()
