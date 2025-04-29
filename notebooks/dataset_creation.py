import os
import pickle
import pprint
from collections import defaultdict
from numbers import Number

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_y

import build_preprocessor
import correlation_filter
import data_cleaning
import padel_categorization
from archive.grove_feature_selection import padel_candidate_features
from correlation_filter import get_correlations
from descriptor_processing import get_api_descriptors, get_standardizer
from dmso_utils.data_tools import get_conversions, get_query_data
from samples import safe_mapper


def main():
    pass


if __name__ == "__main__":
    main()


def assemble_dmso_dataset(
    dataset_name, select_params, path_dict, frac_enamine=None, rus=False
):
    if rus:
        postfix = "{:.0f}-rus".format(frac_enamine * 100)
    else:
        postfix = "{:.0f}".format(frac_enamine * 100)
    path_dict["prepped_train_labels"] = path_dict["prepped_train_labels"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    path_dict["prepped_test_labels"] = path_dict["prepped_test_labels"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    path_dict["prepped_train_df_path"] = path_dict["prepped_train_df_path"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    path_dict["prepped_test_df_path"] = path_dict["prepped_test_df_path"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    path_dict["corr_path"] = path_dict["corr_path"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    path_dict["xc_path"] = path_dict["xc_path"].replace(
        ".csv", "_{}.csv".format(postfix)
    )
    (train_df, train_labels, test_df, test_labels) = combine_enamine_new_epa_data(
        frac_enamine, select_params, path_dict, rus=True
    )
    drop_p_and_f_features = True
    # Descriptor space.
    print("Ring descriptors: \n{}".format(c for c in train_df.columns if "Ring" in c))
    # Get Candidate Features and Models.
    # if os.path.isfile(path_dict["search_features_path"]):
    #    search_features = pd.read_csv(path_dict["search_features_path"]).index.tolist()
    # candidate_features = train_df.columns.tolist()
    candidate_features = padel_candidate_features()
    candidate_features = candidate_features + ["nG8Ring", "nG8HeteroRing"]
    search_features = train_df.columns[train_df.columns.isin(candidate_features)]
    search_features.to_series().to_csv(
        "{}candidate_features.csv".format(path_dict["parent_dir"])
    )
    # search_features = search_features.drop(["SsSH"], errors="ignore")
    print(
        "Ring features: {}".format(
            [f for f in search_features if "Ring" in f and f in train_df.columns]
        )
    )
    # Phosphorus is over-represented in EPA data and may bias results if organophosphates tested were more/less soluble.
    if drop_p_and_f_features:
        print("Dropping phosphorus and flourine features")
        for f in search_features:
            if "sP" in f or "nP" == f or "nf" == f or "sf" in f:
                search_features = search_features.drop(f)
    train_df = train_df[search_features]

    return train_df, train_labels, test_df, test_labels


def assemble_dataset(dataset_name):
    test_df = None
    test_labels = None
    preprocessor = None
    if dataset_name == "skclass":
        parent_dir = "{}make_class/".format(os.environ.get("MODEL_DIR"))
        data_dir = parent_dir
        classifier = True
        train_arr, train_labels_arr = make_classification(
            n_samples=10000,
            n_features=100,
            n_informative=30,
            n_redundant=50,
            n_classes=2,
            n_clusters_per_class=2,
            class_sep=1.1,
            hypercube=False,
        )
        train_df = pd.DataFrame(
            train_arr,
            index=[str(a) for a in np.arange(train_arr.shape[0])],
            columns=[str(a) for a in np.arange(train_arr.shape[1])],
        )
        train_labels = pd.Series(train_labels_arr, index=train_df.index)
    if dataset_name == "lowe_aqua":
        classifier = False
        parent_dir = "{}lowe_aqua/".format(os.environ.get("MODEL_DIR"))
        data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/water_solubility/"
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        excel_train_data = pd.read_excel(
            "{}NIHMS1895627-supplement-Table1.xlsx".format(data_dir),
            sheet_name="S1. Training Data",
        )
        smiles_std = excel_train_data["Standardized_SMILES"]
        train_labels = excel_train_data["median_WS"]
        train_df = get_api_descriptors(smiles_std, desc_path=data_dir)

    elif dataset_name == "curated_aqua":
        classifier = False
        parent_dir = "{}curated_aqua/".format(os.environ.get("MODEL_DIR"))
        data_dir = "{}curated_aqua/".format(os.environ.get("MODEL_DIR"))
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        data = pd.read_csv(
            "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/curated-solubility-dataset.csv",
            index_col="ID",
        )
        data.drop(columns=["Name", "InChI", "InChIKey", "SMILES"], inplace=True)
        train_labels = data["Solubility"].copy().squeeze()
        train_df = data.drop(columns=["Solubility", "SD", "Ocurrences", "Group"])
    else:
        raise ValueError
    train_data = train_df, train_labels
    test_data = test_df, test_labels
    return train_data, test_data, preprocessor, data_dir, parent_dir


def load_new_dmso_data(data_dir, dataset, select_params):
    test_df = None
    test_labels = None
    preprocessor = None
    if dataset == "epa":
        parent_dir = "{}all_features/80_enamine_only/".format(
            os.environ.get("MODEL_DIR")
        )
        path_dict = _set_paths(parent_dir, data_dir, select_params)
        train_df, train_labels, test_df, test_labels, best_corrs = _get_solubility_data(
            path_dict,
            select_params,
            conc_splits=(9.9),
            preprocess=False,
            save_results=False,
        )
        # Get data
        max_conc, sid_to_inchikey_mapper = _get_unprocess_new_epa_data(path_dict)
        raw_df = _get_new_epa_padel()
        # Discretize labels and map
        epa_labels = label_solubility_clusters(
            max_conc[raw_df.index], conc_splits=(9.9)
        )
        raw_df.index = raw_df.index.map(
            lambda x: safe_mapper(x, sid_to_inchikey_mapper), na_action="ignore"
        )
        X = pd.concat([train_df, test_df])
        y = pd.concat([train_labels, test_labels])
    elif dataset == "enamine":
        data_dir = "{}all_features/80_enamine_only/mcc/".format(
            os.environ.get("MODEL_DIR")
        )
        parent_dir = (
            "{}response_weighted/enamine_only_models/search_features_3/".format(
                os.environ.get("MODEL_DIR")
            )
        )
        train_df = pd.read_pickle("{}split_0/combo_train_df.pkl".format(data_dir))
        train_labels = pd.read_csv("{}split_0/combo_train_labels.csv".format(data_dir))
        train_labels.set_index(keys="INCHI_KEY", inplace=True)
        train_labels = train_labels.squeeze()

    else:
        parent_dir = "{}all_features/80_enamine_only/".format(
            os.environ.get("MODEL_DIR")
        )
        path_dict = _set_paths(parent_dir, data_dir, select_params)

    """
    # Set paths.
    path_dict = _set_paths(parent_dir, data_dir, select_params)
    path_dict["xc_path"].replace(".pkl", "_{}.pkl".format(select_params["xc_method"]))
    path_dict["corr_path"].replace(
        ".pkl", "_{}.pkl".format(select_params["corr_method"])
    )
    """
    return X, y, preprocessor


def combine_enamine_new_epa_data(frac_enamine, select_params, path_dict, rus=False):
    if frac_enamine < 1.0:
        epa_padel_df = _get_new_epa_padel()
        print(epa_padel_df.shape)
        epa_padel_df.dropna(inplace=True)
        print(epa_padel_df.shape)
        # epa_train_labels, epa_test_labels = _get_new_epa_data_split()
        epa_train_idx = pd.read_pickle(
            "{}new_epa_data/prepped_train_labels.pkl".format(
                os.environ.get("MODEL_DIR")
            )
        )
        epa_test_idx = pd.read_pickle(
            "{}new_epa_data/prepped_test_labels.pkl".format(os.environ.get("MODEL_DIR"))
        )
        epa_raw_labels = pd.read_csv(
            "{}new_epa_data/epa_max_conc_cleaned.csv".format(
                os.environ.get("MODEL_DIR")
            ),
            index_col=0,
        ).squeeze()
        threshold_labels = epa_raw_labels.where(cond=lambda x: x < 9.9, other=0).where(
            cond=lambda x: x >= 9.9, other=1
        )
        pprint.pp(threshold_labels)
        epa_train_labels = threshold_labels.loc[epa_train_idx.index].copy()
        epa_test_labels = threshold_labels.loc[epa_test_idx.index].copy()
        epa_raw_train_df = epa_padel_df.loc[epa_train_labels.index].copy()
        epa_raw_test_df = epa_padel_df.loc[epa_test_labels.index].copy()
        """
        (epa_raw_train_df, epa_train_labels, epa_raw_test_df, epa_test_labels, _, _) = (
            _get_solubility_data(
                path_dict,
                select_params,
                conc_splits=(9.9),
                preprocess=False,
                from_disk=False,
                save_results=False,
            )
        )"""
    if frac_enamine > 0.0:
        # loaded_data = _get_prepped_enamine(select_params, path_dict)
        # if loaded_data is None:
        # print("Previous saved data not loaded. Using old method")
        padel_converter = padel_categorization.padel_convert_length(
            short_to_long=False, three_d=False
        ).to_dict()
        # enamine_df, enamine_labels = data_tools.load_split_data(clean=False)
        enamine_df = pd.read_pickle(
            "{}/enamine_only_data/enamine_only_df.pkl".format(
                os.environ.get("MODEL_DIR")
            )
        )
        enamine_labels = (
            pd.read_csv(
                "{}/enamine_only_data/enamine_labels.csv".format(
                    os.environ.get("MODEL_DIR")
                )
            )
            .set_index(keys="INCHI_KEY")
            .squeeze()
        )
        enamine_df = data_cleaning.remove_duplicate_idx(enamine_df)
        enamine_labels = data_cleaning.remove_duplicate_idx(enamine_labels)
        print([c for c in enamine_df.columns if c not in padel_converter.keys()])
        padel_mapper = defaultdict()
        enamine_df.columns = [
            padel_converter[c] if c in padel_converter.keys() else c
            for c in enamine_df.columns
        ]
        # if len(enamine_labels.shape) > 1 and enamine_labels.shape[1] > 0:
        #     enamine_labels.drop(columns=enamine_labels.columns[1], inplace=True)
    if 0.0 < frac_enamine < 1.0:
        print("\nUsing mixed EPA-Enamine data\n")
        df_overlap = enamine_df.index.intersection(
            epa_raw_train_df.index.union(epa_raw_test_df.index)
        )
        if df_overlap.size > 0:
            print("Overlap between Enamine and EPA data:")
            pprint.pp(df_overlap)
            enamine_df.drop(index=df_overlap, inplace=True)
        if rus:
            pprint.pp(enamine_df)
            pprint.pp([c for c in enamine_df.columns if not isinstance(c, str)])
            full_df, full_labels = RandomUnderSampler(random_state=0).fit_resample(
                X=enamine_df, y=enamine_labels.squeeze()
            )
        else:
            full_df, full_labels = enamine_df, enamine_labels
    (
        enamine_train_df,
        enamine_test_df,
        enamine_train_labels,
        enamine_test_labels,
    ) = train_test_split(
        full_df,
        full_labels,
        test_size=0.2,
        random_state=0,
        stratify=full_labels,
    )
    pprint.pp(enamine_train_df)
    pprint.pp(enamine_train_labels)
    """
    print(enamine_train_df.index.intersection(epa_raw_train_df.index))
    print(enamine_train_df.index.symmetric_difference(epa_raw_train_df.index))
    print(enamine_train_df.shape, epa_raw_train_df.shape)
    print(enamine_test_df.columns.symmetric_difference(epa_raw_train_df.columns))
        enamine_train_df.columns = enamine_df.columns.copy()
        enamine_test_df.columns = enamine_df.columns.copy()
        if False:
            print("Loading from loaded_data")
            (
                enamine_train_df,
                enamine_train_labels,
                enamine_test_df,
                enamine_test_labels,
                best_corrs,
                cross_corr,
                path_dict,
            ) = loaded_data
        if len(enamine_train_labels.shape) > 1 and enamine_train_labels.shape[1] > 0:
            enamine_train_labels = (
                enamine_train_labels[enamine_train_labels.columns[0]].squeeze().copy()
            )
        if len(enamine_test_labels.shape) > 1 and enamine_test_labels.shape[1] > 0:
            enamine_test_labels.drop(
                columns=enamine_test_labels.columns[1], inplace=True
            )
    """
    if frac_enamine == 1.0:
        print("Using Enamine-only data.")
        train_df, train_labels = enamine_train_df, enamine_train_labels
        test_df, test_labels = enamine_test_df, enamine_test_labels
    elif frac_enamine == 0.0:
        print("\nUsing EPA-only data")
        train_df = epa_raw_train_df
        test_df = epa_raw_test_df
        train_labels = epa_train_labels
        test_labels = epa_test_labels
    else:
        for lbl in [
            enamine_train_labels,
            epa_train_labels,
            enamine_test_labels,
            epa_test_labels,
        ]:
            lbl = lbl.squeeze().rename("Solubility")
        train_df = pd.concat([epa_raw_train_df, enamine_train_df])
        test_df = pd.concat([epa_raw_test_df, enamine_test_df])
        train_labels = pd.concat([enamine_train_labels, epa_train_labels])
        test_labels = pd.concat([enamine_test_labels, epa_test_labels])
    return train_df, train_labels, test_df, test_labels


def preprocess_data(train_data, test_data, select_params, path_dict=None):
    train_raw_df, train_raw_labels = train_data
    test_raw_df, test_raw_labels = test_data
    train_arr, train_label_arr = check_X_y(
        train_raw_df,
        train_raw_labels,
        copy=True,
        ensure_all_finite=True,
    )
    test_arr, test_label_arr = check_X_y(
        test_raw_df,
        test_raw_labels,
        copy=True,
        ensure_all_finite=True,
    )
    test_checked_df, test_labels = pd.DataFrame(
        test_arr, index=test_raw_df.index, columns=test_raw_df.columns
    ), pd.Series(test_label_arr, index=test_raw_labels.index)
    train_checked_df = pd.DataFrame(
        train_arr, index=train_raw_df.index, columns=train_raw_df.columns
    )
    train_labels = pd.Series(train_label_arr, index=train_raw_labels.index)
    print(train_labels.value_counts())
    preprocessor, scaler = build_preprocessor.get_standard_preprocessor(
        train_raw_df,
        scaler=StandardScaler(),
        transform_func=np.asinh,
        corr_params=select_params,
    )
    # Fixed NA in columns
    train_checked_df.drop(
        columns=train_checked_df.columns[train_checked_df.columns.isna()],
        inplace=True,
    )
    test_checked_df.drop(
        columns=test_checked_df.columns[test_checked_df.columns.isna()],
        inplace=True,
    )
    # sklearn.pipeline.Pipeline().fit_transform
    train_df = preprocessor.fit_transform(X=train_checked_df, y=train_labels)
    test_df = preprocessor.transform(X=test_checked_df)
    best_corrs = correlation_filter.calculate_correlation(
        train_df, train_labels, method=select_params["corr_method"]
    )
    cross_corr = correlation_filter.calculate_correlation(
        train_df, method=select_params["xc_method"]
    )
    if path_dict is not None:
        with open("{}combo_preprocessor.pkl".format(path_dict["exp_dir"]), "wb") as f:
            pickle.dump(preprocessor, f)
        train_labels.to_csv(path_dict["prepped_train_labels"])
        test_labels.to_csv(path_dict["prepped_test_labels"])
        train_df.to_pickle(path_dict["prepped_train_df_path"])
        test_df.to_pickle(path_dict["prepped_test_df_path"])
        best_corrs.to_pickle(path_dict["corr_path"])
        cross_corr.to_pickle(path_dict["xc_path"])
    return (
        train_df,
        train_labels,
        test_df,
        test_labels,
        preprocessor,
        best_corrs,
        cross_corr,
    )


def _get_prepped_enamine(select_params, path_dict, split_dir=None, dataset=None):
    if dataset is not None:
        path_dict["prepped_train_df_path"] = "{}{}_train_df.pkl(".format(
            split_dir,
            dataset,
        )
        path_dict["prepped_test_df_path"] = "{}{}_test_df.pkl".format(
            split_dir,
            dataset,
        )
        path_dict["prepped_train_labels"] = "{}{}_train_labels.csv".format(
            split_dir,
            dataset,
        )
        path_dict["prepped_test_labels"] = "{}{}_test_labels.csv".format(
            split_dir,
            dataset,
        )
        path_dict["corr_path"] = "{}{}_label_corr_{}.pkl".format(
            split_dir, dataset, select_params["corr_method"]
        )
        path_dict["xc_path"] = "{}{}_cross_corr_{}.pkl".format(
            split_dir, dataset, select_params["xc_method"]
        )
    if all(
        [
            os.path.isfile(f)
            for f in [
                path_dict["prepped_train_df_path"],
                path_dict["prepped_test_df_path"],
                path_dict["prepped_train_labels"],
                path_dict["prepped_test_labels"],
            ]
        ]
    ):
        print(path_dict["prepped_test_labels"])
        train_df = pd.read_pickle(path_dict["prepped_train_df_path"])
        test_df = pd.read_pickle(path_dict["prepped_test_df_path"])
        train_labels = pd.read_csv(path_dict["prepped_train_labels"])
        test_labels = pd.read_csv(path_dict["prepped_test_labels"])
        best_corrs, cross_corr = get_correlations(
            train_df,
            train_labels,
            path_dict["corr_path"],
            path_dict["xc_path"],
            corr_method=select_params["corr_method"],
            xc_method=select_params["xc_method"],
        )
        return (
            train_df,
            train_labels,
            test_df,
            test_labels,
            best_corrs,
            cross_corr,
            path_dict,
        )
    else:
        print("\nCouldn't find files.\n")
        return None
    i += 1
    return (
        train_df,
        train_labels,
        test_df,
        test_labels,
        preprocessor,
        best_corrs,
        cross_corr,
    )


def get_original_enamine(use_cols=["SMILES", "DMSO Solubility", "INCHI_KEY"]):
    enamine_path = "{}enamine_data/tetko_enamine_training_download.csv".format(
        os.environ.get("DATA_DIR")
    )
    all_enamine = pd.read_csv(enamine_path, usecols=use_cols)
    return all_enamine


def _get_solubility_data(
    path_dict,
    select_params,
    conc_splits,
    preprocess=True,
    from_disk=True,
    save_results=True,
):
    loaded_data = None
    if from_disk:
        loaded_data = _get_prepped_enamine(select_params, path_dict)
        if loaded_data is not None:
            (
                train_df,
                test_df,
                train_labels,
                test_labels,
                best_corrs,
                cross_corr,
                path_dict,
            ) = loaded_data
        else:
            raise IOError
    else:
        # Get data
        max_conc, smiles_df, sid_to_inchikey_mapper = _get_unprocess_new_epa_data(
            path_dict
        )
        print(max_conc, smiles_df)
        # Discretize labels and map
        epa_labels = label_solubility_clusters(
            max_conc[smiles_df.index], conc_splits=conc_splits
        )
        smiles_df.index = smiles_df.index.map(
            lambda x: safe_mapper(x, sid_to_inchikey_mapper), na_action="ignore"
        )
        train_smiles_df, test_smiles_df = train_test_split(
            smiles_df, test_size=0.2, stratify=epa_labels.squeeze(), random_state=0
        )
        train_smiles_path = "{}train_epa_queried.pkl".format(os.environ.get("DATA_DIR"))
        test_smiles_path = "{}test_epa_queried.pkl".format(os.environ.get("DATA_DIR"))
        with open(train_smiles_path, "wb") as f:
            pickle.dump(train_smiles_df, f)
        with open(test_smiles_path, "wb") as f:
            pickle.dump(test_smiles_df, f)
        print("SMILES data pickled!")
        train_labels = epa_labels[train_smiles_df.index]
        test_labels = epa_labels[test_smiles_df.index]
        if preprocess:
            preprocessor, scaler = build_preprocessor.get_standard_preprocessor(
                transform_func="asinh", corr_params=select_params
            )
            train_df = preprocessor.fit_transform(train_smiles_df, train_labels)
            test_df = preprocessor.transform(test_smiles_df)
        else:
            preprocessor = None
            train_df = train_smiles_df
            test_df = test_smiles_df
        best_corrs = correlation_filter.calculate_correlation(
            train_df, train_labels, method=select_params["corr_method"]
        )
        if save_results:
            if preprocess:
                with open(path_dict["transformer_path"], "wb") as f:
                    pickle.dump(preprocessor, f)
            train_labels.to_pickle(path_dict["prepped_train_labels"])
            test_labels.to_pickle(path_dict["prepped_test_labels"])
            train_df.to_pickle(path_dict["prepped_train_df_path"])
            test_df.to_pickle(path_dict["prepped_test_df_path"])
    return train_df, train_labels, test_df, test_labels, best_corrs


def _get_new_epa_padel():
    raw_df = pd.read_pickle(
        "{}new_epa_data/desc_api_out.pkl".format(os.environ.get("MODEL_DIR"))
    )
    print("EPA Data Output: {}".format(raw_df.shape))
    no_na_df = raw_df.dropna(axis=1)
    print("EPA Data Output: {}".format(no_na_df.shape))
    return no_na_df


def _get_new_epa_data_split():
    raise NotImplementedError
    train_labels = pd.read_csv(
        "{}new_epa_data/train_labels.csv".format(os.environ.get("MODEL_DIR"))
    )
    test_labels = pd.read_csv(
        "{}new_epa_data/test_labels.csv".format(os.environ.get("MODEL_DIR"))
    )
    return train_labels, test_labels


def _get_unprocess_new_epa_data(path_dict):
    insol_labels, maxed_sol_labels, _ = get_query_data()
    combo_labels, convert_df = get_conversions(
        maxed_sol_labels, path_dict["lookup_path"]
    )
    smiles_df, sid_to_key = standardize_smiles(
        convert_df, combo_labels, path_dict["std_pkl"]
    )
    max_conc = combo_labels.set_index(keys="inchiKey", drop=True)[
        "sample_concentration"
    ].astype(np.float32)
    max_conc.index = max_conc.index.map(
        lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
    )
    return max_conc, smiles_df, sid_to_key


def label_solubility_clusters(labels, conc_splits=(9.9), algo=False):
    """
    Bins discretized labels as pandas Series for use with scikit-learn estimators.
    KMeans splitting algorithm is not stable.

    Parameters
    ----------
    labels: pd.Series, data to be binned
    conc_splits: Iterable[float], threshold values for binning
    algo: bool, whether to use KMeans clustering algorithm, experimental

    Returns
    -------
    checked_labels: pd.Series, discretized labels, checked by scikit-learns y_checker.
    """
    if algo:
        from sklearn.cluster import BisectingKMeans

        trimmed_labels = (
            labels.clip(upper=110.0, lower=4.5).multiply(100).apply(np.asinh)
        )
        print("Values = asinh(Solubility * 100)")
        print(trimmed_labels.describe())
        # agg_clusters = AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=3).fit_predict(trimmed_labels.to_frame())
        agg_clusters = BisectingKMeans(
            n_clusters=4, max_iter=5000, bisecting_strategy="largest_cluster"
        ).fit_predict(trimmed_labels.to_frame())
        agg_clusters = pd.Series(agg_clusters, index=trimmed_labels.index)
        for n in agg_clusters.sort_values().unique():
            cluster = labels.sort_values()[agg_clusters == n]
            print(labels[cluster.index].min(), labels[cluster.index].max())
        """
        label_clusterer = (
            KMeans(n_clusters=6, random_state=0)
            .set_output(transform="pandas")
            .fit(trimmed_labels.to_frame())
        )
        epa_labels = pd.Series(label_clusterer.predict(trimmed_labels.to_frame()), index=trimmed_labels.index)
        """
        binned_labels = agg_clusters
    else:
        raw_labels = labels.copy().sort_values().astype(np.float32)
        print(
            "Raw solubility statistics:".format(pprint.pformat(raw_labels.describe()))
        )
        if isinstance(conc_splits, Number):
            conc_splits = [conc_splits]
        splits = tuple(conc_splits)
        print("Concentration splits: {}".format(splits))
        binned_labels = pd.Series(
            data=np.empty_like(raw_labels, dtype=np.int8), index=raw_labels.index
        )
        binned_labels[raw_labels < splits[0]] = 0
        for i in np.arange(len(splits)):
            if i < len(splits) - 1:
                binned_labels[
                    (raw_labels >= splits[i]) & (raw_labels < splits[i + 1])
                ] = i
            if i == len(splits) - 1:
                binned_labels[raw_labels >= splits[i]] = i + 1
    # from sklearn.preprocessing import MultiLabelBinarizer
    # epa_labels = MultiLabelBinarizer(classes=epa_labels).fit_transform(epa_labels)
    checked_labels = pd.Series(data=_check_y(binned_labels), index=binned_labels.index)
    print("Discretized Labels: Value Counts:")
    print(checked_labels.value_counts(sort=False), flush=True)
    return checked_labels


def standardize_smiles(feature_df, combo_labels, std_pkl):
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
            smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.keys()
            )
            print("Retry label to InchiKey failures:\n{}".format(smiles_fails))
            # print(smiles_df.loc[smiles_fails])
        else:
            print("No return from retrying Standardizer on failed labels.")
            if "smiles" in smiles_df.columns:
                smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
        smiles_fails.difference(smiles_df.index).to_series().to_csv(
            "{}failed_standardization.csv".format(exp_dir)
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


def _set_paths(parent_dir, data_dir, select_params):
    # Paths.
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    path_dict = dict(
        [
            ("parent_dir", parent_dir),
            ("data_dir", data_dir),
            (
                "exp_dir",
                parent_dir,
            ),  # "{}{}/".format(parent_dir, select_params["model_name"])),
            ("std_pkl", "{}stdizer_epa_query.pkl".format(data_dir)),
            ("lookup_path", "{}lookup.csv".format(data_dir)),
            ("desc_path", "{}padel_features_output_max_sol.pkl".format(data_dir)),
            ("transformer_path", "{}transformer.pkl".format(data_dir)),
            (
                "transformed_df_path",
                "{}transformed_epa_exact_sol_data.pkl".format(data_dir),
            ),
            ("prepped_train_df_path", "{}prepped_train_df.pkl".format(data_dir)),
            ("prepped_test_df_path", "{}prepped_test_df.pkl".format(data_dir)),
            ("xc_path", "{}xc_df_{}.pkl".format(data_dir, select_params["xc_method"])),
            ("prepped_train_labels", "{}prepped_train_labels.pkl".format(parent_dir)),
            ("prepped_test_labels", "{}prepped_test_labels.pkl".format(parent_dir)),
            ("test_label_path", "{}test_labels.csv".format(parent_dir)),
            ("search_features_path", "{}search_features.csv".format(parent_dir)),
            (
                "corr_path",
                "{}corr_df_{}.pkl".format(parent_dir, select_params["corr_method"]),
            ),
            # Warning: This file may contain asinh transformed data for clustering algorithm input.
            # ("epa_labels", "{}epa_transformed.csv".format(parent_dir)),
        ]
    )
    os.makedirs(path_dict["exp_dir"], exist_ok=True)
    return path_dict
