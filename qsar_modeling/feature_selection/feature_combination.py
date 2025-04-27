from logging import getLogger

import pandas as pd
from sklearn.decomposition import PCA
from type_check import is_iterable, make_lowercase

from feature_name_lists import get_features_dict

logger = getLogger()


def pca_threshold(feature_df, group_name, evr_thresh=0.95, evr_tol=None, **pca_params):
    transformed_list = [feature_df.columns[0]]
    prev_evr = 1.0
    pca_former = None
    for feature in feature_df.columns[1:]:
        transformed_list.append(feature)
        new_pca_former = PCA(n_components=1, **pca_params)
        pca_data = new_pca_former.transform(feature_df[transformed_list])
        new_evr = new_pca_former.explained_variance_ratio_
        logger.debug("EVR with {}: {}".format(feature, new_evr))
        if (evr_thresh is not None and new_evr < evr_thresh) or (
            evr_tol is not None and new_evr < prev_evr - evr_tol
        ):
            transformed_list.pop(transformed_list.index(feature))
            if len(transformed_list) == 1:
                return None, None
        else:
            prev_evr = new_evr
            pca_former = new_pca_former
    if len(pca_data.shape) > 0 and pca_former.n_components_ > 1:
        col_names = [
            "{}_pca_{}".format(group_name, i + 1)
            for i in list(range(pca_data.to_frame().shape[1]))
        ]
    else:
        col_names = ["{}_pca".format(group_name)]
    pca_data.columns = col_names
    print(transformed_list)
    print(group_name, prev_evr)
    return pca_data, pca_former


def combine_feature_groups(
    feature_df, evr_thresh=0.925, feature_dict=None, groups=None
):
    feature_df = feature_df.copy()
    if feature_dict is None:
        feature_dict = get_features_dict(feature_df.columns)
    if groups is None:
        groups = [
            "Bond_Orders",
            "All_Valence_Path",
            "Valence_Path_Counts",
            "Molecular_Path_Counts",
            "Molecular_Walk_Count",
            "Simple_Path",
            "Self_Return",
            "Rotations",
            "HBond",
            "Mol_Wt",
            "LogP",
        ]
    included_groups = dict(
        [(k, v) for k, v in feature_dict.items() if k in groups and len(v) > 0]
    )
    pca_feats, combined_features, pca_obj_list = list(), list(), list()
    for group_name, feat_list in included_groups.items():
        present_features = [f for f in feat_list if f in feature_df.columns]
        if len(present_features) <= 1:
            print("Not enough features in feature list for {}".format(group_name))
            print(feat_list)
            continue
        # if sort_key is not None:
        #    feat_list.sort(key=sort_key)
        pca_data, pca_former = pca_threshold(
            feature_df[present_features], group_name, evr_thresh=evr_thresh
        )
        if pca_data is not None:
            pca_feats.append(pca_data)
            pca_obj_list.append(pca_former)
        else:
            print("Feature group {} did not produce a PCA result.".format(group_name))
            pca_feats.append(pd.DataFrame([]))
            combined_features.append(list())
            pca_obj_list.append(None)
    combined_transforms = pd.concat(pca_feats, axis=1)
    return combined_transforms, pca_obj_list


def weight_by_feature(feature_df, to_norm, norm_by):
    return feature_df[to_norm].divide(feature_df[norm_by], axis=0)


def feature_partial_match(
    features, include, exclude, include_all=None, exclude_all=None, case_strict=False
):
    if is_iterable(features):
        return [
            feature_partial_match(f, include, exclude, case_strict=case_strict)
            for f in features
        ]
    if not case_strict:
        [
            make_lowercase(p)
            for p in [features, include, exclude, include_all, exclude_all]
        ]
    for p in [
        p
        for p in [include, exclude, include_all, exclude_all]
        if p is not None and is_iterable(p)
    ]:
        pass


def pca_features(feature_df, **pca_params):
    raise DeprecationWarning
    # Assumes that features have already been scaled!
    feature_pca = PCA(n_components=1, **pca_params).fit(X=feature_df)
    pca_dev_arr = feature_pca.transform(feature_df)
    pca_dev_X = pd.DataFrame(data=pca_dev_arr, index=feature_df.index)
    print("Explained Variance Ratio: {}".format(feature_pca.explained_variance_ratio_))
    print("Principal Components: {}".format(feature_pca.components_))
    return pca_dev_X, feature_pca
