import itertools
import os
import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.utils.validation import _check_sample_weight

import utils.math_tools
from utils import math_tools


def cross_corr_filter(
    corr, cutoff=0.95, n_drop=None, corr_tol=0.0025, exact=True, verbose=False
):
    """
    This function is the Python implementation of the R function
    `cross_corr_filter()`.

    Relies on numpy and pandas, so must have them pre-installed.

    It searches through a correlation matrix and returns a list of column names
    to remove to reduce pairwise correlations.

    For the documentation of the R function, see
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `cross_corr_filter()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R

    -----------------------------------------------------------------------------

    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be
        recomputed at each step
    -----------------------------------------------------------------------------
    Returns:
    --------
    list of column names
    -----------------------------------------------------------------------------
    Example:
    --------
    R1 = pd.DataFrame({
        'x1': [1.0, 0.86, 0.56, 0.32, 0.85],
        'x2': [0.86, 1.0, 0.01, 0.74, 0.32],
        'x3': [0.56, 0.01, 1.0, 0.65, 0.91],
        'x4': [0.32, 0.74, 0.65, 1.0, 0.36],
        'x5': [0.85, 0.32, 0.91, 0.36, 1.0]
    }, index=['x1', 'x2', 'x3', 'x4', 'x5'])

    cross_corr_filter(R1, cutoff=0.6, exact=False)  # ['x4', 'x5', 'x1', 'x3']
    cross_corr_filter(R1, cutoff=0.6, exact=True)   # ['x1', 'x5', 'x4']
    """

    def _find_correlation_fast(corr_df, col_center, thresh):

        combsAboveCutoff = (
            corr_df.where(lambda x: (np.tril(x) == 0) & (x > thresh)).stack().index
        )

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = col_center[colsToCheck] > col_center[rowsToCheck].values
        delete_list = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()
        if verbose:
            print(delete_list)
        delete_ser = pd.Series(data=1, index=delete_list)
        return delete_ser

    def _find_correlation_exact(corr_df, max_drop, thresh):
        """
        Removed original code to allow custom sorting function to be used.
        corr_sorted = corr_df.loc[(*[col_center.sort_values(ascending=False).index] * 2,)]

        if (corr_sorted.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            corr_sorted = corr_sorted.astype(float)

        corr_sorted.values[(*[np.arange(len(corr_sorted))] * 2,)] = np.Nan
        """
        delete_dict = OrderedDict()
        for i in list(range(corr_df.shape[0])):
            corr_df.iloc[i, i] = 0.0
        sq_corr = corr_df * corr_df
        sum_sq_corr = np.sum(sq_corr, axis=0)
        # corr_sorted = corr_sorted.loc[col_norms_sorted]
        while len(delete_dict.keys()) < np.abs(max_drop):
            max_corr_idx = math_tools.get_max_arr_indices(
                corr_df, threshold=thresh, tol=corr_tol
            )
            max_corr_feats = set()
            [
                [max_corr_feats.add(i) for i in a]
                for a in max_corr_idx
                if corr_df.loc[a] >= thresh and a[0] != a[1]
            ]
            if len(max_corr_feats) == 0:
                if verbose:
                    print("All pairs over {} correlation eliminated".format(thresh))
                break
            del_feature = sum_sq_corr[list(max_corr_feats)].idxmax()
            if verbose:
                print("Deleted Feature: {}".format(del_feature))
            del_partners = set()
            [
                del_partners.add(i)
                for i, j in max_corr_idx
                if j == del_feature and i != del_feature
            ]
            [
                del_partners.add(j)
                for i, j in max_corr_idx
                if i == del_feature and j != del_feature
            ]
            delete_dict[del_feature] = [
                (p, corr_df.loc[del_feature, p]) for p in del_partners
            ]
            sum_sq_corr -= sq_corr[del_feature]
            sum_sq_corr.clip(lower=0.0, inplace=True)
            sq_corr.loc[del_feature] = sq_corr[del_feature] = 0.0
            corr_df.loc[del_feature] = corr_df[del_feature] = 0.0
        delete_ser = pd.Series(delete_dict, name="Correlated Features")
        return delete_ser

    """
    if not np.allclose(corr, corr.T, rtol=1e-3, atol=1e-2):
        raise ValueError("Correlation matrix is not symmetric.")
    elif any(corr.columns != corr.index):
        raise ValueError("Columns aren't equal to indices.")
    """
    print(corr)
    acorr = pd.DataFrame(
        corr.astype(np.float32).abs(), index=corr.index, columns=corr.columns
    )
    if exact or (exact is None and acorr.shape[1] < 100):
        print(acorr)
        if n_drop is None:
            n_drop = acorr.shape[1]
        else:
            n_drop = min(n_drop, acorr.shape[1] - 1)
        print(
            "Initiating exact correlation search and dropping {} at most.".format(
                n_drop
            )
        )
        return _find_correlation_exact(acorr, n_drop, cutoff)
    else:
        avg = np.mean(acorr, axis=1)
        return _find_correlation_fast(acorr, avg, cutoff)


def get_correlations(
    feature_df,
    labels,
    corr_path=None,
    xc_path=None,
    corr_method="kendall",
    xc_method="kendall",
    use_disk=True,
):
    """
    Returns label and pairwise feature correlation matrices using the given methods.
    Methods are valid options for pandas .corr: {"pearson", "spearman", "kendall"}
    Default is "kendall" (implemented as "c" variant, which handles ties and different ranges/cardinalities and is suitable for both continuous and ordinal data.
    Parameters
    ----------

    feature_df: DataFrame, feature set
    labels: Series, dependent variable
    corr_path: str, path to label correlation file
    xc_path: str, path to cross-correlation file
    corr_method: str, method to calculate label correlation coefficients
    xc_method: str, method to calculate cross-correlation coefficients

    Returns
    -------
    label_corr: Series, correlation coefficients with dependent variable
    cross_corr: DataFrame: cross-correlation coefficients

    """

    if corr_path is not None and os.path.isfile(corr_path) and use_disk:
        label_corr = pd.read_pickle(corr_path)
        print("Target correlation retrieved from disk.")
    else:
        print("Target correlation calculated.")
        label_corr = calculate_correlation(feature_df, labels, method=corr_method)
        if corr_path is not None and not labels.empty:
            label_corr.to_pickle(corr_path)
    if xc_path is not None and os.path.isfile(xc_path) and use_disk:
        print("Cross-correlation retrieved from disk.")
        cross_corr = pd.read_pickle(xc_path)
    else:
        print("Cross-correlation calculated.")
        cross_corr = calculate_correlation(feature_df, method=xc_method)
        if corr_path is not None and not cross_corr.empty:
            cross_corr.to_pickle(xc_path)
    # print("Cross-correlation:\n{}".format(cross_corr))
    return label_corr, cross_corr


def calculate_correlation(x1, x2=None, method="kendall"):

    def _get_method(meth):
        if meth == "kendall":
            return math_tools.calculate_kendalls_c
        else:
            return meth

    corr_method = _get_method(method)
    if x2 is None:
        corr_df = x1.corr(method=corr_method)
        # assert corr_df.shape == (x1.shape[1], x1.shape[1])
    else:
        corr_df = x1.corrwith(other=x2, method=corr_method)
        if False:
            if len(x2.shape) <= 1 or x2.shape[1] == 1:
                assert corr_df.shape[0] == x1.shape[1]
            elif len(x1.shape) <= 1 or x1.shape[1] == 1:
                assert corr_df.shape[0] == x2.shape[1]
    return corr_df


def weighted_mean(values, weights, axis=1):
    if len(np.shape(np.squeeze(values))) > 1:
        print(np.sum(values.T.mul(weights), axis=1).T / np.sum(weights))
        return np.sum(values.T.mul(weights), axis=1).T / np.sum(weights)
    else:
        return np.sum(weights * values) / np.sum(weights)


def weighted_covariance(x, y, weights):
    delta_x = x - weighted_mean(x, weights=weights)
    print(delta_x)
    if len(np.shape(np.squeeze(delta_x))) > 1:
        delta_x = delta_x.T
    if y is None:
        delta_y = delta_x
    else:
        delta_y = y - weighted_mean(y, weights=weights)
        print(delta_y)
    assert not np.any(np.isnan(delta_x))
    assert not np.any(np.isnan(delta_y))
    if len(np.shape(np.squeeze(delta_x))) > 1:
        return np.sum(delta_x * delta_y * weights, axis=1) / np.sum(weights)
    else:
        return np.sum(delta_x * delta_y * weights) / np.sum(weights)


def single_weighted_pearson_correlation(x, y, weights):
    # Sort index is necessary to get broadcasting to work?
    print(np.shape(weights), np.shape(x), np.shape(y))
    cov_xy = weighted_covariance(x, y, weights)
    cov_xx = weighted_covariance(x, y=None, weights=weights)
    cov_yy = weighted_covariance(y, y=None, weights=weights)
    print(cov_xy, cov_xx, cov_yy)
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def weighted_correlation_matrix(feature_df, weights):
    """
    Calculates pairwise weighted Pearson correlation coefficients for a DataFrame.

    Parameters
    ----------
    feature_df: pd.DataFrame, (n_samples, n_features)
    weights: pd.Series, (n_samples), relative importance weights

    Returns
    -------
    wcm: pd.DataFrame, (n_features, n_features) diagonally symmetric weighted correlations between features. All diagonal components are set to 1.0
    """
    # weights = (weights - weights.min()) / weights.max()
    # print(weights.describe())
    # print(weights.head())
    if not weights[weights > 0.000001].size > 0:
        return calculate_correlation(feature_df)
    w_sums = weights.sum()
    print("w_sums: {}".format(w_sums))
    w_df = feature_df.multiply(weights, axis="index")
    print("weighted features: {}".format(w_df))
    w_means = w_df.sum() / w_sums
    print("w_means: {}".format(w_means))
    delta_df = feature_df.sub(w_means)  # , axis=1)
    print("deltas: {}\n{}".format(delta_df.head(), delta_df.shape))
    assert delta_df.shape == feature_df.shape
    # pprint.pp(delta_df.apply(np.square, raw=True).multiply(weights, axis="index"))
    w_var = (
        delta_df.apply(np.square, raw=True)
        .multiply(weights, axis="index")
        .sum(axis=0)
        .divide(w_sums)
    ).to_numpy()
    # print("w_vars:\n{}\n{}".format(w_var, w_var.shape))
    w_std = np.sqrt(w_var)
    delta_arr = delta_df.to_numpy(dtype=np.float32)
    w_arr = weights.to_numpy(dtype=np.float32)
    # print("w_stds:\n{}\n{}".format(w_std, w_std.shape))
    wcm = np.empty(shape=(feature_df.shape[1], feature_df.shape[1]), dtype=np.float32)
    for i, j in itertools.combinations(np.arange(feature_df.shape[1]), 2):
        wcm[i, j] = (np.sum(w_arr * delta_arr[:, i] * delta_arr[:, j])) / (
            w_sums * w_std[i] * w_std[j]
        )
        wcm[j, i] = wcm[i, j]
        # Removed division by w_sum bc normalized above
    weighted_corr_df = pd.DataFrame(
        data=wcm,
        index=feature_df.columns,
        columns=feature_df.columns,
    )
    for i in np.arange(weighted_corr_df.shape[1]):
        weighted_corr_df.iloc[i, i] = 1.0
    print("Weighted corr matrix:\n{}".format(pprint.pformat(weighted_corr_df)))
    return weighted_corr_df


def bootstrapped_weighted_correlation(
    feature_df,
    weights,
    labels=None,
    method="kendall",
    n_bootstraps=100,
    sample_size=None,
):
    """
    Uses bootstrap sampling to estimate the weighted correlation. Correlations are simple averages of bootstrapped results.
    If labels is given, returns pd.Series [n_features] with weighted-correlation of features with label.
    So far, only Kendall's Tau-C is supported.

    Warnings: Not Implented!
    If labels is None, returns pairwise weighted-correlation matrix as pd.DataFrame [n_features, n_features]
    For non-bootstrapped, weighted Pearson correlation matrix, use weighted_correlation_matrix.

    Parameters
    ----------
    feature_df: pd.DataFrame shape [n_samples, n_features]
    weights: pd.Series,  shape [n_samples], Sample importance weights,
    labels: pd.Series, shape[n_samples],
    method: str or callable, So far only "kendall" is supported
    n_bootstraps: int, Number of sampling iterations to perform.
    sample_size: Number of samples drawn for each bootstrap iteration.

    Returns
    -------
    corr_mean: pd.DataFrame | pd.Series, Arithmetic mean of boostrapped correlations for each feature pair.
    """
    if sample_size is None:
        sample_size = int(
            min(feature_df.shape[0], max(500, np.sqrt(feature_df.shape[0])))
        )
    corr_list = list()
    for i in np.arange(n_bootstraps):
        sample = feature_df.sample(n=sample_size, axis="index", weights=weights)
        if method == "kendall":
            corr_list.append(
                [
                    utils.math_tools.calculate_kendalls_c(
                        sample[c], labels[sample.index]
                    )
                    for c in feature_df.columns
                ]
            )
        else:
            raise NotImplementedError
    print(corr_list[0])
    corr_df = pd.concat([pd.Series(c, index=feature_df.columns).T for c in corr_list])
    if len(corr_df.shape) < 2:
        corr_df = pd.DataFrame(
            data=np.stack([np.array(c) for c in corr_list]), columns=feature_df.columns
        )
    print(corr_df)
    # print(corr_df.columns)
    corr_mean = corr_df.mean(axis="index")
    print(corr_mean)
    corr_mean.index = feature_df.columns
    corr_std = corr_df.std(axis="index")
    corr_std.index = corr_df.columns
    print(
        "Correlation {} Mean/Std:\n{}\n{}\n".format(
            method, pprint.pformat(corr_mean), pprint.pformat(corr_std)
        )
    )
    return corr_mean


def get_weighted_correlations(feature_df, labels, select_params, subset_dir, weights):
    weighted_xc_path = "{}weighted_cross_corr.csv".format(subset_dir)
    weighted_corr_path = "{}weighted_label_corr.csv".format(subset_dir)
    # Get sample-weighted pairwise correlation matrix
    assert weights.shape[0] == feature_df.shape[0]
    assert weights[weights.isna()].shape[0] == 0
    if not weights.sum() > 0.0:
        return get_correlations(
            feature_df,
            labels,
            corr_method=select_params["corr_method"],
            xc_method=select_params["xc_method"],
        )
    feature_df.sort_index(inplace=True)
    labels.sort_index(inplace=True)
    weights.sort_index(inplace=True)
    weights = pd.Series(
        data=_check_sample_weight(weights, X=feature_df, ensure_non_negative=True),
        index=feature_df.index,
    )

    if os.path.isfile(weighted_xc_path):
        cross_corr = pd.read_csv(weighted_xc_path, index_col=0, header=0)
        for col in cross_corr.columns:
            cross_corr.loc[col, col] = 1.0
    else:
        cross_corr = weighted_correlation_matrix(
            feature_df=feature_df,
            weights=weights,
        )
        for col in cross_corr.columns:
            cross_corr.loc[col, col] = 1.0
        cross_corr.to_csv(weighted_xc_path)
    # Get sample-weighted label correlation.
    if os.path.isfile(weighted_corr_path):
        label_corr = pd.read_csv(weighted_corr_path, index_col=0).squeeze()
        # label_corr.drop(columns=label_corr.columns[0])
        # label_corr.index = feature_df.columns
    else:

        if "kendall" in select_params["corr_method"]:
            label_corr = bootstrapped_weighted_correlation(
                feature_df,
                weights=weights,
                labels=labels,
            )
        else:
            label_corr = single_weighted_pearson_correlation(
                x=feature_df, y=labels, weights=weights
            )
        label_corr.to_csv(weighted_corr_path)
    return label_corr, cross_corr
