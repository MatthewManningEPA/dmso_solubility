from collections import OrderedDict

import numpy as np
import pandas as pd

from utils import math_tools


def find_correlation(
    corr, cutoff=0.95, n_drop=None, corr_tol=0.0025, exact=True, verbose=False
):
    """
    This function is the Python implementation of the R function
    `find_correlation()`.

    Relies on numpy and pandas, so must have them pre-installed.

    It searches through a correlation matrix and returns a list of column names
    to remove to reduce pairwise correlations.

    For the documentation of the R function, see
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `find_correlation()`, see
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

    find_correlation(R1, cutoff=0.6, exact=False)  # ['x4', 'x5', 'x1', 'x3']
    find_correlation(R1, cutoff=0.6, exact=True)   # ['x1', 'x5', 'x4']
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
    acorr = corr.astype(float).abs()
    if exact or (exact is None and corr.shape[1] < 100):
        if n_drop is None:
            n_drop = corr.shape[1]
        else:
            n_drop = min(n_drop, corr.shape[1] - 1)
        print(
            "Initiating exact correlation search and dropping {} at most.".format(
                n_drop
            )
        )
        return _find_correlation_exact(acorr, n_drop, cutoff)
    else:
        avg = np.mean(corr, axis=1)
        return _find_correlation_fast(acorr, avg, cutoff)
