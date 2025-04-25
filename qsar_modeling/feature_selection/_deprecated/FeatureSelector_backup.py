import functools
import itertools
import logging
import numbers
import os
import pprint

import cachetools
import numpy as np
import pandas as pd
import sklearn.utils.validation
from sklearn.feature_selection import (
    chi2,
    f_classif,
    mutual_info_classif,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.utils.validation import check_X_y
from statsmodels.stats.outliers_influence import variance_inflation_factor

import constants
from qsar_modeling.data_handling.padel_categorization import padel_df

pd.options.display.max_colwidth = 30
pd.options.display.width = 30
PADEL_GROUPS_DIR = "{}padel/padel_desc_groups.csv".format(os.environ.get("FINAL_DIR"))
GROUP_COL_NAMES = ("Type", "Number", "Descriptor", "Class")
WEIGHT_NAMES = (
    "mass",
    "charges",
    "van der Waals",
    "Sanderson",
    "electronegativites",
    "polarizabilities",
    "ionization",
    "I-state",
)
# Model running options.
linear_dim_reduction = False
plot_tsne = False
plot_logreg = True
forest = True
"""
@dataclasses.dataclass
class TrainingOpts:
    run_debug: bool
    corr_meth: str
    sample_wts: pd.Series
    pjobs: int
    data_dir: str | os.PathLike
    stats_dir: str | os.PathLike
"""


# TODO: Check assumptions/requirements of estimators.
class FeatureSelectorTrainer:
    _group_names = (
        "zero_var",
        "nonzero_var",
        "sparse",
        "dense",
        "discrete",
        "cont",
        "nonneg",
        "neg",
    )

    def __init__(
        self,
        data,
        labels=None,
        fs=None,
        options=None,
        logger=None,
        subsets=None,
        *args,
        **kwargs,
    ):
        self.options = options
        self._logger = self.set_options(log_opts=logger)
        if data.constants.run_debug:
            self._X = FeatureFrame(
                df=data.sample(n=2500, axis="index")
                .sample(n=100, axis="columns")
                .copy(deep=True),
                options=self.options,
                logger=self._logger,
            )
            self.options.sample_wts = self.options.sample_wts[self.X.feat_frame.index]
        elif fs and type(fs) is FeatureFrame:
            self._X = fs
        else:
            self._X = FeatureFrame(
                df=data.copy(deep=True), options=self.options, logger=self._logger
            )
        self._y = labels
        self._corr = None
        self._descriptor_groups_df = None
        self.condition_num = None
        # self._calc_values()
        # self.feature_properties = dict()
        # self.feat_scorer_types = dict()
        self._scalers = dict()
        # self._get_desc_groups(grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)
        if not data.constants.run_debug:
            if len(self.X.sparse) < self.options.min_sparse:
                self._logger.warning(
                    "Too few sparse features: \n{}".format(self.X.sparse)
                )
                raise ValueError
            if len(self.X.dense) < self.options.min_dense:
                self._logger.warning(
                    "Too few dense features: \n{}".format(self.X.dense)
                )
                raise ValueError
        cweights = self.y.copy().replace({1: 1, 0: 10})

    """
    def __initialize_data_history__(self):
        for a in self.data_dict:
            self.data_history[a.__name] = [a]
    """

    # Set default values for options named tuple.
    def set_options(self, log_opts=None):
        self._logger = logging.getLogger(name="dmso_logger.logFeatureSelector")
        logging.basicConfig(**log_opts)
        if constants.run_debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)
        return self._logger

    def _calc_values(self):
        # self._corr = self.calc_cov_mat()
        self._corr = self.X.feat_frame[self.X.nonzero_vars].corrwith(
            other=self.y, axis="columns", method=constants.cov_method
        )
        # self._set_descriptor_groups_df()

    """
    # aweights is used to balance covariance estimates for different subsets of observations.
    def calc_cov_mat(self, calc_cov=None, filter_nan=False, sample_wts='auto', *args, **kwargs):
        if calc_cov is not None:
            self._corr_mat = calc_cov
        cov_path = '{}cov.pkl'.format(self.options.stats_dir)
        if type(self.X.cov_mat) is pd.DataFrame and not self.X.cov_mat.empty and self.X.cov_mat.shape[1] == \
                self.X.feat_frame.shape[1]:
            self._logger.info('Covariance matrix has already been calculated.')
            return self.X.cov_mat
        if sample_wts == 'auto':
            sample_wts = self.options.sample_wts
        # _corr = self.X.feat_frame[self.X.dense].corr(method=self.options.cov_method)
        dense_ind = self.X.feat_frame.index.intersection(self.X.dense)
        # print(self.X.feat_frame[self.X.dense].shape, sample_wts.shape)
        _corr = pd.DataFrame(np.cov(self.X.feat_frame, rowvar=False, aweights=sample_wts, ddof=0),
                             index=self.X.feat_frame.columns
                             , columns=self.X.feat_frame.columns)
        assert _corr is not None
        logging.info('Covariance Matrix:\n{}'.format(_corr))
        logging.info('CoV NaNs: {}'.format(_corr.isna().astype(int).sum().sum()))
        if filter_nan:
            # Remove features with no non-NaN correlations.
            na_corr = _corr.isna().astype(int)
            sum_na = na_corr.sum(axis=0)
            while na_corr[sum_na > 0].size > 0:
                col_na = na_corr[sum_na == 0]
                most_na = na_corr.idxmax()
                self._logger.info('NA value for cols: {}'.format(col_na))
                # self._logger.warning('These features have no valid correlations: {}'.format(col_na))
                _corr.drop(columns=col_na, inplace=True).drop(index=col_na, inplace=True)
                na_corr = _corr.isna().astype(int)
                sum_na = na_corr.sum(axis=0)
            # self._logger.info('Correlation matrix: {}'.format(_corr))
        assert _corr is not None
        self._logger.debug('Covariance matrix with weighting of {}:\n{}'.format(self.options.sample_wts, _corr))
        self._corr_mat = _corr
        return _corr
        """

    @cachetools.cached(cache={})
    def set_condition_num(self):
        # Eigenvalue method for collinearity.
        eigenvalues = np.linalg.eigvals(self.X.cov_mat)
        condition_index = np.sqrt(max(eigenvalues) / eigenvalues)
        pprint.PrettyPrinter().pprint(condition_index)
        self._logger.info(f"Condition Index: {condition_index}")
        np.save("{}condition_index".format(self.options.stats_dir), arr=condition_index)
        self.condition_num = condition_index

    @cachetools.cached(cache={})
    def multi_collin_feats(self):
        dense_df = self.X.feat_frame[self.X.dense].copy(deep=True)
        self._logger.debug(self.X.dense)
        # sklearn.utils.assert_all_finite(dense_df)
        vif = pd.DataFrame()
        vif["features"] = dense_df.columns
        vif["VIF Factor"] = [
            variance_inflation_factor(
                np.array(dense_df.values.tolist(), dtype=float), i
            )
            for i in range(dense_df.shape[1])
        ]
        vif.set_index(keys="features", drop=True, inplace=True)
        self._logger.info(vif)
        return vif

    def seq_linear_select(
        self,
        df=None,
        label=None,
        method="auto",
        seq_linear_select_scoring="balanced_accuracy",
        num_parallel=1,
    ):
        if not df:
            df = self.X
        if not label:
            label = self.y
        check_X_y(df, label)
        if method == "elastic" or (method == "auto" and df.shape[1] < 50):
            seq_linear_select_estimator = ElasticNetCV(
                n_jobs=num_parallel, random_state=0, max_iter=5000, tol=1e-5, cv=3
            )
        else:
            seq_linear_select_estimator = LassoCV(
                fit_intercept=False,
                max_iter=2500,
                tol=1e-5,
                n_jobs=num_parallel,
                random_state=0,
                cv="balanced",
            )
        seq_linear_selector = SequentialFeatureSelector(
            estimator=seq_linear_select_estimator,
            scoring=seq_linear_select_scoring,
            n_jobs=num_parallel,
        )
        return seq_linear_selector.fit(df)

    # TODO: Delete groups with no members or just one member.
    def _set_descriptor_groups_df(
        self, grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES, use3d=False
    ):
        padel_name_df = padel_df
        short_long_zip = zip(
            padel_name_df["Descriptor name"].tolist(),
            padel_name_df["Description"].tolist(),
        )
        short_long_dict = dict([(a, b) for a, b in short_long_zip])
        if grouping_dir:
            desc_groups_df = pd.read_csv(filepath_or_buffer=grouping_dir, usecols=cols)
            # desc_groups_df.dropna(subset='Descriptor', inplace=True)
            if not use3d:
                # logging.info(desc_groups_df['Descriptor'])
                desc_groups_df.drop(
                    desc_groups_df[desc_groups_df["Class"] == "3D"].index, inplace=True
                )
            long_dict = dict()
            for _, i, desc_group in desc_groups_df[desc_groups_df["Number"] > 1][
                ["Type", "Descriptor"]
            ].itertuples():
                # logging.warning('{} {}'.format(type(desc_group), desc_group))
                # i, desc_group = vals.values
                ind = str(i)
                logging.info("Desc row: {}: {}".format(ind, desc_group))
                if (
                    desc_group == "nan"
                    or desc_group == np.NaN
                    or not desc_group
                    or type(desc_group) is float
                ):
                    long_dict.update([(ind, list())])
                    continue
                elif type(desc_group) is str:
                    if "," not in desc_group:
                        if desc_group in short_long_dict.keys():
                            long_dict.update([(ind, list(desc_group))])
                            continue
                        else:
                            logging.error(
                                "COULD NOT FIND DESCRIPTOR IN LIST: {}".format(
                                    desc_group
                                )
                            )
                            continue
                    elif "," in desc_group:
                        key_list = [d.strip() for d in desc_group.split(",")]
                        desc_list = list()
                        if any([len(k) == 1 for k in key_list]):
                            logging.warning("SPLITTING INTO SINGLE CHARACTERS!")
                        else:
                            for d in key_list:
                                if len(d) <= 1:
                                    logging.error(
                                        "Descriptor splitting for {} gives single characters".format(
                                            key_list
                                        )
                                    )
                                elif d not in short_long_dict.keys():
                                    logging.warning(
                                        'Descriptor not found in key list!!!: "{}"'.format(
                                            d
                                        )
                                    )
                                    continue
                                    # key_list.remove(d)
                                else:
                                    desc_list.append((d, short_long_dict[d]))
                                    self._logger.debug("{} in keys!".format(d))
                            long_dict = dict(desc_list)
                            if len(desc_list) == 0:
                                logging.warning(
                                    "Empty descriptor group label: {}: {}".format(
                                        ind, desc_group.split(",")
                                    )
                                )
                                long_dict.update({ind: list()})
                            elif len(desc_list) == 0 and any(
                                [
                                    k in desc_group.strip(",")
                                    for k in short_long_dict.keys()
                                ]
                            ):
                                logging.error(
                                    "DESCRIPTOR LIST NOT BEING SPLIT CORRECTLY! \n{}VS.\n{}".format(
                                        desc_group, desc_list
                                    )
                                )
                                raise ValueError
                            else:
                                long_dict.update({ind: list()})
                    else:
                        logging.error(
                            "Desc list {} is string but is not in descriptor DF and does not contain comma delimiter.".format(
                                key_list
                            )
                        )
                        raise ValueError
                elif type(desc_group) is list:
                    long_dict.update((ind, desc_group))
                elif type(desc_group) is tuple or type(desc_group) is set:
                    long_dict.update((ind, list(desc_group)))
                else:
                    logging.error(
                        "Unknown type {} for descriptor group {}".format(
                            type(desc_group), desc_group
                        )
                    )
                    raise TypeError
                    # long_dict.update([(ind, [short_long_dict[str(d)] for d in desc_group.split(',') if str(d) in short_long_dict.keys()])])
            self._logger.info("Input dictionary: {}".format(long_dict.items()))
            for k, v in long_dict.items():
                if type(v) is not list:
                    long_dict[k] = list(v)
            self._logger.info("Input dictionary: {}".format(long_dict.items()))
            long_df = pd.DataFrame.from_dict(long_dict)
            if long_df.empty:
                long_df = pd.DataFrame.from_dict(long_dict, orient="index")
            if long_df.empty:
                long_df = pd.DataFrame.from_records(data=long_dict)
            if long_df.empty:
                long_df = pd.Series(
                    data=long_dict.values(), index=long_dict.keys(), name="Long"
                )
            if long_df.empty:
                self._logger.warning("Long descriptor DF is empty!")
                raise ValueError
            if (
                type(long_df) is pd.DataFrame
                and len(long_df.shape) > 1
                and long_df.shape[0] < long_df.shape[1]
            ):
                long_df = long_df.T
            else:
                self._logger.info(type(long_df))
            self._logger.info("Brand new long_df: {}".format(long_df))
            try:
                long_df.rename(columns=["Long"], inplace=True)
            except TypeError:
                self._logger.error("Could not rename long_df")
            self._logger.info("Long DF: \n{}".format(long_df.to_string()))
            new_desc_df = desc_groups_df.set_index(keys="Type").join(
                long_df, how="inner"
            )
            if new_desc_df.empty:
                new_desc_df = desc_groups_df.join(long_df.reset_index(), how="inner")
            # desc_groups_df['Long'] = desc_groups_df['Descriptor'].apply(func=lambda x: [[short_long_dict[a.strip()] for a in d if a != 'NaN' and a != np.NaN] for d.split(',') in x if len(d) > 1 else x])
            # desc_groups_df['Long'] = desc_groups_df['Descriptor'].apply(func=lambda x: [short_long_dict[d.rstrip(',')] for d in x.split()])
            self._logger.info("Col Names: {}".format(new_desc_df.columns))
            self._logger.info("Long Name Descriptors: \n{}".format(new_desc_df))
            # new_desc_df.sort_values(key=lambda x: len(x), inplace=True, ascending=False)
        else:
            raise FileNotFoundError
        self._descriptor_groups_df = new_desc_df

    @cachetools.cached(cache={})
    def scale_df(self, df=None, scale_to=None, select_scalers="all", *args, **kwargs):
        df_scaled = dict()
        if not df:
            df = self.X.cont
        if not self.scalers:
            self.scalers = self.scale_df(scale_to, *args, **kwargs)
        if select_scalers != "all":
            select_scalars = [
                key for key in self.scalers.keys() if key in select_scalers
            ]
        else:
            select_scalers = self.scalers
        for name, scaler in self.scalers.items():
            if sklearn.utils.validation.check_is_fitted(scaler):
                df_scaled.update((name, scaler.transform(df)))
            else:
                if scale_to:
                    scaler.fit(
                        scale_to,
                    )
                    scaler.transform(df)
                else:
                    scaler.fit_transform(df)
                self.scalers.update((name, self.scalers))
        return df_scaled

    # Set Sample weight option.
    def isolate_observations(
        self,
        fit_to=None,
        sample_wts="auto",
        contamination=0.001,
        rstate=None,
        n_jobs=-1,
    ):
        from sklearn.ensemble import IsolationForest

        if not fit_to:
            fit_to = self.X.feat_frame[self.X.nonzero_vars]
        if sample_wts == "auto":
            sample_wts = None
        iso = IsolationForest(
            n_jobs=n_jobs, contamination=contamination, random_state=0
        )
        if fit_to:
            iso = iso.fit(fit_to, sample_weight=sample_wts)
        return iso

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @staticmethod
    def get_high_vars(self, n=50):
        scaler = RobustScaler()
        high_thresh = VarianceThreshold()
        high_thresh_pipe = Pipeline(
            steps=[("high_var_scaler", scaler), ("high_thresh", high_thresh)]
        )
        return high_thresh_pipe

    @property
    def corr(self):
        return self._corr

    @property
    def descriptor_groups_df(self):
        return self._descriptor_groups_df

    @descriptor_groups_df.setter
    def descriptor_groups_df(self, value):
        if self._descriptor_groups_df.shape[0] > 1:
            pass
        self._descriptor_groups_df = value

    @property
    def scalers(self):
        return self._scalers

    @cachetools.cached(cache={})
    @scalers.setter
    def scalers(
        self,
        fit_to=None,
        robust_iqr=(0.05, 0.95),
        unit_robust_iqr=(0.05, 0.95),
        normalizer_norm="l2",
        *args,
        **kwargs,
    ):
        """
        :param fit_to:
        :param robust_iqr:
        :param unit_robust_iqr:
        :param normalizer_norm:
        :param args:
        :param kwargs:
        :return:
        """

        # noinspection PyArgumentEqualDefault
        scaler_dict = dict(
            [
                (
                    "robust",
                    RobustScaler(
                        quantile_range=robust_iqr, unit_variance=False, **kwargs
                    ),
                ),
                (
                    "robust_unit",
                    RobustScaler(
                        quantile_range=unit_robust_iqr, unit_variance=True, **kwargs
                    ),
                ),
                ("normal", Normalizer(norm=normalizer_norm, **kwargs)),
            ]
        )
        if fit_to:
            [val.fit(fit_to.to_numpy()) for val in scaler_dict.values()]
        self._scalers = scaler_dict

    @X.setter
    def X(self, value):
        if type(value) is not FeatureFrame:
            raise AttributeError
        self._X = value

    @y.setter
    def y(self, value):
        self._y = value

    # TODO: Code dimensionality reduction using covariance, VIF, MI.
    @cachetools.cached(cache={})
    def set_cov_pairs(self, cov_thresh=None):
        if not cov_thresh:
            cov_thresh = 0.9
        corr_mat = self.X.cov_mat
        # corr_dict = cov_mat.to_dict(orient='series')
        same_list, close_list = list(), list()
        for ind, col in itertools.combinations(corr_mat.columns.tolist(), r=2):
            val = corr_mat.loc[ind, col]
            # self._logger(ind, col, val, flush=True)
            if not isinstance(val, numbers.Number) or np.abs(val) > 1.0:
                self._logger.warning(
                    "CoV matrix element is non-numerical: {}.\nThis happened on row: \n{:25}\nand column:\n{:25}\n".format(
                        val, ind, col
                    )
                )
            elif np.abs(val) == 1.0:
                same_list.append((col, ind))
                self._logger.warning(
                    "Perfectly correlated feature pair with R of {}:\n{:25}\n{:25} ".format(
                        val, ind, col
                    )
                )
            elif np.abs(val) > self.options["cov_thresh"]:
                self._logger.info(
                    "Highly correlated features with an R of {:25}\n{:25}\n{}".format(
                        val, ind, col
                    )
                )
                close_list.append([(ind, col), val])
        self._logger.info(close_list)
        self.X._cov_pairs = close_list
        return close_list


def iter_feats(feat_groups):
    for name, feat_dict in feat_groups:
        col_group, selector = feat_dict


REQUIRED_OPTS = ("tol_discrete", "tol_sparse", "sparsity")

"""def set_corr_pairs(matrix, cov_thresh, *args, **kwargs):
    corr_set = set()
    for i in range(1, matrix.shape[0]):
        for j in range(0, i + 1):
            if matrix[i, j] >= cov_thresh:
                corr_set.add((matrix.columns[i], matrix.columns[j]))
    return corr_set"""


def set_corr_mat(
    df, corr_method="auto", filter_nan=False, sample_wts=None, *args, **kwargs
):
    if corr_method == "auto":
        norm_wt = np.sum(np.array(sample_wts)) / len(sample_wts.keys())
        corr_arr = np.divide(
            np.cov(df, rowvar=False, ddof=0, aweights=sample_wts), np.sum(sample_wts)
        )
        cov_mat = pd.DataFrame(data=corr_arr, index=df.columns, columns=df.columns)
    elif (
        corr_method == "pearson"
        or corr_method == "spearman"
        or corr_method == "kendall"
    ):
        cov_mat = df.corr(method=corr_method)
    else:
        raise ValueError
    # if cov_mat.isna().astype(int).sum().sum() > 0:
    #    print('Covariance matrix contains invalid values.')
    excess_cov = np.nonzero(1 - cov_mat.abs())
    if len(excess_cov) > cov_mat.size:
        print(
            "There are {} covariance values greater outside of [-1, 1]".format(
                len(excess_cov)
            )
        )
    if filter_nan:
        for tup in excess_cov:
            ind, col = tup
            val = cov_mat[ind, col]
            print(
                "Invalid value in CoV matrix: {}\nRow: {}\nColumn:{}\n".format(
                    val, ind, col
                )
            )
        if invalid:
            raise ValueError
    return cov_mat


def feat_types_clf_scorers(value):
    cont_mi_clf = functools.partialmethod(mutual_info_classif, discrete_features=False)
    disc_mi_clf = functools.partialmethod(mutual_info_classif, discrete_features=True)
    return dict(
        [
            (f_classif, "real"),
            (cont_mi_clf, "real"),
            (disc_mi_clf, "discrete"),
            (chi2, "nonneg"),
        ]
    )


class FeatureFrame:

    def __init__(self, df=None, options=None, logger=None, *args, **kwargs):
        self._feat_frame = df
        self._original = self.feat_frame.copy(deep=True)
        self._logger = logger
        self._options = options
        self._feat_vars = None
        self._nonzero_vars = None
        self._zero_vars = None
        self._sparse = None
        self._dense = None
        self._discrete = None
        self._cont = None
        self._neg = None
        self._nonneg = None
        self._feat_stat_scorer = None
        self._cov_mat = None
        self._cov_pairs = None
        self._desc_groups = None
        self._set_attr()

    @property
    def feat_frame(self):
        return self._feat_frame

    @property
    def original(self):
        return self._original

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self.format_options(value)

    def _set_attr(self):
        self._set_feat_vars()
        self._set_sparse_dense()
        self._set_discrete_cont()
        self._set_neg_nonneg()
        self.cov_mat = None
        # self.scalers()
        # feat_types_clf_scorers(None)

    @property
    def feat_vars(self):
        return self._feat_vars

    def _set_feat_vars(self, skipna=True, numeric_only=False, *args, **kwargs):
        feat_vars = self.original.var(
            axis="index", skipna=skipna, numeric_only=numeric_only
        )
        self._logger.info("Feature variances: {}".format(feat_vars))
        if feat_vars.empty or feat_vars.dropna().empty:
            raise ValueError
        self._feat_vars = feat_vars
        return feat_vars
        # self.nonzero_vars((self.feat_vars > 0).index.append(pd.Index([c for c in self.original.columns if c not in self.feat_vars.index])))
        # self.zero_vars(self.original.columns.symmetric_difference(self.nonzero_vars))

    @property
    def desc_groups(self):
        return self._desc_groups

    @property
    def feat_stat_scorer(self):
        return self._feat_stat_scorer

    @property
    def discrete(self):
        return self._discrete

    @discrete.setter
    def discrete(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._discrete = value

    @property
    def cont(self):
        return self._cont

    @cont.setter
    def cont(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._cont = value

    def _set_neg_nonneg(self):
        nonneg_cols, neg_cols = list(), list()
        for col, ser in self.feat_frame.items():
            if (ser.values < 0).any():
                neg_cols.append(col)
            else:
                nonneg_cols.append(col)
        self.nonneg = pd.Index(nonneg_cols)
        self.neg = pd.Index(neg_cols)
        self._logger.info("Nonnegative features: {}".format(self.nonneg.size))
        self._logger.info("Features with negative values: {}".format(self.neg.size))

    def _set_discrete_cont(self):
        # remainder = self.feat_frame.copy().round(0).subtract(self.feat_frame).astype(float)
        # self._logger.info(remainder.max(axis='rows').sort_values(ascending=False)[:5])
        # self._logger.info(remainder.max(axis='rows').sort_values(ascending=True)[:5])
        count_cols = pd.Index(
            [
                x
                for x in self.feat_frame.columns
                if ("count" in x.lower() or "number" in x.lower())
                and "measure" not in x.lower()
            ]
        )
        non_counts = self.feat_frame.columns.difference(count_cols)
        uniques = self.feat_frame[non_counts].nunique(axis="columns")
        self._logger.info(
            'Columns containing "count" or "number": \n{}'.format(count_cols.tolist())
        )
        self.discrete = uniques[
            uniques < (self.options.discrete_max * self.feat_frame[non_counts].shape[1])
        ].index.union(count_cols)
        # self.discrete = remainder.columns[((remainder <= self.options.tol_discrete) | (remainder >= (1 - self.options.tol_discrete))).astype(int).sum() == 0]
        self.cont = self.feat_frame.columns.difference(self.discrete)
        self._logger.info("Discrete features: {}".format(self.discrete.size))
        self._logger.info("Continuous features: {}".format(self.cont.size))
        if (
            self.discrete.size < self.options.min_discrete
            or self.cont.size < self.options.min_continuous
        ):
            self._logger.error(
                "Too few features for tol_sparse: {}.".format(self.options.tol_discrete)
            )
            raise AttributeError

    @property
    def cov_mat(self):
        return self._cov_mat

    @cov_mat.setter
    def cov_mat(self, val):
        if (
            self.cov_mat is not None
            and type(self.cov_mat) is pd.DataFrame
            and not self.cov_mat.empty
        ):
            self._logger.warn("Covariance matrix has already been calculated.")
            pass
        else:
            self._cov_mat = set_corr_mat(
                self.feat_frame[self.dense],
                corr_method=constants.cov_method,
                sample_wts=self.options.sample_wts,
            )

    @property
    def cov_pairs(self):
        return self._cov_pairs

    @cov_pairs.setter
    def cov_pairs(self, val):
        if type(self.cov_mat) is pd.DataFrame and not self.cov_mat.empty:
            self._cov_pairs = set_corr_pairs(
                self.cov_mat, cov_thresh=self.options.corr_thresh
            )

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._sparse = value

    @property
    def zero_vars(self):
        return self._zero_vars

    @zero_vars.setter
    def zero_vars(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._zero_vars = value

    @property
    def nonzero_vars(self):
        return self._nonzero_vars

    @nonzero_vars.setter
    def nonzero_vars(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._nonzero_vars = value

    @cachetools.cached(cache={})
    def _set_sparse_dense(self, all_freq=True, ignore_nan=False):
        if 0 < self.options.tol_sparse < 1:
            freq_cut = 1 - self.options.tol_sparse
        elif 1 < self.options.tol_sparse < self.feat_frame.vary.shape[0]:
            freq_cut = self.options.tol_sparse / self.feat_frame.vary.shape[0]
        else:
            raise ValueError
        # self._logger.info('Sparse tolerance: {}'.format(self.options.tol_sparse))
        # self._logger.info('Freq cut: {}'.format(freq_cut))
        sparse_list, dense_list, zero_list, freq_dict = list(), list(), list(), dict()
        for col, ser in self._pd_feat_freq(ignore_nan=ignore_nan):
            if all_freq:
                freq_dict[col] = ser
            sermax = ser.max(skipna=ignore_nan)
            # print('Feature freq max: {}'.format(sermax))
            logging.debug("Feature maximum: {}".format(sermax))
            if freq_cut < sermax < 1.0:
                sparse_list.append(col)
            elif freq_cut > sermax:
                dense_list.append(col)
            else:
                zero_list.append(col)
        self.sparse = pd.Index(sparse_list)
        self.dense = pd.Index(dense_list)
        self.zero_vars = pd.Index(zero_list)
        self.nonzero_vars = self.original.columns.difference(self.zero_vars)
        self._logger.info("Zero variance features: {}".format(self.zero_vars.size))
        self._logger.info(
            "Non-zero variance features: {}".format(self.nonzero_vars.size)
        )
        self._logger.info("Sparse features: {}".format(self.sparse.size))
        self._logger.info("Dense features: {}".format(self.dense.size))

        # self.feature_properties['zero_vars'] = self.X.vary.columns.symmetric_difference(pd.Index(current_data=nonzeros))

    @cachetools.cached(cache={})
    def _pd_feat_freq(self, ignore_nan=False):
        # self._logger.info('Feature variances: ', self.feat_vars)
        for col, ser in self.original.items():
            yield col, ser.value_counts(normalize=True, dropna=ignore_nan)

    @functools.singledispatch
    def format_options(self, arg):
        self._options = arg

    @format_options.register(dict)
    def _(self, arg):
        for o in REQUIRED_OPTS:
            if o not in arg.keys():
                # self._logger.error('Required option {} not found.'.format(o))
                raise KeyError
        self._options = arg


# TODO Balanced covariance matrix.
# def balanced_cov(df, corr_method, ):

# selector = sklearn.feature_selection.SelectFromModel(estimator=score_f, max_features=n_feats, prefit=True, threshold='median')

# padel_groups = _get_desc_groups(X, grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)
# feat_group_names = ['Dense', 'Matrix-based', 'E-States', 'Sparse', 'All Features']
# feat_selector_bool = [False, True, True, True, False]
# feat_col_list = [dense, matrix, estate, sparse,  X_train.columns]
# zip(feat_group_names, feat_col_list)
# feat_col_list = [sparse, dense, matrix, estate, X_train.columns]


# train_feats = X_train.copy()
# train_labels = train_y.copy()
