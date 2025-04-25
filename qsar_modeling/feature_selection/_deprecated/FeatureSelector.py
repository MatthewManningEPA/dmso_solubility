import logging
import pickle
import pprint
from functools import partial

import cachetools
import numpy as np
import pandas as pd
import sklearn.utils.validation
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.utils.validation import check_X_y

from _deprecated.FeatureFrame import FeatureFrame
from constants import names, paths, run_params, selector_params
from qsar_modeling.data_handling.padel_categorization import padel_df

pd.options.display.max_colwidth = 30
pd.options.display.width = 30
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


# Use for supervised learning metrics.
# TODO: Check assumptions/requirements of estimators.
# noinspection PyPep8Naming
class FeatureSelectorTrainer:

    def __init__(self, options=None):
        self.options = options
        self._logger = logging.getLogger("dmso_logger.FeatureSelector")
        self._X = None
        self._y = None
        self._uninterpret_kws = None
        self._uninterpret_feats = None
        self._corr = None
        self._descriptor_groups_df = None
        self._condition_num = None
        self._mutual_info = None
        self._vif_mats = None
        self.paths = paths
        self.names = names
        self.run_params = run_params
        self.selector_params = selector_params
        # self.feature_properties = dict()
        # self.feat_scorer_types = dict()
        self._scalers = dict()
        # self._get_desc_groups(grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)

    """
    def __initialize_data_history__(self):
        for a in self.data_dict:
            self.data_history[a.__name] = [a]
    """

    def fit(self, X, y, uninterpret_kws=None, cov_mat=None):
        if self.run_params.debug:
            self._X = FeatureFrame(
                frame_data=X.sample(n=2500, axis="index")
                .sample(n=100, axis="columns")
                .copy(deep=True),
                options=self.options,
                logger=self._logger,
            )
            self.options.sample_wts = self.options.sample_wts[self.X.feat_frame.index]
            self._y = y[X.index]
        else:
            self._X = FeatureFrame(
                frame_data=X.copy(deep=True),
                options=self.options,
                logger=self._logger,
                cov_mat=cov_mat,
            )
            self._y = y
        if uninterpret_kws is not None:
            self._remove_uninterpretable()
        self._corr = self.X.feat_frame.corrwith(
            other=self.y, axis="columns", method=self.selector_params.cov_meth
        )

    def to_pickle(self, pkl_path):
        # self._options_dict = dataclasses.asdict(self.options)
        # self.__delattr__(self.options.__name__)
        logging.shutdown()
        self.__delattr__(self._logger.name)
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

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
        # corr = self.X.feat_frame[self.X.dense].corr(method=self.options.cov_method)
        dense_ind = self.X.feat_frame.index.intersection(self.X.dense)
        # print(self.X.feat_frame[self.X.dense].shape, sample_wts.shape)
        corr = pd.DataFrame(np.cov(self.X.feat_frame, rowvar=False, aweights=sample_wts, ddof=0),
                             index=self.X.feat_frame.columns
                             , columns=self.X.feat_frame.columns)
        assert corr is not None
        logging.info('Covariance Matrix:\n{}'.format(corr))
        logging.info('CoV NaNs: {}'.format(corr.isna().astype(int).sum().sum()))
        if filter_nan:
            # Remove features with no non-NaN correlations.
            na_corr = corr.isna().astype(int)
            sum_na = na_corr.sum(axis=0)
            while na_corr[sum_na > 0].size > 0:
                col_na = na_corr[sum_na == 0]
                most_na = na_corr.idxmax()
                self._logger.info('NA value for cols: {}'.format(col_na))
                # self._logger.warning('These features have no valid correlations: {}'.format(col_na))
                corr.drop(columns=col_na, inplace=True).drop(index=col_na, inplace=True)
                na_corr = corr.isna().astype(int)
                sum_na = na_corr.sum(axis=0)
            # self._logger.info('Correlation matrix: {}'.format(corr))
        assert corr is not None
        self._logger.debug('Covariance matrix with weighting of {}:\n{}'.format(self.options.sample_wts, corr))
        self._corr_mat = corr
        return corr
        """

    def _remove_uninterpretable(self):
        self._uninterpret_feats = pd.Index(
            [
                feat
                for feat in self.X.feat_frame.columns
                if len([kw for kw in self._uninterpret_kws if kw in feat]) > 0
            ]
        )

    @cachetools.cached(cache={})
    def set_condition_num(self):
        # Eigenvalue method for collinearity.
        eigenvalues = np.linalg.eigvals(self.X.cov_mat)
        self._condition_num = np.sqrt(max(eigenvalues) / eigenvalues)
        pprint.pp(self._condition_num)
        self._logger.info(f"Condition Index: {self._condition_num}")
        np.save("{}condition_index".format(self.paths.stats), arr=self._condition_num)

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
            seq_linear_select_estimator = partial(
                ElasticNetCV,
                n_jobs=self.run_params.pjobs,
                random_state=0,
                max_iter=5000,
                tol=1e-5,
                selection="random",
                cv=sklearn.model_selection.GroupKFold(n_splits=3),
            )
        else:
            seq_linear_select_estimator = partial(
                LassoCV,
                fit_intercept=False,
                max_iter=2500,
                tol=1e-5,
                n_jobs=self.run_params.pjobs,
                random_state=0,
                cv=GroupKFold(n_splits=3),
            )
        seq_linear_selector = SequentialFeatureSelector(
            estimator=seq_linear_select_estimator,
            scoring=seq_linear_select_scoring,
            n_jobs=num_parallel,
        )
        return seq_linear_selector.fit(df)

    # TODO: Delete groups with no members or just one member.
    def _set_descriptor_groups_df(self, grouping_dir=None, cols=None, use3d=False):
        padel_name_df = padel_df
        short_long_zip = zip(
            padel_name_df["Descriptor name"].tolist(),
            padel_name_df["Description"].tolist(),
        )
        short_long_dict = dict([(a, b) for a, b in short_long_zip])
        if cols is None:
            cols = self.names.feat_cols
        if grouping_dir is None:
            grouping_dir = self.paths.feature_groups
        if grouping_dir:
            desc_groups_df = pd.read_csv(filepath_or_buffer=grouping_dir, usecols=cols)
            # desc_groups_df.dropna(subset='Descriptor', inplace=True)
            if not use3d:
                # logging.info(desc_groups_df['Descriptor'])
                desc_groups_df.drop(
                    desc_groups_df[desc_groups_df["Class"] == "3D"].index, inplace=True
                )
            long_dict = dict()
            key_list = list()
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
                    long_dict.update({ind: desc_group})
                elif type(desc_group) is tuple or type(desc_group) is set:
                    long_dict.update({ind: list(desc_group)})
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

    # TODO: Find workaround to return df with no uninterpretable features.
    @property
    def X(self):
        """
        if self._uninterpret_feats is not None:
            return = self._X.feat_frame.drop(self._uninterpret_feats.intersection(self._X.feat_frame.columns))
        else:
        """
        return self._X

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def descriptor_groups_df(self):
        return self._descriptor_groups_df

    @descriptor_groups_df.setter
    def descriptor_groups_df(self, value):
        if self._descriptor_groups_df.shape[0] > 1:
            pass
        self._descriptor_groups_df = value

    @property
    def uninterpret_feats(self):
        return self._uninterpret_feats

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


"""def set_corr_pairs(matrix, cov_thresh, *args, **kwargs):
    corr_set = set()
    for i in range(1, matrix.shape[0]):
        for j in range(0, i + 1):
            if matrix[i, j] >= cov_thresh:
                corr_set.add((matrix.columns[i], matrix.columns[j]))
    return corr_set"""

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
