import constants

Index: qsar_modeling / feature_selection / FeatureFrame.py
IDEA
additional
info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
< + > UTF - 8
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
diff - -git
a / qsar_modeling / feature_selection / FeatureFrame.py
b / qsar_modeling / feature_selection / FeatureFrame.py
new
file
mode
100644
--- / dev / null(date
1723496185068)
+++ b / qsar_modeling / feature_selection / FeatureFrame.py(date
1723496185068)

@ @-0

, 0 + 1, 245 @ @
+

+

+
+

+

+
+

+
+
+


class FeatureFrame:
    +


+


def __init__(self, df=None, options=None, logger=None, *args, **kwargs):
    +        self.feat_frame = None


+        self.labels = None
+        self._logger = logger
+        self.options = options
+        self.feat_vars = None
+        self.nonzero_vars = None
+        self.zero_vars = None
+        self.sparse = None
+        self.dense = None
+        self.discrete = None
+        self.cont = None
+        self.neg = None
+        self.nonneg = None
+        self.feat_stat_scorer = None
+        self.cov_mat = None
+        self.cov_pairs = None
+        self.desc_groups = None
+        self.fit(,
+
+


def fit(self, df, labels=None):
    +        self.feat_frame = df


+        self.labels = labels
+
+


def _set_attr(self, calc_cov=False):
    +        self._set_feat_vars()


+        self._set_sparse_dense()
+        self._set_discrete_cont()
+        self._set_neg_nonneg()
+
if calc_cov:
    +            self.cov_mat = set_corr_mat(self.feat_frame, corr_method=constants.cov_method,
                                             +                                        sample_wts = self.options.sample_wts)
    +  # self.scalers()
    +  # feat_types_clf_scorers(None)
    +
    +


    def _set_feat_vars(self, df, skipna=True, numeric_only=False, *args, **kwargs):

        +        feat_vars = df.var(axis='index', skipna=skipna, numeric_only=numeric_only)


    +        self._logger.info('Feature variances: {}'.format(feat_vars))
    +
    if feat_vars.empty or feat_vars.dropna().empty:
        +
    raise ValueError
    +        self._feat_vars = feat_vars
    +
    return feat_vars
    +  # self.nonzero_vars((self.feat_vars > 0).index.append(pd.Index([c for c in self.original.columns if c not in self.feat_vars.index])))
    +  # self.zero_vars(self.original.columns.symmetric_difference(self.nonzero_vars))
    +
    +


    def _set_neg_nonneg(self, df):

        +        nonneg_cols, neg_cols = list(), list()


    +
    for col, ser in self.feat_frame.items():
        +
    if (ser.values < 0).any():
        +                neg_cols.append(col)
    + else:
    +                nonneg_cols.append(col)
    +        self.nonneg = pd.Index(nonneg_cols)
    +        self.neg = pd.Index(neg_cols)
    +        self._logger.info('Nonnegative features: {}'.format(self.nonneg.size))
    +        self._logger.info('Features with negative values: {}'.format(self.neg.size))
    +
    +


    def _set_discrete_cont(self, df):

        +  # remainder = self.feat_frame.copy().round(0).subtract(self.feat_frame).astype(float)


    +  # self._logger.info(remainder.max(axis='rows').sort_values(ascending=False)[:5])
    +  # self._logger.info(remainder.max(axis='rows').sort_values(ascending=True)[:5])
    +        count_cols = pd.Index([x for x in self.feat_frame.columns if
                                    +                               (
                                                'count' in x.lower() or 'number' in x.lower()) and 'measure' not in x.lower()])
    +        non_counts = self.feat_frame.columns.difference(count_cols)
    +        uniques = self.feat_frame[non_counts].nunique(axis='columns')
    +        self._logger.info('Columns containing "count" or "number": \n{}'.format(count_cols.tolist()))
    +        self.discrete = uniques[
        +            uniques < (self.options.discrete_max * self.feat_frame[non_counts].shape[1])].index.union(
        count_cols)
    +  # self.discrete = remainder.columns[((remainder <= self.options.tol_discrete) | (remainder >= (1 - self.options.tol_discrete))).astype(int).sum() == 0]
    +        self.cont = self.feat_frame.columns.difference(self.discrete)
    +        self._logger.info('Discrete features: {}'.format(self.discrete.size))
    +        self._logger.info('Continuous features: {}'.format(self.cont.size))
    +
    if self.discrete.size < self.options.min_discrete or self.cont.size < self.options.min_continuous:
        +            self._logger.error('Too few features for tol_sparse: {}.'.format(self.options.tol_discrete))
    +
    raise AttributeError
    +
    + @ cachetools.cached(cache={})
    +


    def _set_sparse_dense(self, all_freq=True, ignore_nan=False):

        +


    if 0 < self.options.tol_sparse < 1:
        +            freq_cut = 1 - self.options.tol_sparse
    + elif 1 < self.options.tol_sparse < self.feat_frame.vary.shape[0]:
    +            freq_cut = self.options.tol_sparse / self.feat_frame.vary.shape[0]
    + else:
    +
    raise ValueError
    +  # self._logger.info('Sparse tolerance: {}'.format(self.options.tol_sparse))
    +  # self._logger.info('Freq cut: {}'.format(freq_cut))
    +        sparse_list, dense_list, zero_list, freq_dict = list(), list(), list(), dict()
    +
    for col, ser in self._pd_feat_freq(ignore_nan=ignore_nan):
        +
    if all_freq:
        +                freq_dict[col] = ser
    +            sermax = ser.max(skipna=ignore_nan)
    +  # print('Feature freq max: {}'.format(sermax))
    +            logging.debug('Feature maximum: {}'.format(sermax))
    +
    if freq_cut < sermax < 1.:
        +                sparse_list.append(col)
    + elif freq_cut > sermax:
    +                dense_list.append(col)
    + else:
    +                zero_list.append(col)
    +        self.sparse = pd.Index(sparse_list)
    +        self.dense = pd.Index(dense_list)
    +        self.zero_vars = pd.Index(zero_list)
    +        self.nonzero_vars = self.original.columns.difference(self.zero_vars)
    +        self._logger.info('Zero variance features: {}'.format(self.zero_vars.size))
    +        self._logger.info('Non-zero variance features: {}'.format(self.nonzero_vars.size))
    +        self._logger.info('Sparse features: {}'.format(self.sparse.size))
    +        self._logger.info('Dense features: {}'.format(self.dense.size))
    +
    + @ property
    +


    def feat_frame(self):

        +


    return self.feat_frame
    +
    + @ property
    +


    def original(self):

        +


    return self.original
    +
    + @ property
    +


    def options(self):

        +


    return self.options
    +
    + @ options.setter
    +


    def options(self, value):

        +        self.format_options(value)


    +
    + @ property
    +


    def feat_vars(self):

        +


    return self.feat_vars
    +
    + @ property
    +


    def cov_mat(self):

        +


    return self._cov_mat
    +
    + @ cov_mat.setter
    +


    def cov_mat(self, val):

        +


    if self.cov_mat is not None and type(self.cov_mat) is pd.DataFrame and not self.cov_mat.empty:
        +            self._logger.warn('Covariance matrix has already been calculated.')
    +
    pass
    + else:
    +            self._cov_mat = set_corr_mat(self.feat_frame[self.dense], corr_method=data.constants.cov_method,
                                              +                                         sample_wts = self.options.sample_wts)
    +
    + @ property
    +


    def cov_pairs(self):

        +


    return self.cov_pairs
    +
    + @ cov_pairs.setter
    +


    def cov_pairs(self, val):

        +


    if type(self.cov_mat) is pd.DataFrame and not self.cov_mat.empty:
        +            self.cov_pairs = set_corr_pairs(self.cov_mat, cov_thresh=self.options.corr_thresh)
    +
    + @ property
    +


    def sparse(self):

        +


    return self.sparse
    +
    + @ sparse.setter
    +


    def sparse(self, value):

        +


    if type(value) is not pd.Index:
        +
    raise AttributeError
    +        self.sparse = value
    +
    + @ property
    +


    def zero_vars(self):

        +


    return self.zero_vars
    +
    + @ zero_vars.setter
    +


    def zero_vars(self, value):

        +


    if type(value) is not pd.Index:
        +
    raise AttributeError
    +        self.zero_vars = value
    +
    + @ property
    +


    def nonzero_vars(self):

        +


    return self.nonzero_vars
    +
    + @ nonzero_vars.setter
    +


    def nonzero_vars(self, value):

        +


    if type(value) is not pd.Index:
        +
    raise AttributeError
    +        self.nonzero_vars = value
    +  # self.feature_properties['zero_vars'] = self.X.vary.columns.symmetric_difference(pd.Index(current_data=nonzeros))
    +
    + @ property
    +


    def desc_groups(self):

        +


    return self.desc_groups
    +
    + @ property
    +


    def feat_stat_scorer(self):

        +


    return self.feat_stat_scorer
    +
    + @ property
    +


    def discrete(self):

        +


    return self.discrete
    +
    + @ discrete.setter
    +


    def discrete(self, value):

        +


    if type(value) is not pd.Index:
        +
    raise AttributeError
    +        self.discrete = value
    +
    + @ property
    +


    def cont(self):

        +


    return self.cont
    +
    + @ cont.setter
    +


    def cont(self, value):

        +


    if type(value) is not pd.Index:
        +
    raise AttributeError
    +        self.cont = value
    +
    + @ cachetools.cached(cache={})
    +


    def _pd_feat_freq(self, ignore_nan=False):

        +  # self._logger.info('Feature variances: ', self.feat_vars)


    +
    for col, ser in self.original.items():
        +
    yield col, ser.value_counts(normalize=True, dropna=ignore_nan)
    +
    + @ functools.singledispatch
    +


    def format_options(self, arg):

        +        self.options = arg


    +
    + @ format_options.register(dict)
    +


    def _(self, arg):

        +


    for o in REQUIRED_OPTS:
        +
    if o not in arg.keys():
        +  # self._logger.error('Required option {} not found.'.format(o))
    +
    raise KeyError
    +        self._options = arg
    +
    + @ feat_vars.setter
    +


    def feat_vars(self, value):

        +        self._feat_vars = value


    +
    + @ feat_frame.setter
    +


    def feat_frame(self, value):

        +        self._feat_frame = value
