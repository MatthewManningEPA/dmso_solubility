import os.path
import pickle
import pprint
from functools import partial

import numpy as np
import pandas as pd
import sklearn.pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler, CondensedNearestNeighbour
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFE, SelectPercentile
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid, StratifiedKFold, \
    cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer, KBinsDiscretizer, StandardScaler
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils import estimator_html_repr, compute_sample_weight

from data_handling.persistence import logger

pd.options.display.max_colwidth = 30
pd.options.display.width = 80
pp = partial(pprint.pp, compact=True)


def feature_sampler(fs=None, feat_df=None, sparse_feats='auto'):
    # Get sparse feats.
    assert not (fs is None and feat_df is None)
    if sparse_feats is None or sparse_feats == 'auto':
        if fs is not None:
            if feat_df is None:
                feat_df = fs.X.feat_frame
            sparse_feats = [f for f in fs.X.sparse if 'E-state' in f]

        elif feat_df is not None:
            sparse_feats = [c for c in feat_df.columns if not 'E-state' not in c]
    # Get more predictive features.
    feats_dict = dict()
    for feat in sparse_feats:
        counts = feat_df[feat].count_values(ascending=False)
        first_val = counts.index[0]
        if counts.values[0] >= 0.99 * feat_df.shape[1]:
            continue
        ser = feat_df[feat]
        classes = ser.where(ser == first_val, 1).where(ser != first_val, 0)
        feats_dict[feat] = classes
    return feats_dict


def get_pred_feats(feat_df, target, var_thresh=0.5, n_feats=25):
    high_var = VarianceThreshold(var_thresh).fit_transform(feat_df).columns
    pred_mi = pd.Series(data=mutual_info_classif(X=feat_df[high_var], y=target, discrete_features=False,
                                                 n_neighbors=10, random_state=0, n_jobs=-1), index=high_var)
    pred_feats = pred_mi.sort_values(ascending=False, inplace=True)[:n_feats].index
    logger.info('Features with variance > 0.5 to use as helpers.')
    logger.info(pprint.pformat(pred_feats, compact=True, width=60))
    return pred_feats


def undersampler(feature_df, class_labels, sampler='all', scaling=StandardScaler, random_state=0, *args, **kwargs):
    subsamples_dir = '{}/subsamples/'.format(os.environ.get('MODELS_DIR'))
    if type(sampler) is str and os.path.isfile('{}{}_undersamples_ind.pkl'.format(subsamples_dir, sampler)):
        with open(os.path.isfile('{}{}_undersamples_ind.pkl'.format(subsamples_dir, sampler)), 'rb') as f:
            samples_ind = pickle.load(f)
        return samples_ind
    rus = RandomUnderSampler(random_state=random_state, **kwargs)
    near_miss_one = NearMiss(n_jobs=-1, version=1, **kwargs)
    near_miss_two = NearMiss(n_jobs=-1, version=2, **kwargs)
    near_miss_three = NearMiss(n_jobs=-1, version=3, **kwargs)
    condensed = CondensedNearestNeighbour(n_jobs=-1, **kwargs)
    default_samplers = {'RandomUnderSampler': rus, 'NearMiss1': near_miss_one, 'NearMiss2': near_miss_two, 'NearMiss3':
                                              near_miss_three, 'CondensedNN': condensed}
    if sampler == 'auto':
        sampler_dict = default_samplers
    elif type(sampler) is str and sampler in default_samplers.keys():
        sampler_dict = {sampler: default_samplers[sampler]}
    else:
        sampler_dict = sampler
    samples_ind = dict()
    scaled_df = scaling(*args, **kwargs).fit_transform(X=feature_df)
    for name, sampler in sampler_dict.items():
        x_samp, y_samp = sampler.fit_resample(scaled_df, y=class_labels)
        print('Sampled {} with {} samples out of {}'.format(name, x_samp.shape[0], scaled_df.shape[0]))
        x_out_ind = feature_df.index.difference(x_samp.index)
        samples_ind[name] = (x_samp.index, x_out_ind)
    if not os.path.isdir(subsamples_dir):
        os.mkdir(subsamples_dir)
    subsamples_path = '{}{}_undersamples_ind.pkl'.format(subsamples_dir, str(sampler))
    with open(subsamples_path, 'wb') as f:
        pickle.dump(samples_ind, f)
    return samples_ind


def cross_val(feat_df, target, sampler, estimator, score_dict, name_dict):
    results = cross_validate(estimator=estimator, X=feat_df, y=target, scoring=score_dict, cv=sampler, n_jobs=-1,
                             return_train_score=True, return_indices=True, error_score='raise')
    logger.info(['Dev Scores: {}'.format(' '.join(['{:.5f}'.format(s.train_score) for s in results]))])
    logger.info(['Eval Scores: {}'.format(' '.join(['{:.5f}'.format(s.test_score) for s in results]))])


def preprocessing_pipe(fs, feat_cols, score_dict, cv_dir, jmem=None, n_jobs=-1):
    insol_label, sol_label = fs.y.count_values().index.tolist()
    insol_counts, sol_counts = fs.y.count_values().tolist()
    for dev, eval in StratifiedGroupKFold().split(X=fs.X.feat_frame, y=fs.y, groups=fs.y):
        feat_labels = feature_sampler(fs=fs, feat_df=fs.X.feat_frame)
        for feature, class_ind in feat_labels.items():
            for sample_method in ['RandomUnderSampler', 'NearMiss2']:
                pred_feats = get_pred_feats(fs.X.feat_frame.iloc[dev], fs.y.iloc[dev])
                sampled, dropped = undersampler(fs.X.feat_frame[pred_feats.extend(feature)], sampler=sample_method,
                                                class_labels=class_ind)[sample_method]
                sol_sampled = fs.y[fs.y == sol_label].iloc[sampled]
                insol_sampled = fs.y[fs.y == insol_label].iloc[sampled]
                logger.info('{} chose {} of {} soluble and {} of {} insoluble compounds'.format(sample_method,
                                                                                                sol_sampled.size,
                                                                                                sol_counts,
                                                                                                insol_sampled.size,
                                                                                                insol_counts))
    prepipe = Pipeline(steps=[('varthresh', VarianceThreshold()), ('scaler', 'passthrough'), ('corr', 'passthrough'),
                              ('redund', 'passthrough')])


def redundancy_importance():
    # Estimate importance/info -> Remove most redundant feature(s)
    # Importance methods: Lasso[LarsIC], [C]MI Algos, C4.5/ExtraTrees, OLS
    # Redundancy: Covariance/correlation matrix, [C]MI, VIF

    # Preprocessing -> Feature Selection -> Model Estimator
    # Variance Thresholding + Constitutional Descriptors + Interpretable (Column Transformers) ->
    var_thresh = VarianceThreshold()
    # Scaling (Robust, Asinh, or None) ->
    asinh_tform = FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh,
                                      check_inverse=False, feature_names_out='one-to-one')
    robust_tform = RobustScaler(unit_variance=True)
    binner = KBinsDiscretizer(encode='ordinal', strategy='kmeans')
    # binner = ColumnTransformer(transformers=[('cont_bin', KBinsDiscretizer(encode='ordinal', strategy='kmeans'),
    # bin_feats.tolist())],remainder='passthrough', force_int_remainder_cols=False,verbose_feature_names_out=False, sparse_threshold=0.05)
    # Feature Selection -> {Extra Trees, VIF, MI} (Later use CMI/FMIM)
    xtrees = ExtraTreeClassifier(random_state=0, class_weight='balanced')
    feat_elim = RFE(estimator=xtrees, importance_getter='feature_importances_')

    # Models: {LogisticCV, BalancedRF, MLP}
    # Must scale inputs before.
    # logcv = LogisticRegression(class_weight='balanced')

    # mlp = MLPClassifier(random_state=0)
    # ("var_thresh", var_thresh)
    splits = NearMiss
    # run_dict = {'params': list(), 'results': list(), 'models': list(), 'indices': list(), 'final_feats': list(),
    # 'scores': list()}
    run_list = list()
    best_score = (10000, dict(), 0.)
    cv_splits = 3
    param_frac = 5
    # indices_list = StratifiedGroupKFold(n_splits=cv_splits).split(X=fs.X.feat_frame[feat_cols], y=fs.y)
    param_grid = ParameterGrid(param_grid={  # 'binner__cont_bin__n_bins': [5, 20],
        'selector__estimator__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'selector__n_features_to_select':                [25, 50, 100],
        'selector__step':                                [0.1],
        'model__n_estimators':                           [10, 50, 250],
        'model__max_depth':                              [10, 15, 25, None]})
    param_list = list(param_grid)
    # pipe = Pipeline(steps=[("binner", binner), ("selector", feat_elim), ("model", brf)], memory=jmem, verbose=True)
    logger.info('Results from Hyperparameter Optimization')
    the_best = dict()
    final_feats = list()
    results, samples_wts, indices = {'true': {'dev': [], 'eval': []}, 'rf': {'dev': [], 'eval': []}}, {'dev':  [],
                                                                                                       'eval': []}, {
        'dev':  [],
        'eval': []}
    used_params, models = {'rf': []}, {'rf': []}
    param_ints = []
    cv = -1
    labels = pd.Series(fs.y, index=fs.X.feat_frame.index)
    for dev, ev in StratifiedKFold(n_splits=cv_splits).split(X=fs.X.feat_frame[feat_cols], y=labels.squeeze()):
        cv = cv + 1
        X_dev = fs.X.feat_frame[feat_cols].iloc[dev, :]
        y_dev = labels.squeeze().iloc[dev]
        X_eval = fs.X.feat_frame[feat_cols].iloc[ev, :]
        y_eval = labels.squeeze().iloc[ev]
        # print(cv, y_dev.size, y_eval.size)
        assert y_dev.nunique() == 2
        assert y_eval.nunique() == 2
        sklearn.utils.check_X_y(X=X_dev, y=y_dev)
        sklearn.utils.check_X_y(X=X_eval, y=y_eval)
        results['true']['dev'].append(y_dev)
        results['true']['eval'].append(y_eval)
        indices['dev'].append(dev)
        indices['eval'].append(ev)
        samples_wts['dev'].append(compute_sample_weight(class_weight='balanced', y=y_dev))
        samples_wts['eval'].append(compute_sample_weight(class_weight='balanced', y=y_eval))
    # while i <= len(param_list) < param_frac:


def param_run(param_list):
    i = -1
    param_list.append(param_list[np.random.random_integers(low=0, high=len(param_list))])
    param_list.append([param_ints])
    i += 1
    for run_num, params in enumerate(param_list):
        # print(type(params))
        pprint.pp(params, compact=True)

        print(samples_wts.items())
        pos_label = np.sort(y_dev.unique())[0]
        used_params['rf'].append(params)
        models['rf'] = list()
        # set_pipe = sklearn.pipeline.clone(pipe).set_params(**param_list[params])
        labels = pd.Series(fs.y, index=fs.X.feat_frame.index)
        insols = labels[labels == np.sort(labels.unique())[0]].index
        dense_feats = feat_cols.intersection(fs.X.cont.intersection(fs.X.dense))
        nondense_feats = feat_cols.difference(dense_feats)

        # pprint.pp(bin_feats, compact=True)
        # pprint.pp(nondense_feats, compact=True)
        if False:
            bin_feats = dense_feats[fs.X.feat_frame[dense_feats].loc[insols].nunique(axis=0) > int(1.5 * params[
                'binner__cont_bin__n_bins'])]
            nondense_feats = feat_cols.difference(bin_feats)
            X_binned = KBinsDiscretizer(n_bins=params['binner__cont_bin__n_bins'], encode='ordinal',
                                        strategy='kmeans').fit_transform(X_dev.copy()[bin_feats],
                                                                         y_dev.copy())
            print(type(X_binned))
        else:
            bin_feats = dense_feats[fs.X.feat_frame[dense_feats].nunique(axis=0) > int(fs.X.feat_frame.shape[0] /
                                                                                       3)]
            nondense_feats = feat_cols.difference(bin_feats)
            X_binned = X_dev[bin_feats]
        xtrees = ExtraTreesClassifier(random_state=0, min_samples_split=5, max_features=5,
                                      min_weight_fraction_leaf=params[
                                          'selector__estimator__min_weight_fraction_leaf']).fit(X_binned, y_dev,
                                                                                                sample_weight=
                                                                                                samples_wts['dev'][cv])

        eliminator = RFE(estimator=xtrees, n_features_to_select=params['selector__n_features_to_select'],
                         step=params['selector__step'], verbose=2).fit(X_binned, y_dev,
                                                                       sample_weight=samples_wts['dev'][cv])
        survived_feats = pd.Index(eliminator.get_feature_names_out(X_binned.columns))
        if nondense_feats.size > 10:
            # lars = LassoLarsIC(max_iter=1000, eps=0.005)
            # lars_df = RFE(estimator=mic, n_features_to_select=int(np.ceil(params[
            # 'selector__n_features_to_select'] /4))).fit_transform(fs.X.feat_frame[nondense_feats], y=fs.y)
            # mic = mutual_info_classif(fs.X.feat_frame[nondense_feats], y=labels, n_neighbors=7, random_state=0, n_jobs=-1)
            select_nondense = SelectPercentile(
                score_func=partial(mutual_info_classif, n_jobs=-1, discrete_features=True, random_state=0,
                                   n_neighbors=1),
                percentile=0.25).fit_transform(X=fs.X.feat_frame[nondense_feats], y=labels)
        else:
            select_nondense = fs.X.feat_frame[nondense_feats]
        logger.debug('Survived feats from RFE: ExtraTrees & LassoLarsIC')
        logger.debug(pprint.pformat(survived_feats))
        logger.debug(pprint.pformat(select_nondense.columns))
        tog_feats = survived_feats.union(select_nondense.columns)
        final_feats.append(tog_feats)
        brf = BalancedRandomForestClassifier(n_estimators=params['model__n_estimators'], random_state=0,
                                             class_weight='balanced', min_weight_fraction_leaf=params[
                'selector__estimator__min_weight_fraction_leaf'], max_depth=params['model__max_depth'])
        models['rf'].append(brf.fit(X=X_dev[tog_feats], y=y_dev))
        results['rf']['dev'].append(brf.predict(X_dev[tog_feats]))
        results['rf']['eval'].append(brf.predict(X_eval[tog_feats]))
    scores = dict()
    logger.info('Score metrics for grid search using:')
    logger.info(pprint.pformat([params], sort_dicts=False, compact=True))
    logger.info('Matthews Correlation Coefficients')
    mcc_dev = [matthews_corrcoef(y_true=results['true']['dev'][i], y_pred=results['rf'][
        'dev'][i]) for i in list(range(cv_splits))]
    mcc_eval = [matthews_corrcoef(y_true=results['true']['eval'][i], y_pred=results['rf'][
        'eval'][i]) for i in list(range(cv_splits))]
    assert not all([np.equal(m, 0) for m in mcc_eval])
    score_dict['mcc'] = {'dev': mcc_dev, 'eval': mcc_eval}
    logger.info('Dev: {:.5f} {:.5f} {:.5f}'.format(*mcc_dev))
    logger.info('Eval: {:.5f} {:.5f} {:.5f}'.format(*mcc_eval))
    logger.info('Mean: Dev: {:.4f}     Eval: {:.4f}'.format(np.mean(mcc_dev), np.mean(mcc_eval)))
    logger.info('StDev: Dev: {:.4f}     Eval: {:.4f}'.format(np.std(mcc_dev), np.std(mcc_eval)))

    logger.info('Balanced Accuracy Scores')
    bac_dev = [balanced_accuracy_score(y_true=results['true']['dev'][i], y_pred=results['rf'][
        'dev'][i]) for i in list(range(cv_splits))]
    bac_eval = [balanced_accuracy_score(y_true=results['true']['eval'][i], y_pred=results['rf'][
        'eval'][i]) for i in list(range(cv_splits))]
    logger.info('Dev: {:.5f} {:.5f} {:.5f}'.format(*bac_dev))
    logger.info('Eval: {:.5f} {:.5f} {:.5f}'.format(*bac_eval))
    logger.info('Mean: Dev: {:.4f}     Eval: {:.4f}'.format(np.mean(bac_dev), np.mean(bac_eval)))
    logger.info('StDev: Dev: {:.4f}     Eval: {:.4f}'.format(np.std(bac_dev), np.std(bac_eval)))
    score_dict['bac'] = {'dev': bac_dev, 'eval': bac_eval}

    run_list.append((params, indices, results, final_feats, score_dict))
    if np.mean(mcc_eval) > best_score[2]:
        best_score = (run_num, params, np.mean(mcc_eval))
        '''
        if len(the_best.keys()) == 0 or np.mean(mcc_eval) > the_best['mcc']:
            the_best['mcc'] = np.mean(mcc_eval)
            the_best['model'] = models['rf'][cv]

        plot_results(results_dict=results, models=models, X=fs.X.feat_frame[feat_cols],
                     indices=[x[1] for x in indices],
                     weights=samples_wts,
                     save_dir=run_dir)
        '''

        '''    
        search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=StratifiedGroupKFold(), n_jobs=-1,
                              scoring=balanced_accuracy_score, return_train_score=True, verbose=3, error_score='raise')
        grid_search = GridSearchCV(estimator=pipe, factor=5, param_grid=param_grid, resource="model__n_iter",
                                          max_resources=1000,
                                          cv=StratifiedGroupKFold, n_jobs=n_jobs,
                                          min_resources=100, random_state=0, scoring=matthews_corrcoef)
        '''

    run_dir = '{}run_{}/'.format(cv_dir, run_num)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    try:
        with open(run_dir, 'wb') as f:
            pickle.dump(tuple(run_list), f)
        with open('{}best_scores.csv'.format(run_dir), 'w', encoding='utf-8') as f:
            [f.write(str(a)) for a in best_score]
    except:
        print('Out of pickles...')
    plot_det_curves(estimators=models, data=(fs.X.feat_frame[feat_cols], fs.y), indices=indices,
                    sample_wt=samples_wts, save_dir=run_dir)
    return run_list


def run_cv_search(fs, use_feats, cv_dir, score_dict):
    # X = fs.X.feat_frame[use_feats]
    # logger.info(X.head())
    # jmem_dir = "C:/Users/mmanning/PycharmProjects/data/joblib_cache/joblib/"
    # jmem = Memory(location=jmem_dir)
    # jmem.reduce_size(age_limit=datetime.timedelta(seconds=3600))
    run_list = pipeline(fs=fs, feat_cols=use_feats, score_dict=score_dict, cv_dir=cv_dir)
    #                    jmem=jmem)
    exit()
    pl.fit(X=X, y=y)
    print(pl.score(X=X, y=y), flush=True)
    search.fit(X=X, y=y)
    try:
        with open('{}best_pipeline.html'.format(cv_dir), 'w') as f:
            f.write(estimator_html_repr(search.best_estimator_))
    except:
        print('No HTML today.')
    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_hdf5('{}halving_search_results_df.hdf'.format(cv_dir), complevel=9, nan_rep='NaN')
    with open('{}best_estimator.pkl'.format(cv_dir), 'wb') as g:
        g.write(search.best_estimator_)


# Feed best_estimator_ into Pipeline's params. Use CV splitter to retrain-test to use in scoring/displays.

def plot_det_curves(estimators, data, indices, sample_wt, save_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, det_curve, roc_curve

    insol_label = np.unique(data[1])[0]
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))
    fig.set_dpi(600)
    roc_dict, det_dict = dict(), dict()
    for name, clf in estimators.items():
        roc_dict[name], det_dict[name] = list(), list()
        for cv, (dev, ev) in enumerate(indices.values()):
            # X_train, X_test = data[1].iloc[indices]
            # y_train, y_test = data
            roc_dict[name].append(roc_curve(y_true=data[1].iloc[ev], y_score=clf.predict(data[1].iloc[ev]),
                                            pos_label=insol_label,
                                            sample_wt=sample_wt[cv]))
            det_dict[name].append(det_curve(y_true=data[1].iloc[ev], y_score=clf.predict(data[1].iloc[ev]),
                                            pos_label=insol_label,
                                            sample_wt=sample_wt[cv]))
            # RocCurveDisplay.from_estimator(clf, X_test, y_test, pos_label=0, sample_weight=sample_wt)
            # DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name, pos_label=0,
            # sample_weight=sample_wt)
        DetCurveDisplay(fpr=np.mean([dc.fpr for dc in det_dict[name]]), fnr=np.mean([dc.fnr for dc in det_dict[name]]),
                        estimator_name=name, pos_label=insol_label).plot(ax=ax_det, name='Mean {}'.format(name))
        RocCurveDisplay(fpr=np.mean([dc.fpr for dc in roc_dict[name]]), tpr=np.mean([dc.tpr for dc in roc_dict[
            name]]), estimator_name=name, pos_label=insol_label).plot(ax=ax_roc, name='Mean {}'.format(name))

    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax_det.set_title("Detection Error Tradeoff (DET) Curves")

    ax_roc.grid(linestyle="--")
    ax_roc.legend(loc='lower right')
    ax_det.grid(linestyle="--")
    ax_det.legend(loc='lower left')
    plt.legend()
    fig_dir = '{}plots/'.format(save_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    try:
        fig.savefig('{}DET_ROC_figs.png'.format(fig_dir), transparent=True)
    except:
        plt.show()
