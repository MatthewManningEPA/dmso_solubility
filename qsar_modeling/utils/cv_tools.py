import copy
import itertools
import logging

import numpy as np
import pandas as pd
from numpy.random import get_bit_generator
# from numpy.random._examples.numba.extending_distributions import bit_generator
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


def data_shuffling(x, method="pandas", stratified=False, **kwargs):
    if method == "numpy":
        raise NotImplementedError
        shuffled = copy.deepcopy(x)
        np.random.Generator(get_bit_generator()).shuffle(x)
    elif method == "pandas":
        if not stratified:
            shuffled = x.sample(frac=1.0, **kwargs)
        else:
            if type(stratified) is pd.Series:
                shuffled = x.groupby(by=stratified).sample(frac=1.0, **kwargs)
            else:
                shuffled = x.sample(frac=1.0, **kwargs)
    return shuffled


def get_split_ind(data, labels, n_splits=5, splitter=None, **splitter_kws):
    if splitter is None:
        splitter = StratifiedKFold
    indices_list = list()
    for train_ind, test_ind in splitter(n_splits=n_splits, **splitter_kws).split(X=data, y=labels.astype(int)):
        if train_ind.size == 0 or test_ind.size == 0:
            print('One of the indices is length 0!!!')
            raise ValueError
        indices_list.append((train_ind, test_ind))
    assert len(indices_list) == n_splits
    return tuple(indices_list)


def split_df(data, labels, indices_list=None, n_splits=5, splitter=StratifiedKFold, **splitter_kws):
    if indices_list is None:
        indices_list = get_split_ind(data, labels, n_splits, splitter=splitter, **splitter_kws)
    split_list = list()
    assert len(indices_list) == n_splits
    for (train_ind, test_ind) in indices_list:
        train_data = data.iloc[train_ind]
        train_labels = labels.iloc[train_ind].squeeze()
        test_data = data.iloc[test_ind]
        test_labels = labels.iloc[test_ind].squeeze()
        split_list.append((train_data, train_labels, test_data, test_labels))
    return split_list


def package_output(train_y, test_y, model_tuple):
    true_predict_tuples = (train_y, model_tuple[1]), (test_y, model_tuple[2])
    return true_predict_tuples


def score_cv_results(true_predict_tuples, scoring_list='balanced', scoring_dict=None):
    if scoring_dict is None:
        if scoring_list == 'balanced':
            scoring_list = [matthews_corrcoef, balanced_accuracy_score]
        scoring_dict = dict([(s, (s.__name__)) for s in scoring_list])
        scores_dict = dict([(s.__name__, {'train': list(), 'test': list()}) for s in scoring_list])
    else:
        scores_dict = dict([(k, {'train': list(), 'test': list()}) for k, v in scoring_dict])
    for scorer, scorer_name in scoring_dict.items():
        dev_scores = [scorer(a[0], a[1]) for (a, b) in true_predict_tuples]
        eval_scores = [scorer(b[0], b[1]) for (a, b) in true_predict_tuples]
        scores_dict[scorer_name]['train'] = dev_scores
        scores_dict[scorer_name]['test'] = eval_scores
    return scores_dict


def log_score_summary(scores_dict, level=10, score_file=None, score_logger=None):
    score_list = list()
    for scorer_name, split_scores_dict in scores_dict.items():
        for stat_name, stat in zip(['Mean', 'StDev', 'Median', 'Min', 'Max'],
                                   [np.mean, np.std, np.median, np.min, np.max]):
            for split_name, scores in split_scores_dict.items():
                score_list.append('{}: {:.5f}'.format(stat_name, stat([[s] for s in scores])))
            print('\t'.join(score_list))
            if score_logger is not None:
                score_logger.log(level=level, msg='\t'.join(score_list))
    return score_list


def quadratic_splits(grouped_sers, n_splits=5):
    # Takes separate groups (each in separate Series), gets n_splits splits, and yields indices of test set.
    test_list, train_list = list(), list()
    indices = [s.copy().index.tolist() for s in grouped_sers]
    [np.random.shuffle(idx) for idx in indices]
    nested_splits = list()
    for ind in indices:
        spaces = np.linspace(0, len(ind) - 1, num=n_splits + 1, dtype=int)
        nested_splits.append([ind[int(a):int(b)] for a, b in itertools.pairwise(spaces)])
    return nested_splits


def get_quadratic_test_folds(grouped_sers, n_splits=5):
    fold_list = list()
    quad_splits = quadratic_splits(grouped_sers, n_splits)
    # print(quad_splits)
    for qx in quad_splits:
        for fold_id, idxs in enumerate(qx):
            fold_list.append(pd.Series(data=[fold_id for _ in idxs], index=idxs))
    return pd.concat(fold_list)
