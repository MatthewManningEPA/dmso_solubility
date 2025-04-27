import copy
from inspect import signature

import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import _check_sample_weight

from dmso_utils import data_tools


def sample_wt_opt(func, sample_weight, *args, **kwargs):
    if (
        "sample_weight" in signature(func).parameters.keys()
        and sample_weight is not None
    ):
        return func(*args, sample_weight=sample_weight, **kwargs)
    else:
        return func(*args, **kwargs)


def get_sample_info(inchi_keys, source=None, labels=None, drop_dupes=False):
    # Returns QSAR-ready SMILES and INCHI strings for list of INCHI keys.
    prop_list = ["SMILES_QSAR", "INCHI"]
    meta_loaded = data_tools.load_metadata()
    if source is not None:
        if type(source) is not str:
            source = dict((k, v) for k, v in meta_loaded.items() if k in source)
        meta_loaded = dict([(source, meta_loaded[source])])
    if labels is not None:
        if type(labels) is not str:
            source = dict((k, v) for k, v in meta_loaded.items() if k in labels)
        meta_loaded = dict([(labels, meta_loaded[labels])])
    metadata = pd.concat([m[1][prop_list] for m in meta_loaded.values()])
    try:
        info = metadata.loc[inchi_keys]
    except:
        info = metadata[inchi_keys]
        print("Except worked!")
    return info


def get_dmso_source_label_df(inchi_keys=None, include_only=None, exclude=None):
    # Returns DataFrame listing DMSO solubility and data source for each INCHI key.
    raise DeprecationWarning
    meta_loaded = data_tools.load_metadata()
    data_dfs = list()
    for k, v in meta_loaded.items():
        if include_only is not None:
            if not any([s == k for s in include_only]):
                continue
        if exclude is not None:
            if any([s == k for s in exclude]):
                continue
        if inchi_keys is not None:
            idx = v[0].index.intersection(inchi_keys)
            if idx.empty:
                continue
        else:
            idx = v[0].index
        source, label = k.split("_")
        data_df = pd.DataFrame(index=idx)
        data_df["Source"] = source
        data_df["Solubility"] = label
        data_dfs.append(data_df)
    return pd.concat(data_dfs)


def weight_dmso_samples(
    inchi_idx, by, group_by=None, include_only=None, exclude=None, class_wt="balanced"
):
    # Returns sample weights based on solubility and data source.
    meta_df = get_dmso_source_label_df(
        inchi_idx, include_only=include_only, exclude=exclude
    )
    if group_by.lower() == "solubility" or (by.lower() == "source" and group_by):
        by_source = [
            meta_df[meta_df["Solubility" == u]] for u in meta_df["Source"].unique()
        ]
        weights = pd.concat([compute_sample_weight(class_wt, s) for s in by_source])
    elif group_by.lower() == "source" or (by.lower() == "solubility" and group_by):
        by_source = [
            meta_df[meta_df["Source" == u]] for u in meta_df["Source"].unique()
        ]
        weights = pd.concat([compute_sample_weight(class_wt, s) for s in by_source])
    elif by.lower() == "source":
        weights = compute_sample_weight(class_wt, meta_df["Source"])
    elif by.lower() == "solubility":
        weights = compute_sample_weight(class_wt, meta_df["Solubility"])
    return weights


def get_confusion_samples(true_predict_tuple, labels=(1, 0)):
    # Returns INCHI keys corresponding to True/False Positive/Negatives.
    # tn, fp, fn, tp == sklearn confusion matrix convention
    tn, fp, fn, tp = list(), list(), list(), list()
    if all([(type(t) is list or type(t) is tuple) for t in true_predict_tuple]):
        for tup in true_predict_tuple:
            if all([(type(t) is list or type(t) is tuple) for t in tup]):
                false_list = [get_confusion_samples(a) for a in true_predict_tuple]
                return false_list
            else:
                print("Something is very wrong")
                print(tup)
                raise RuntimeError
    else:
        tup = true_predict_tuple
        assert len(tup) == 2
        if labels is not None:
            bin_tuple = [t.map(dict(zip(labels, [0, 1]))) for t in tup]
        else:
            bin_tuple = tup
        diff = bin_tuple[1].subtract(bin_tuple[0])
        truth = diff[diff == 0].index
        pos = bin_tuple[0][bin_tuple[0] == 1].index
        neg = bin_tuple[0][bin_tuple[0] == 0].index
        fp = diff[diff == 1].index
        fn = diff[diff == -1].index
        tp = truth.intersection(pos)
        tn = truth.intersection(neg)
    return tn, fp, fn, tp


def safe_mapper(x, map):
    if x in map.keys():
        return map[x]
    else:
        return x


def weights_from_predicts(
    y_true, y_predict, predict_model, select_params, score_func, combo_type="best"
):
    if isinstance(y_predict, (pd.Series, pd.DataFrame)):
        test_scores = [y_predict]
    elif isinstance(y_predict, (list, tuple, set)):
        test_scores = list()
        for val in y_predict:
            if isinstance(val[score_func]["test"], (pd.Series, pd.DataFrame)):
                test_scores.append(val[score_func]["test"])
            elif all([(isinstance(a, (pd.Series, pd.DataFrame)) for a in val)]):
                test_scores.append(pd.concat(val[score_func]["test"]))
            else:
                print(val)
                raise ValueError
    else:
        raise ValueError
    if is_classifier(predict_model):
        new_weights = weight_by_proba(
            y_true=y_true,
            probs=test_scores,
            prob_thresholds=select_params["brier_clips"],
            combo_type=combo_type,
        )
    elif is_regressor(predict_model):
        new_weights = weight_by_error(
            y_true=y_true, predicts=test_scores, loss=score_func
        )
    else:
        raise ValueError
    assert isinstance(new_weights, pd.Series)
    return new_weights


def weight_by_error(y_true, predicts, loss=mean_squared_error):
    y_pred = predicts.copy()
    y_true = y_true.copy()
    if isinstance(y_pred, (list, tuple)):
        resids = pd.concat([y_true - p for p in y_pred], axis=1).max(axis=1)
    else:
        resids = y_true - y_pred
    # sample_losses = loss(y_true=y_true[y_pred.index], y_pred=y_pred, multioutput="raw_values")
    # print(sample_losses)
    return pd.Series(data=resids**2, index=y_true.index, name="losses")


def weight_by_proba(
    y_true,
    probs,
    prob_thresholds=(0, 1.0),
    label_type="binary",
    combo_type="best",
    class_labels=None,
):
    probs = copy.deepcopy(probs)
    y_true = y_true.copy()
    onehot_labels, onehot_normed = one_hot_conversion(
        y_true, label_type=label_type, class_labels=class_labels
    )
    if (
        isinstance(probs, (list, tuple, set))
        and len(probs) == 1
        and isinstance(probs[0], (pd.DataFrame, pd.Series))
    ):
        probs = probs[0]
    if isinstance(probs, pd.DataFrame):
        # sample_weights_raw = weigh_single_proba(onehot_true=onehot_normed, probs=probs, prob_thresholds=prob_thresholds)
        if prob_thresholds is not None:
            probs.clip(lower=prob_thresholds[0], upper=prob_thresholds[1], inplace=True)
            if class_labels is not None:
                probs.columns = class_labels
        sample_weights = probs.multiply(onehot_labels).sum(axis=1).squeeze()
        # pd.DataFrame(data=np.zeros_like(probs.to_numpy()), index=y_true, columns=probs.columns)
        # [ (p - (1-T)/n_classes) ]
        # where: p = probs, T = one-hot encoded classes
        # onehot_labels = OneHotEncoder(categories=y_true.tolist(), sparse_output=False, dtype=np.int16).fit_transform(y_true.to_frame())

        # print("Normed:\n{}".format(pprint.pformat(onehot_normed.shape)))
    elif isinstance(probs, pd.Series):
        sample_weights = probs
    elif (
        isinstance(probs, (list, tuple, set))
        and len(probs) > 1
        and not any([p is None for p in probs])
    ):
        if class_labels is not None:
            for p in probs:
                p.columns = class_labels
        unweighted = pd.concat(
            [p.multiply(onehot_labels).sum(axis=1).squeeze() for p in probs],
            axis=1,
        )
        # print("Unweighted len: {}\n".format(len(unweighted)))
        # unweighted_sols = pd.concat([p["Soluble"] for p in unweighted], axis=1)
        # unweighted_ins = pd.concat([p["Insoluble"] for p in unweighted], axis=1)
        # print("Unweighted Insolubles")
        # pprint.pp(unweighted_ins)
        # print("Unweighted:")
        # pprint.pp(unweighted)
        # unweighted_min = unweighted.min(axis=1)
        # unweighted_avg = unweighted.mean(axis=1)
        # print("Unweighted Minimum")
        # pprint.pp(unweighted_min)
        # print("Unweighted Average")
        # pprint.pp(unweighted_avg)
        # unweighted = unweighted.max(axis=1)
        if combo_type == "mean":
            sample_weights = unweighted.mean(axis=1)
        elif combo_type == "min":
            sample_weights = unweighted.min(axis=1)
        elif combo_type == "median":
            sample_weights = unweighted.median(axis=1)
        # elif combo_type == "max":
        else:
            sample_weights = unweighted.max(axis=1)
        # if "sq" in activate:
        #     pd.Series.apply()
    else:
        print("Unknown type for probabilities processing...")
        print(probs)
        print(type(probs))
        raise TypeError
    try:
        sample_weights.reindex_like(other=probs)
    except:
        # print(probs)
        sample_weights.set_index = probs.index
    print("\nSample weights:")
    sample_weights = pd.Series(
        data=_check_sample_weight(
            sample_weight=sample_weights, X=y_true, ensure_non_negative=True
        ),
        index=y_true.index,
    ).sort_index()
    assert not sample_weights.isna().any()
    return sample_weights


def one_hot_conversion(
    y_true, threshold="auto", label_type="binary", class_labels=None
):
    if not isinstance(y_true.squeeze(), pd.Series) and label_type == "binary":
        y_true = LabelBinarizer().fit_transform(y_true.to_frame())
    onehot_labels = pd.concat([y_true, 1 - y_true], axis=1)
    assert onehot_labels.shape == (y_true.shape[0], y_true.nunique())
    if class_labels is not None:
        onehot_labels.columns = class_labels
    if threshold == "auto" or threshold is None:
        threshold = pd.DataFrame(
            data=1.0 / y_true.nunique(),
            columns=onehot_labels.columns,
            index=onehot_labels.index,
        )
    try:
        onehot_normed = onehot_labels.sub(threshold)
    except TypeError:
        print(threshold)
        onehot_normed = onehot_labels.sub(1.0 / y_true.nunique())
    onehot_normed.columns = onehot_labels.columns
    return onehot_labels, onehot_normed
