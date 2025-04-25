import logging
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, matthews_corrcoef

import cv_tools
import samples
import scoring


def get_predictions(
    model, train_data, train_labels, test_data, sample_wts=None, **model_params
):
    if model_params:
        model.set_params(**model_params)
    model.fit(,
    train_predict = pd.Series(model.predict(X=train_data), index=train_data.index)
    test_predict = pd.Series(model.predict(X=test_data), index=test_data.index)
    train_proba = pd.DataFrame(
        model.predict_proba(X=train_data), index=train_data.index
    )
    test_proba = pd.DataFrame(model.predict_proba(X=test_data), index=test_data.index)
    return model, train_predict, test_predict, train_proba, test_proba


def balanced_forest(
    train_data, train_labels, test_data, sample_wts=None, **model_params
):
    n_trees = 100
    model = BalancedRandomForestClassifier(
        n_estimators=n_trees,
        max_depth=15,
        n_jobs=-1,
        random_state=0,
        sampling_strategy="auto",
        replacement=False,
        verbose=0,
        class_weight="balanced_subsample",
        bootstrap=True,
    )
    return get_predictions(
        model,
        train_data,
        train_labels,
        test_data,
        sample_wts=sample_wts,
        **model_params
    )


# min_weight_fraction_leaf=0.05,


def logistic_clf(train_data, train_labels, test_data, sample_wts=None, **model_params):
    model = LogisticRegressionCV(
        max_iter=500,
        solver="newton-cg",
        class_weight="balanced",
        cv=3,
        n_jobs=-1,
        scoring=make_scorer(matthews_corrcoef),
        random_state=0,
    )
    return get_predictions(model, train_data, train_labels, test_data, **model_params)


def cv_predictions(estimator, feature_df, labels, cv=None, sample_wts=None):
    score_list = list()
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        feature_df, labels, splitter=cv
    ):
        model, train_predict, test_predict, train_proba, test_proba = get_predictions(
            estimator, dev_X, dev_y, eva_X, eva_y, sample_wts
        )
        score_dict = cv_tools.score_cv_results(
            cv_tools.package_output(dev_y, eva_y, (train_predict, test_predict))
        )
        score_list.append(score_dict)
        cv_tools.log_score_summary(score_dict, score_logger=logging.getLogger())


def cv_model_documented(
    input_X,
    input_y,
    cv_model,
    save_dir,
    model_params=None,
    cv_splitter=None,
    sweight=None,
    **splitter_kw
):
    if model_params is None:
        cv_model = clone(cv_model())
    else:
        cv_model = clone(cv_model()).set_params(**model_params)
    cv = 0
    dev_score_list, eva_score_list = list(), list()
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        input_X, input_y, splitter=cv_splitter, **splitter_kw
    ):
        cv += 1
        cv_dir = "{}cv_{}/".format(save_dir, cv)
        model_dir = cv_dir  # '{}{}/'.format(cv_dir, model_name)
        if not os.path.isdir(cv_dir):
            os.makedirs(cv_dir, exist_ok=True)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model_obj_path = "{}model_obj.pkl".format(model_dir)
        cv_dev_path = "{}dev_true.csv".format(cv_dir)
        cv_eva_path = "{}eval_true.csv".format(cv_dir)
        cv_dev_pred_path = "{}dev_pred.csv".format(model_dir)
        cv_eva_pred_path = "{}eval_pred.csv".format(model_dir)
        dev_y.to_csv(cv_dev_path)
        eva_y.to_csv(cv_eva_path)
        model_inst = clone(cv_model).fit(dev_X, dev_y, sample_weight=sweight)
        with open(model_obj_path, "wb") as f:
            pickle.dump(model_inst, f)
        dev_pred = pd.Series(data=model_inst.predict(dev_X), index=dev_y.index)
        eva_pred = pd.Series(data=model_inst.predict(eva_X), index=eva_y.index)
        dev_pred.to_csv(cv_dev_pred_path, index_label="INCHI_KEY")
        eva_pred.to_csv(cv_eva_pred_path, index_label="INCHI_KEY")
        dev_score_dict = scoring.score_model(model_inst, dev_X, dev_y)
        eva_score_dict = scoring.score_model(model_inst, eva_X, eva_y)
        dev_score_list.append(dev_score_dict)
        eva_score_list.append(eva_score_dict)
        tn, fp, fn, tp = samples.get_confusion_samples((eva_y, eva_pred))
        for iks, s in zip([tn, fp, fn, tp], ["tn", "fp", "fn", "tp"]):
            samples.get_sample_info(iks).to_csv(
                "{}{}_eval_samples.csv".format(model_dir, s), index_label="INCHI_KEY"
            )
    return dev_score_list, eva_score_list


def combined_rus_cv_results(
    feature_df,
    labels,
    model,
    model_params,
    save_dir,
    n_rus=3,
    cv_splitter=None,
    sweight=None,
    **splitter_kw
):
    dev_score_list, eva_score_list = list(), list()
    for r in np.arange(n_rus):
        rus_dir = "{}rus_{}/".format(save_dir, r)
        if not os.path.isdir(rus_dir):
            os.makedirs(rus_dir, exist_ok=True)
        else:
            continue
        rus_state = 1000 * r
        X_under, y_under = RandomUnderSampler(random_state=rus_state).fit_resample(
            feature_df, labels
        )
        print(X_under.shape)
        rus_dev_scores, rus_eva_scores = cv_model_documented(
            X_under,
            y_under,
            model,
            rus_dir,
            model_params,
            cv_splitter=cv_splitter,
            sweight=sweight,
            **splitter_kw
        )
        assert len(rus_dev_scores) > 0 and len(rus_eva_scores) > 0
        for dl in [rus_dev_scores, rus_eva_scores]:
            for cv_num, d in enumerate(dl):
                d["RUS"] = r
                d["CV"] = cv_num
        dev_score_list.extend(rus_dev_scores)
        eva_score_list.extend(rus_eva_scores)
    if len(dev_score_list) == 0 or len(eva_score_list) == 0:
        raise RuntimeWarning
        print("Model score list is empty")
        return pd.DataFrame(), pd.DataFrame(), list(), list()
    dev_summary = [
        dict([(k, v) for k, v in d.items() if k != "RUS" and k != "CV"])
        for d in dev_score_list
    ]
    eva_summary = [
        dict([(k, v) for k, v in d.items() if k != "RUS" and k != "CV"])
        for d in eva_score_list
    ]
    # print("Eva summary: {}".format(eva_summary))
    dev_score_df = scoring.summarize_scores(dev_summary)
    eva_score_df = scoring.summarize_scores(eva_summary)
    assert not dev_score_df.empty and not eva_score_df.empty
    dev_score_path = "{}dev_score_summary.csv".format(save_dir)
    eva_score_path = "{}eval_score_summary.csv".format(save_dir)
    dev_score_df.to_csv(dev_score_path, index_label="Metric")
    eva_score_df.to_csv(eva_score_path, index_label="Metric")
    total_dev_scores = pd.DataFrame.from_dict(dev_score_list)
    total_eva_scores = pd.DataFrame.from_dict(eva_score_list)
    return dev_score_df, eva_score_df, total_dev_scores, total_eva_scores
