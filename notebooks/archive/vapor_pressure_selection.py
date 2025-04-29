import logging
import os
import pickle

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.frozen import FrozenEstimator
# from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.pipeline import clone

import sample_clusters
from build_preprocessor import get_standard_preprocessor
from dataset_creation import padel_candidate_features
from fuzzy_controller import select_feature_subset

mcc = make_scorer(matthews_corrcoef)


def get_clf_model(model_name):
    if "log" in model_name:
        grove_model = LogisticRegressionCV(
            scoring=mcc,
            solver="newton-cholesky",
            tol=2e-4,
            cv=5,
            max_iter=10000,
            class_weight="balanced",
            n_jobs=-4,
        )
    elif "ridge" in model_name:
        from sklearn.linear_model import RidgeClassifierCV

        grove_model = RidgeClassifierCV(scoring=mcc, class_weight="balanced")
    elif "svc" in model_name:
        from sklearn.svm import SVC

        if "poly" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="poly")
        elif "sigmoid" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="sigmoid")
        elif "linear" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="linear")
        else:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="rbf")
    elif "passive" in model_name:
        from sklearn.linear_model import PassiveAggressiveClassifier

        grove_model = PassiveAggressiveClassifier(
            C=5.0, class_weight="balanced", random_state=0
        )
    elif "xtra" in model_name:
        grove_model = ExtraTreesClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    else:
        grove_model = RandomForestClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    return grove_model


def get_regression_model(model_name):
    if "line" in model_name:
        from sklearn.linear_model import LinearRegression

        grove_model = LinearRegression()
    elif "elastic" in model_name:
        from sklearn.linear_model import ElasticNetCV

        grove_model = ElasticNetCV(
            l1_ratio=[0.25, 0.5, 0.75, 0.9],
            tol=1e-4,
            max_iter=10000,
            selection="random",
        )
    elif "hub" in model_name:
        from sklearn.linear_model import HuberRegressor

        grove_model = HuberRegressor(max_iter=1000, tol=1e-04)
    elif "sgd" in model_name:
        from sklearn.linear_model import SGDRegressor

        grove_model = SGDRegressor(max_iter=5000, random_state=0)
    elif "ridge" in model_name:
        from sklearn.linear_model import RidgeCV

        grove_model = RidgeCV()
    elif "lasso" in model_name:
        from sklearn.linear_model import LassoCV

        grove_model = LassoCV(random_state=0)
    elif "xtra" in model_name:
        grove_model = ExtraTreesRegressor(
            max_leaf_nodes=300,
            min_impurity_decrease=0.005,
        )
    elif "krr" in model_name:
        from sklearn.kernel_ridge import KernelRidge

        if "poly" in model_name:
            grove_model = KernelRidge(kernel="polynomial")
        elif "rbf" in model_name:
            grove_model = KernelRidge(kernel="rbf")
        elif "sigmoid" in model_name:
            grove_model = KernelRidge(kernel="sigmoid")
        else:
            grove_model = KernelRidge(kernal="linear")
    elif "gauss" in model_name:
        from sklearn.gaussian_process import GaussianProcessRegressor

        grove_model = GaussianProcessRegressor(n_restarts_optimizer=3, normalize_y=True)
    elif "pls" in model_name:
        from sklearn.cross_decomposition import PLSRegression

        grove_model = PLSRegression(
            n_components=1, max_iter=5000, tol=1e-05
        ).set_output(transform="pandas")
    else:
        grove_model = RandomForestRegressor(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            bootstrap=False,
        )
    return grove_model


def main(model_name, importance_name):
    # data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/test_data/Vapor pressure OPERA/Vapor pressure OPERA/"
    # opera_dir = "{}test_train_split/".format(data_dir)
    """
    train_dir = "{}Vapor pressure OPERA T.E.S.T. 5.1 training.tsv".format(opera_dir)
    train_dir = "{}Vapor pressure OPERA Padelpy webservice single training.tsv".format(
        data_dir
    )
    # train_data = pd.read_csv(train_dir, delimiter="\t")
    train_data.set_index(keys=train_data.columns[0], inplace=True)
    labels = train_data[train_data.columns[0]].copy()
    feature_df = train_data.drop(columns=train_data.columns[0])
    logger.info(train_data.head())
    logger.info(train_data.shape)
    """
    raise DeprecationWarning
    select_params = {
        "corr_method": "kendall",
        "xc_method": "kendall",
        "max_features_out": 30,
        "fails_min_vif": 5,
        "fails_min_perm": 6,
        "fails_min_sfs": 6,
        "features_min_vif": 15,
        "features_min_perm": 15,
        "features_min_sfs": 20,
        "thresh_reset": 0.05,
        "thresh_vif": 15,
        "thresh_perm": 0.0025,
        "thresh_sfs": -0.005,
        "thresh_xc": 0.95,
        "max_trials": 200,
        "cv": 5,
        "importance": True,
        "scoring": None,
    }
    data_transform = "asinh"
    data_dir = "{}enamine_transform_test/".format(os.environ.get("MODEL_DIR"))
    opera_dir = data_dir
    search_dir = "{}test_train_split/{}_1/".format(
        opera_dir, "_".join([model_name, data_transform])
    )
    os.makedirs(search_dir, exist_ok=True)
    train_data, labels, best_corrs, cross_corr, scaler = process_selection_data(
        save_dir=data_dir,
        select_params=select_params,
        transform=data_transform,
    )
    search_features = padel_candidate_features()
    if labels.nunique() > 2:
        selection_models = {
            "predict": get_regression_model(model_name),
            "permutation": get_regression_model(model_name),
            "importance": get_regression_model(importance_name),
            "vif": LinearRegression(),
        }
    else:
        selection_models = {
            "predict": get_clf_model(model_name),
            "permutation": get_clf_model(model_name),
            "importance": get_clf_model(importance_name),
            "vif": LinearRegression(),
        }
    trained_model = get_clf_model(model_name).fit(train_data, labels)
    with open("{}best_model.pkl".format(search_dir), "wb") as f:
        pickle.dump(FrozenEstimator(trained_model), f)
    # members_dict = membership(feature_df, labels, grove_cols, search_dir)
    dev_idx, eval_idx = [
        (a, b)
        for a, b in StratifiedKFold(shuffle=True, random_state=0).split(
            train_data, labels
        )
    ][0]
    dev_data = train_data.iloc[dev_idx]
    dev_labels = labels[dev_data.index]
    eval_data = train_data.iloc[eval_idx]
    eval_labels = labels[eval_data.index]
    model_dict, score_dict, dropped_dict, best_features = select_feature_subset(
        train_df=dev_data,
        labels=dev_labels,
        target_corr=best_corrs,
        cross_corr=cross_corr,
        select_params=select_params,
        selection_models=selection_models,
        save_dir=search_dir,
    )
    if any([a is None for a in [model_dict, score_dict, dropped_dict, best_features]]):
        print(model_dict, score_dict, dropped_dict, best_features)
        raise ValueError
    trained_model = clone(selection_models["predict"]).fit(
        dev_data[best_features], dev_labels
    )
    train_score = trained_model.score(dev_data[best_features], dev_labels)
    test_score = trained_model.score(eval_data[best_features], labels[eval_data.index])
    print(
        "Best feature subset scores: Train: {:.5f}, Test: {:.5f}".format(
            train_score, test_score
        )
    )
    print("Using {} features".format(len(best_features)))
    print("Dropped features: ", dropped_dict.items())
    print(best_features)
    return None
    """
    for i, (col, members) in enumerate(list(members_dict.items())):
        if (
            labels[members][labels[members] == 0].size < 15
            or labels[members][labels[members] == 1].size < 15
        ):
            continue
        col_dir = "{}/{}/".format(search_dir, i)
        os.makedirs(col_dir, exist_ok=True)
        with sklearn.config_context(
            enable_metadata_routing=True, transform_output="pandas"
        ):
            model_dict[col], score_dict[col], dropped_dict, best_features = (
                select_feature_subset(
                    feature_df.loc[members][search_features],
                    labels[members],
                    None,
                    None,
                    select_params=None,
                    initial_subset=col,
                    save_dir=col_dir,
                )
            )
    """


if __name__ == "__main__":
    importance = "rfc"
    logger = logging.getLogger(name="selection")
    for md in ["svc_rbf", "xtra", "ridge", "rfc", "passive", "log"]:
        main(model_name=md, importance_name=importance)


def process_selection_data(
    feature_df=None,
    labels=None,
    dropped_dict=None,
    save_dir=None,
    select_params=None,
    scaler="standard",
    transform=None,
):
    raise DeprecationWarning
    # TODO: Delete function.
    preloaded = False
    drop_corr = False
    best_corrs, cross_corr = None, None
    if feature_df is None:
        feature_df, labels = sample_clusters.grab_enamine_data()
    if dropped_dict is None:
        dropped_dict = dict()
    combo_transform, scaler, cross_corr = get_standard_preprocessor(
        scaler=scaler, transform_func=transform, corr_params=select_params
    )
    combo_transform.fit(,
    with open("{}transformer.pkl".format(save_dir), "wb") as f:
        pickle.dump(combo_transform, f)
    train_df = combo_transform.transform(feature_df)
    return train_df, labels, best_corrs, cross_corr, scaler
