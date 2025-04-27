import copy
import numbers
import pickle
import pprint
import random

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.feature_selection._from_model import (
    _get_feature_importances as get_importances,
)
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import _check_sample_weight
from stats import geometric_mean

import correlation_filter
import cv_tools
import math_tools
import samples
import scoring
import vif
from vif import calculate_vif, repeated_stochastic_vif


def fit_weighted(estimator, X, y=None, weights=None):
    if (
        HasMethods("sample_weight").is_satisfied_by(estimator)
        and weights is not None
        and weights.shape[0] == X.shape[0]
    ):
        weight_kwargs = {"sample_weight": weights}
    else:
        weight_kwargs = {}
    fit_model = estimator.fit(X=X, y=y, **weight_kwargs)
    return fit_model


class FuzzyAnnealer:

    ALLOW_DIRECT_SCORE = False

    def __init__(self, params, models, save_dir, **kwargs):
        # Constants
        self.params = params
        self.models = models
        self.save_dir = save_dir
        # State Dependent
        self._current_features = None
        self._current_score = None
        self.best_subset = None
        self.best_score = -999
        self.subset_scores = dict()
        self.chosen_subsets = list()
        self.best_model = None
        self.clean_up = False
        self.temp = 1
        # Inputs and their calculated values
        self.feature_df = None
        self.labels = None
        self.cross_corr = None
        self.sq_xcorr = None
        self.label_corr = None
        self.other_probs = list()
        self.proba_weight = None
        self.sample_weight = None
        self.best_preds = None

    # Setters and Getters

    @property
    def current_score(self):
        return self._current_score

    @current_score.setter
    def current_score(self, value=None):
        if tuple(sorted(self._current_features)) not in self.subset_scores.keys():
            if self.ALLOW_DIRECT_SCORE and value is not None:
                score = value
                self.subset_scores[tuple(sorted(self._current_features))] = score
            else:
                score, _ = self.score_subset(
                    tuple(sorted(self._current_features)),
                    sample_weight=self.sample_weight,
                )
            if not (
                isinstance(score, numbers.Real)
                or all([isinstance(s, numbers.Real) for s in score])
            ):
                print(score)
                raise ValueError
            self._current_score = score
        else:
            self._current_score = self.subset_scores[self._current_features]

    @current_score.getter
    def current_score(self):
        if tuple(sorted(self._current_features)) not in self.subset_scores.keys():
            self.current_score = None
        return self._current_score

    @property
    def current_features(self):
        return self._current_score

    @current_features.getter
    def current_features(self):
        return sorted(list(self._current_features))

    @current_features.setter
    def current_features(self, value):
        if any([pd.isnull(v) or v is None for v in value]):
            raise ValueError
        if isinstance(value, (str, int)):
            value = [value]
        else:
            value = list(set(value))
        if tuple(sorted(value)) not in self.subset_scores.keys():
            self._current_score, _ = self.score_subset(
                value, sample_weight=self.sample_weight
            )
        self._current_features = value

    def partial_fit(
        self,
        feature_df=None,
        labels=None,
        other_probs=None,
        initial_subset=None,
        proba_weight=None,
    ):
        self.subset_scores = dict()
        self.chosen_subsets = list()
        self.best_score = -999
        self.clean_up = False
        self.temp = 1
        self.cross_corr = None
        self.sq_xcorr = None
        self.label_corr = None
        self.other_probs = other_probs
        self.proba_weight = proba_weight
        self.sample_weight = None
        self.fit(
            feature_df=feature_df,
            labels=labels,
            other_probs=other_probs,
            initial_subset=initial_subset,
            proba_weight=proba_weight,
            randomize=False,
        )

    def fit(
        self,
        feature_df,
        labels,
        other_probs=None,
        initial_subset=None,
        proba_weight=None,
        randomize=True,
        **kwargs
    ):
        """
        Selects feature subsets with the highest score (as defined by parameters) for the data and sample weights.

        Parameters
        ----------
        feature_df : pd.DataFrame
        labels : pd.Series
        other_probs : list[pd.Series | pd.DataFrame]
        Prediction (regression) or class probabilities from other models
        initial_subset : (list, tuple, set)
        Features to be included in first model built.
        proba_weight : pd.Series | pd.DateaFrame
        Sample weighting based on prior predictions.
        Multiplied by self.params["base_weight"] to get final sample weights.
        If None, calculated from other_probs and self.params["combo_type"]
        randomize: bool
            Whether to randomize initial prior_prob weights or leave at naive (mean or 1/n_classes).

        Returns
        -------
        FuzzyAnnealer
        Fitted estimator.
        """
        self.feature_df = feature_df.sort_index()
        self.labels = labels.sort_index()
        self.other_probs = other_probs
        self.proba_weight = proba_weight
        self.cross_corr = None
        self.label_corr = None
        self.sq_xcorr = None
        self._current_features = initial_subset
        self.params["min_features_elim"] = min(
            self.params["features_min_vif"],
            self.params["features_min_perm"],
            self.params["features_min_sfs"],
        )
        # Type and value checking for probabilities of other subsets.
        self.initialize_other_probs(labels, other_probs, randomize=randomize)
        # Set sample weights to bias feature selection.
        if self.proba_weight is None or self.proba_weight.mask(lambda x: x == 0).all():
            self.proba_weight = samples.weight_by_proba(
                y_true=labels,
                probs=self.other_probs,
                prob_thresholds=self.params["brier_clips"],
                combo_type=self.params["pred_combine"],
            )
        self.proba_weight = pd.Series(
            data=_check_sample_weight(
                self.proba_weight, X=self.feature_df, ensure_non_negative=True
            ),
            index=feature_df.index,
        )
        if (
            self.params["base_weight"] is None
            or self.params["base_weight"].mask(lambda x: x == 0).all()
        ):
            self.params["base_weight"] = pd.Series(1, index=self.labels.index)
        self.params["base_weight"] = pd.Series(
            _check_sample_weight(
                self.params["base_weight"], X=self.feature_df, ensure_non_negative=True
            ),
            index=self.feature_df.index,
        )
        if self.proba_weight is not None and self.proba_weight.sum() > 0:
            self.sample_weight = pd.Series(
                _check_sample_weight(
                    self.params["base_weight"] * self.proba_weight,
                    X=self.feature_df,
                    ensure_non_negative=True,
                ),
                index=feature_df.index,
            )
        print(self.params["base_weight"])
        print(self.proba_weight)
        print(self.sample_weight)
        # Calculate Correlations.
        if (
            (self.sq_xcorr is None or self.label_corr is None)
            and self.sample_weight is not None
            and self.sample_weight.var() > 0.0001
            and self.sample_weight[self.sample_weight == 0.0].size
            != self.sample_weight.size
        ):
            self.label_corr, x_corr = correlation_filter.get_weighted_correlations(
                self.feature_df,
                self.labels,
                self.params,
                self.save_dir,
                weights=self.sample_weight,
            )
        else:
            self.label_corr, x_corr = correlation_filter.get_correlations(
                feature_df,
                labels,
                corr_method=self.params["corr_method"],
                xc_method=self.params["xc_method"],
            )
        self.sq_xcorr = x_corr**2
        # <editor-fold desc="Factor Out">
        if False:
            with open("{}selection_params.txt".format(self.save_dir), "w") as f:
                for k, v in self.params.items():
                    try:
                        f.write("{}: {}\n".format(k, v))
                    except AttributeError:
                        pass
        # </editor-fold>

        if self._current_features is None:
            self._current_features = sorted(
                self.label_corr.index.to_series()
                .sample(weights=self.label_corr.abs(), n=1)
                .tolist()
            )
        if initial_subset is None:
            while len(self._current_features) < self.params["min_features_out"]:
                self.choose_next_feature(score_new_set=False)
        self.chosen_subsets.append(tuple(self.current_features))
        self.best_subset = tuple(self.current_features)
        print("\n\nInitial subset: {}".format(self.current_features))
        score, score_list = self.score_subset(
            subset=self.current_features, sample_weight=self.sample_weight
        )
        sl = [score]
        for i in np.arange(self.params["max_trials"]):
            print("Running round {} of selection.".format(i + 1))
            self._train()
            assert not pd.isnull(self.current_score)
            sl.append(self.current_score)
            latest_scores = sl[max(-self.params["n_iter_no_change"], -len(sl)) :]
            pprint.pprint(latest_scores, compact=True)
            if (
                len(latest_scores) > self.params["n_iter_no_change"]
                and (max(latest_scores) - min(latest_scores)) * 2 / (min(sl) + max(sl))
                < self.params["tol"]
            ):
                print(
                    "Early stopping due to insufficient improvement. Last scores: \n{}".format(
                        sl[: self.params["n_iter_no_change"]]
                    )
                )
                break
        # Use best subset for sequentual elimination.
        self.current_features = self.best_subset
        i = len(self.current_features)
        while (
            self.current_score >= self.best_score * self.params["thresh_reset"]
            and i > 0
        ):
            subset_score, brier_list = self.sequential_elimination(randomize=False)
            if subset_score is None:
                break
            i -= 1
        return self

    def initialize_other_probs(self, labels, other_probs, randomize=False):
        if (
            other_probs is not None
            and isinstance(other_probs, (list, tuple, set))
            and len(other_probs) > 0
        ):
            self.other_probs = other_probs

        elif other_probs is None or (
            isinstance(other_probs, (list, tuple, set)) and len(other_probs) == 0
        ):
            if self.other_probs is None:
                self.other_probs = list()
            if self.params["randomize_init_weights"] and randomize:
                if is_classifier(self.models["predict"]):
                    onehot = samples.one_hot_conversion(y_true=self.labels)[0].loc[
                        self.labels.index
                    ]
                    n_obs = self.labels.shape[0]
                    poissons = pd.Series(
                        np.random.default_rng().poisson(
                            lam=np.int32(np.log2(n_obs)),
                            size=n_obs,
                        ),
                        index=self.labels.index,
                    ).div(np.int32(np.log2(n_obs)))

                    p_df = poissons - poissons.min() + 0.1
                    p_df = p_df / (p_df.max() * 1.1)
                    values = pd.concat([p_df, 1 - p_df], axis=1)
                    # onehot.mul(-1).add(1).sub(
                    print("\nRandomized probs: \n{}".format(values))
                    self.other_probs.append(values)
                else:
                    gausses = pd.Series(
                        np.random.default_rng().normal(
                            loc=self.labels.mean(),
                            scale=self.labels.std(),
                            size=self.labels.shape,
                        ),
                        index=self.labels.index,
                    )
                    self.other_probs.append(gausses)
            else:
                if is_classifier(self.models["predict"]):
                    self.other_probs.append(
                        pd.DataFrame(
                            1 / labels.nunique(),
                            index=self.labels.index,
                            columns=np.arange(self.labels.nunique()),
                        )
                    )
                elif is_regressor(self.models["predict"]):
                    self.other_probs.append(
                        pd.Series(labels.mean(), index=labels.index)
                    )
        else:
            print(other_probs)
            print(type(other_probs))
            if not isinstance(
                other_probs, (pd.Series, pd.DataFrame)
            ) and not isinstance(other_probs, (list, tuple, set)):
                raise ValueError
            else:
                raise ValueError

    def _get_add_remove(self):
        p_add, p_remove = math_tools.add_remove_probs(
            len(self.current_features),
            self.params["max_features_out"],
            self.params["min_features_out"],
        )
        return p_add, p_remove

    def _train(self):
        # Get probs.
        # Choose new feature. Evaluate (Accept/Reject).
        # Eliminate feature. Evaluate (Accept/Reject).
        p_add, p_remove = self._get_add_remove()
        if random.random() >= p_add:
            print("Choosing next features")
            self.choose_next_feature(score_new_set=True)
        if pd.isnull(self.current_score):
            print(self.current_features)
            raise ValueError
        i = 0
        print("Eliminating features.")
        while random.random() < p_remove and i < 10:
            p_add, p_remove = self._get_add_remove()
            self.random_elimination(chance=p_remove)
            i += 1
        if pd.isnull(self.current_score):
            print(self.current_features)
            raise ValueError

    def random_elimination(self, chance, sampling="importance"):
        if len(self.current_features) <= self.params["min_features_elim"]:
            return
        if sampling == "importance":
            fit_importance = fit_weighted(
                estimator=self.models[sampling],
                X=self.feature_df[self.current_features],
                y=self.labels,
                weights=self.sample_weight,
            )
            feat_weights = 1 / pd.Series(
                get_importances(estimator=fit_importance, getter="auto"),
                index=self.current_features,
            ).add(1e-6)
        elif sampling == "vif":
            vifs = (
                pd.Series(
                    vif.calculate_vif(
                        self.feature_df,
                        self.models["vif"],
                        fit_kwargs=self.sample_weight,
                    )
                )
                + 1e-6
            )
            feat_weights = 1 / vifs
        else:
            feat_weights = pd.Series(1, index=self.current_features)
        chosen_feat = (
            feat_weights.index.to_series().sample(n=1, weights=feat_weights).iloc[0]
        )
        new_subset = copy.deepcopy(self.current_features)
        new_subset.remove(chosen_feat)
        score, score_list = self.score_subset(
            subset=new_subset,
            record_results=True,
            sample_weight=self.sample_weight,
        )
        if score is None or pd.isnull(score):
            print(score)
            raise ValueError
        if self.current_score is None:
            print("Current score is undefined!!!")
            print(self.current_features)
            self.score_subset(sample_weight=self.sample_weight)
            print(self.current_score)
            score_diff = 0
        else:
            score_diff = 1 - (score / self.current_score)
        accept_prob = np.exp(-score_diff / self.temp)
        # print(score, score_diff, accept_prob, chance)
        if accept_prob > chance:
            self.current_features = sorted(new_subset)
            self.chosen_subsets.append(self.current_features)
        return

    def random_feature(self, n_feats="auto"):
        # OPTIMIZE: Use weighting from appearance in previous subsets (weighted by number of other additional features) to decrease repeat subsets.
        feat_corrs = self.label_corr.drop(index=self.current_features)
        sum_sqcc = (
            (1 - self.sq_xcorr[self.current_features].loc[feat_corrs.index])
            .sum(axis=1)
            .squeeze()
        )
        x = sum_sqcc.multiply(other=np.abs(feat_corrs), fill_value=0.0)
        feat_probs = pd.Series(math_tools.softmax(x), index=x.index).sort_values(
            ascending=False
        )
        # feat_probs.drop(index=tuple(previous_features), inplace=True, errors="ignore")
        # feature_list = list(set(feature_list))
        # noinspection PyTypeChecker
        if n_feats == "auto":
            n_feats = min(
                max(self.params["add_n_feats"], self.params["n_vif_choices"]),
                self.feature_df.shape[1],
            )
        new_feats = feat_probs.index.to_series().sample(n=n_feats, weights=feat_probs)
        if new_feats is None:
            print("No new feature selected.")
            return None
        elif isinstance(new_feats, (str, int)):
            new_feats = [new_feats]
        new_feats = list(set(new_feats))
        assert any([n not in self.current_features for n in new_feats])
        return new_feats

    def choose_next_feature(self, score_new_set=True, n_feats="auto", use_thresh=True):
        if n_feats == "auto":
            n_feats = self.params["add_n_feats"]
        vif_list = list()
        for new_i in np.arange(n_feats):
            vifs = pd.Series()
            new_feats = self.random_feature(n_feats=n_feats)
            for nf in new_feats:
                predictors = copy.deepcopy(self.current_features)
                predictors.append(nf)
                vifs_ser = calculate_vif(
                    feature_df=self.feature_df[predictors],
                    model=clone(self.models["vif"]),
                    subset=self.feature_df[[nf]],
                    sample_wts=self.sample_weight,
                )
                vifs[nf] = vifs_ser.mean() + 1e-6
            if use_thresh:
                vifs = vifs[(vifs < self.params["thresh_vif"])]
            top_feats = vifs.sort_values().iloc[: min(vifs.shape[0], n_feats)]
            if top_feats.shape[0] > 1:
                vif_selected = top_feats.sample(
                    n=min(top_feats.size, n_feats),
                    weights=1 - math_tools.softmax(top_feats),
                ).index.tolist()
            elif top_feats.shape[0] == 1:
                vif_selected = [top_feats.index[0]]
            else:
                print("No VIFs below threshold and above 0.")
                print("Feature set size: {}".format(len(self.current_features)))
                vif_selected = [random.choice(new_feats)]
            # if isinstance(vif_selected, str):
            #    vif_list = [vif_selected]
            if vif_selected is None:
                continue
            if isinstance(vif_selected, str):
                vif_list.append(vif_selected)
            elif isinstance(vif_selected, (tuple, list, set)):
                vif_list.extend(vif_selected)
        in_current = [a for a in vif_list if a in self.current_features]
        if len(in_current) > 0:
            print("VIF list in current features.")
            print(in_current)
            raise ValueError
        old_current = copy.deepcopy(self.current_features)
        if isinstance(vif_list, (list, tuple, set)):
            if len(self.current_features) == 0:
                self.current_features = sorted(vif_list)
            elif len(vif_list) > 1:
                old_current.extend(vif_list)
            else:
                old_current.append(vif_list[0])
            self.current_features = old_current
        """
        if score_new_set:
            self.chosen_subsets.append(tuple(self.current_features))
            self.score_subset(
                subset=self.current_features,
                sample_weight=self.sample_weight,
            )
        """
        return

    def score_subset(
        self,
        subset=None,
        record_results=True,
        sample_weight=None,
        class_weight="balanced",
    ):
        """
        Cross-validated scoring using a subset of features.

        Parameters
        ----------
        subset
        record_results
        sample_weight
        class_weight

        Returns
        -------
        score : float
        brier_list : list
        """

        if isinstance(subset, str):
            # raise KeyError
            subset_feats = tuple(copy.deepcopy(subset))
        else:
            subset_feats = copy.deepcopy(tuple(sorted(subset)))
        assert subset_feats is not None
        # subset_feats = tuple(sorted(subset_feats))
        score_tuple = [(self.params["score_name"], self.params["scoring"])]
        score = None
        for prior_set in self.subset_scores.keys():
            if len(set(subset_feats).symmetric_difference(set(prior_set))) == 0:
                if prior_set is not None:
                    score = self.subset_scores[prior_set]
                # print("Duplicate scoring found:\n{}\n".format(subset_feats, prior_set))
                break
        if score is None:
            if isinstance(subset_feats, str):
                subset_feats = [subset_feats]
            if (
                sample_weight is not None
                and subset_feats is not None
                and len(subset_feats) > 0
            ):
                results, exploded_dict, (test_idx_list) = scoring.cv_model_generalized(
                    estimator=self.models["predict"],
                    feature_df=self.feature_df[list(subset_feats)],
                    labels=self.labels,
                    cv=self.params["cv"],
                    methods=self.params["model_output"],
                    sample_weight=sample_weight,
                    randomize_classes=False,
                )
            elif subset_feats is not None and len(subset_feats) > 0:
                results, exploded_dict, (test_idx_list) = scoring.cv_model_generalized(
                    estimator=self.models["predict"],
                    feature_df=self.feature_df[list(subset_feats)],
                    labels=self.labels,
                    cv=self.params["cv"],
                    methods=["predict_proba"],
                    randomize_classes=False,
                )
            else:
                print("\nSubset is of length zero!")
                print(subset_feats)
                return None, None
            results = pd.concat(
                results["original"][self.params["model_output"]]["test"]
            )
            assert not results.empty
            """
            if len(score_tuple) > 0:
                score = scoring.score_cv_results(
                    results, dict(score_tuple), y_true=self.labels, **self.params
                )["Score"].tolist()
            """
            if is_classifier(self.models["predict"]):
                # TODO: Add functionality for multiple predict_proba results from previous submodels.
                brier_scores = list()
                if isinstance(self.other_probs, (pd.Series, pd.DataFrame, np.ndarray)):
                    other_probs = [self.other_probs]
                elif isinstance(self.other_probs, (list, tuple)) and all(
                    [
                        isinstance(p, (pd.Series, pd.DataFrame, np.ndarray))
                        for p in self.other_probs
                    ]
                ):
                    other_probs = self.other_probs
                else:
                    other_probs = [None]
                # print("Other probs")
                # print(other_probs)
                for prior in other_probs:
                    rel_brier_score, brier_obs = scoring.relative_brier_score(
                        y_true=self.labels,
                        y_proba=results,
                        y_prior=prior,
                        clips=self.params["brier_clips"],
                        sample_weight=self.sample_weight,
                        class_weight=class_weight,
                        best_k=self.params["best_k"],
                    )
                    print(
                        "Scoring used {} observations for {}.".format(
                            brier_obs.shape[0], rel_brier_score
                        )
                    )
                    brier_scores.append(rel_brier_score)
                avg_score = np.min(brier_scores)
                score = avg_score
            else:
                print(self.models["predict"].__repr__)
                raise NotImplementedError
            self.subset_scores[tuple(sorted(subset_feats))] = score
            self._compare_to_best(
                score=avg_score, sample_preds=results, features=subset_feats
            )
            if pd.isnull(avg_score) or avg_score is None:
                print(avg_score, brier_scores)
                raise ValueError
            if record_results:
                self.record_score(features=subset_feats, score=avg_score)
            # print("Subset scored: {}".format(avg_score))
        else:
            results = None
            if isinstance(score, (list, tuple, pd.Series)):
                avg_score = np.mean(score)
            else:
                avg_score = score
        assert avg_score is not None and not pd.isnull(avg_score)
        return avg_score, results

    def record_score(self, features, score, test_score=None):
        if isinstance(score, list):
            score = np.mean(score)
        score_str = "{}\t{}\n".format(
            "{:.5f}".format(float(score)),
            "\t".join(features),
        )
        with open(
            "{}feature_score_path.csv".format(self.save_dir), "a", encoding="utf-8"
        ) as f:
            f.write(score_str)
        if test_score is not None:
            with open(
                "{}test_scores.csv".format(self.save_dir), "a", encoding="utf-8"
            ) as f:
                f.write(
                    "{:.5f}\t{}\n".format(
                        test_score,
                        "\t".join(features),
                    )
                )

        return

    def _compare_to_best(self, score, sample_preds, features=None):
        if self.best_subset is None:
            self.best_subset = self.current_features

        if tuple(sorted(self.best_subset)) not in list(self.subset_scores.keys()):
            print("Previous best score was not found in list of scored subsets.")
            self.score_subset(subset=self.best_subset, sample_weight=self.sample_weight)
        if score >= self.best_score:
            print(
                "New top results for {} feature model: {:.4f}".format(
                    len(features), score
                )
            )
            self.best_subset = tuple(sorted(copy.deepcopy(features)))
            self.subset_scores[self.best_subset] = score
            self.best_score = score
            self.best_preds = sample_preds
            best_yet = True
            self.temp = 1
        else:
            best_yet = False
        return best_yet

    def score_drop_exceeded(
        self,
        new_score,
        set_size,
        replace_current=True,
    ):
        if (
            self._over_sfs_thresh(
                scores=new_score,
                set_size=set_size,
                factor="reset",
            )
            and set_size >= self.params["features_min_sfs"]
        ):
            print(
                "Score (adjusted) drop exceeded: {:.4f} {:.4f}".format(
                    self.subset_scores[self.best_subset], new_score
                )
            )
            if replace_current:
                self.current_features = copy.deepcopy(list(self.best_subset))
            return True
        else:
            return False

    def _get_subset_scores(self, subset):
        feats = tuple(sorted(subset))
        assert len(feats) > 0
        if feats not in self.subset_scores.keys():
            self.score_subset(subset=feats, sample_weight=self.sample_weight)
        return self.subset_scores[feats]

    def best_score(self):
        return self._get_subset_scores(tuple(sorted(self.best_subset)))

    def _over_sfs_thresh(self, scores="current", set_size=None, factor="sfs"):
        # Returns boolean of whether current score is within some amount of the best score.
        # Fewer features = Lower overmax = Greater adjust = Easier to pass
        if "sfs" in factor:
            factor = float(self.params["thresh_sfs"])
        elif "reset" in factor:
            factor = float(self.params["thresh_reset"])
        elif "cleanup" in factor:
            factor = float(self.params["thresh_sfs_cleanup"])
        else:
            factor = 0
        if factor is None:
            factor = 0
        reference, _ = self.score_subset(
            subset=self.best_subset, sample_weight=self.sample_weight
        )
        if isinstance(scores, str):
            if "current" in scores:
                scores = self.current_score
                if set_size is None:
                    set_size = len(self.current_features)
            else:
                raise ValueError
        elif scores is None:
            return False
        elif set_size is None:
            raise UserWarning
            print("Feature set size needed if not using current feature set.")
            set_size = self.params["max_features_out"]
        adjust = math_tools.complexity_penalty(
            math_tools.size_factor(set_size, self.params), factor
        )
        if any(
            [
                (a is None or isinstance(a, dict) or not isinstance(a, numbers.Real))
                for a in [scores, reference, factor]
            ]
        ):
            print(scores, reference, factor)
        return scores >= reference * (1 + factor)

    def sequential_elimination(
        self,
        randomize=True,
        depth=1,
    ):
        """

        Parameters
        ----------
        randomize : bool, whether to use probabalistic method for elimination selection and accept-reject decision
        depth : int, (Not Implemented) number of features to eliminate

        Returns
        -------
        subset_score: float, score for feature set after elimination
        brier_list: str, list of scores (relative to all other submodels)
        """
        if self.clean_up:
            clean = "thresh_sfs_cleanup"
        else:
            clean = "thresh_sfs"
        # TODO: Implement configurable predict function for boosting.
        sfs_score_dict = dict()
        for left_out in self.current_features:
            new_subset = copy.deepcopy(self.current_features)
            new_subset.remove(left_out)
            out_score, brier_list = self.score_subset(
                subset=sorted(new_subset),
                record_results=True,
                sample_weight=self.sample_weight,
            )
            sfs_score_dict[left_out] = out_score
            if out_score is not None:
                sfs_score_dict[left_out] = out_score
        if len(sfs_score_dict.items()) == 0:
            print("No valid SFS scores returned.")
            raise RuntimeWarning
            out_score = self.subset_scores[tuple(self.current_features)]
        feats = list(sfs_score_dict.keys())
        if randomize:
            zwangzig = math_tools.zwangzig(
                scores_list=[self.subset_scores[tuple(s)] for s in self.chosen_subsets],
                test_score=out_score,
                lamb=self.params["lang_lambda"],
                temp=self.temp,
                k=math_tools.size_factor(len(new_subset), self.params),
            )
            if random.random() < zwangzig:
                subset_scores = self.subset_scores[tuple(sorted(self.current_features))]
                chosen_feat = None
            else:
                new_subset = ""
                while tuple(new_subset) not in self.subset_scores.keys():
                    chosen_feat = self.choose_by_softmax(sfs_score_dict)[0]
                    new_subset = copy.deepcopy(self.current_features)
                    new_subset.remove(chosen_feat)
                if new_subset is not None and new_subset != "":
                    subset_scores = sfs_score_dict[chosen_feat]
                    self.record_score(features=new_subset, score=subset_scores)
                    self.current_features = sorted(new_subset)
                else:
                    subset_scores = self.subset_scores[tuple(self.current_features)]
        else:
            chosen_feat = sorted(list(sfs_score_dict.items()), key=lambda x: (x[1]))[0][
                0
            ]
            subset_scores = sfs_score_dict[chosen_feat]
            set_size = len(self.current_features) - 1
            if self._over_sfs_thresh(
                scores=subset_scores,
                set_size=set_size,
                factor=clean,
            ):
                self.current_features = chosen_feat
            else:
                subset_scores = None
        return subset_scores, chosen_feat

    def choose_by_softmax(self, score_dict, n=1):
        geoms = dict()
        for k, v in score_dict.items():
            if np.isscalar(v):
                geoms[k] = v
            elif len(v) > 0:
                geoms[k] = geometric_mean([a for a in v if a > 0])
            else:
                print("Features {} have a value of {}".format(k, v))
        soft_scores = math_tools.scaled_softmax(geoms.values(), center=self.temp)
        scores_ser = pd.Series(soft_scores, index=geoms.keys())
        if scores_ser.sum() == 0.0:
            raise ValueError
        chosen_feat = (
            pd.Series(data=geoms.keys(), index=geoms.keys())
            .sample(n=n, weights=scores_ser.squeeze())
            .tolist()
        )
        return chosen_feat

    def _permutation_removal(self):
        """

        Selects feature for removal based on score drop after permutation of value.

        Parameters
        ----------

        Returns
        -------

        """
        # TODO: Consider switching to partial permutation to avoid tree-based bias towards high cardinality features.
        n_repeats = self.params["perm_n_repeats"]
        while len(self.current_features) > self.params["features_min_perm"]:
            estimator = clone(self.models["permutation"]).fit(
                self.feature_df[self.current_features], self.labels
            )
            perm_results = permutation_importance(
                estimator,
                self.feature_df[self.current_features],
                self.labels,
                n_repeats=n_repeats,
                scoring=self.params["scorer"],
                n_jobs=-1,
            )
            import_mean = pd.Series(
                data=perm_results["importances_mean"],
                index=self.current_features,
                name="Mean",
            ).sort_values()
            import_std = pd.Series(
                data=perm_results["importances_std"],
                index=self.current_features,
                name="StD",
            )
            adj_importance = import_mean.iloc[0] + import_std[import_mean.index].iloc[0]
            if len(self.current_features) < self.params["min_feats_perm"]:
                import_thresh = min(self.params["thresh_perm"] + 0.05, adj_importance)
            else:
                import_thresh = min(self.params["thresh_perm"], adj_importance)
            unimportant = import_mean[import_mean <= import_thresh].index.tolist()
            if len(unimportant) > 1 and n_repeats <= 50:
                n_repeats = n_repeats + 25
                continue
            elif len(unimportant) == 1 or (n_repeats >= 50 and len(unimportant) > 0):
                low_feats = unimportant[0]
            else:
                break
                # [self.current_features.remove(c) for c in low_feats]
            self.current_features.remove(low_feats)
            print(
                "Permutation drop: \n{}: {} {}".format(
                    low_feats,
                    import_mean[low_feats],
                    import_std[low_feats],
                )
            )

    def _vif_elimination(self, verbose=False):
        model_size = 10
        vif_rounds = int(
            len(self.current_features) * (len(self.current_features) + 1) / model_size
        )
        # fweights = 0.5 - pd.Series(scipy.linalg.norm(cross_corr.loc[self.current_features, self.current_features], axis=1), index=self.current_features).squeeze().sort_values(ascending=False)
        vif_results = repeated_stochastic_vif(
            feature_df=self.feature_df[self.current_features],
            importance_ser=self.label_corr[self.current_features].abs(),
            threshold=self.params["thresh_vif"],
            model_size=model_size,
            feat_wts=self.cross_corr[self.current_features].loc[self.current_features],
            min_feat_out=len(self.current_features) - 1,
            rounds=vif_rounds,
        )
        if len(vif_results["vif_stats_list"]) > 1:
            all_vif_mean = (
                pd.concat([df["Max"] for df in vif_results["vif_stats_list"]], axis=1)
                .max()
                .sort_values(ascending=False)
            )
            all_vif_sd = pd.concat(
                [df["StdDev"] for df in vif_results["vif_stats_list"]], axis=1
            ).std(ddof=len(self.current_features) - model_size)
        else:
            all_vif_mean = vif_results["vif_stats_list"][0]["Max"].sort_values(
                ascending=False
            )
            all_vif_sd = vif_results["vif_stats_list"][0]["SD"]
        vif_dropped = all_vif_mean.index[0]
        # [print(c, [d.loc[c] for d in vif_results["vif_stats_list"] if c in d.index]) for c in vif]
        if (
            vif_dropped in self.current_features
            and all_vif_mean[vif_dropped] > self.params["thresh_vif"]
        ):
            self.current_features.remove(vif_dropped)
            # self.subset_scores[self.best_subset].update([(vif_dropped, "VIF")])
            if False:
                print("VIF Scores")
                pprint.pp(
                    pd.concat(
                        [
                            all_vif_mean,
                            all_vif_sd[all_vif_mean.index],
                            vif_results["votes_list"][-1].loc[all_vif_mean.index],
                        ],
                        axis=1,
                    ).head(n=3),
                    width=120,
                )
        else:
            if verbose:
                print("Nothing above threshold.")
        return selection_state

    def _check_no_change(self):
        score_list = [self.subset_scores[tuple(s)] for s in self.chosen_subsets[:1]]

    """
    def delete_asap(self):

        Parameters
        ----------

        Returns
        -------

        raise DeprecationWarning.with_traceback()
        # Start feature loop
        for i in np.arange(self.params["max_trials"]):
            print(
                "\n\nSelection step {} out of {}.".format(i, self.params["max_trials"])
            )
            if i > 0:
                self.current_best_ratio()
                self.temp = self.temp * math.exp(1 - self.temp)
            print("New temperature: {}".format(self.temp))
            if not self._prelim_sfs():
                continue

            # Variance Inflation Factor: VIF check implemented in "new feature" selection function.
            if False and (
                len(self.current_features) >= self.params["features_min_vif"]
            ):
                _ = self._vif_elimination()

            # Feature Importance Elimination
            if True and len(self.current_features) > self.params["features_min_perm"]:

                if self.params["importance"] == "permutate":
                    self._permutation_removal()

                    elif (
                        self.params["importance"] != "permutate"
                        and self.params["importance"] is not False
                    ):
                    # rfe = RFECV(estimator=grove_model, min_features_to_select=len(self.current_features)-1, n_jobs=-1).set_output(transform="pandas")
                else:
                    fit_model = clone(self.models["importance"]).fit(
                        self.feature_df[self.current_features], self.labels
                    )
                    importances = pd.Series(
                        get_importances(estimator=fit_model, getter="auto"),
                        index=self.current_features,
                    )
                    importances.sort_values(inplace=True)
                    for candidate in self.current_features:
                        subset = copy.deepcopy(self.current_features)
                        subset.remove(candidate)
                        if tuple(sorted(subset)) not in self.subset_scores.keys():
                            # print("Dropped: {}".format(candidate))
                            self.current_features = subset
                            self.chosen_subsets.append(subset)
                            break
            while (
                len(self.current_features) >= self.params["features_min_sfs"]
            ) or self.clean_up:
                if len(self.current_features) >= self.params["max_features_out"]:
                    self.clean_up = True
                # DEBUG
                if len(self.current_features) <= self.params["min_features_out"]:
                    break
                n_features_in = len(copy.deepcopy(self.current_features))
                subset_score, elim_feat = self.sequential_elimination()
                # SFS fails to eliminate a feature.
                # DEBUG
                if n_features_in > len(self.current_features):
                    "Failed to eliminate a feature!"
                    self.clean_up = False
                    break
                too_much = self.score_drop_exceeded(
                    new_score=subset_score,
                    set_size=len(self.current_features),
                    replace_current=False,
                )
                if too_much or n_features_in == len(self.current_features):
                    self.clean_up = False
                    break
        print(
            "Best adjusted score of {} with feature set: {}".format(
                self.subset_scores[self.best_subset],
                self.best_subset,
            )
        )
        self.best_model = self.freeze_best()
        return self.best_model, self.subset_scores, self.best_subset
        """

    def freeze_best(self):
        best_fit_model = fit_weighted(
            estimator=self.models["predict"],
            X=self.feature_df,
            y=self.labels,
            weights=self.sample_weight,
        )
        frozen_model = FrozenEstimator(best_fit_model)
        with open("{}best_model.pkl".format(self.save_dir), "wb") as f:
            pickle.dump(frozen_model, f)
        return best_fit_model

    def freeze_cv_best(self, cv):
        model_tuples = list()
        for train_X, train_y, test_X, test_y in cv_tools.split_df(
            self.feature_df, self.labels
        ):
            trained = fit_weighted(
                clone(self.models["predict"]),
                X=train_X,
                y=train_y,
                weights=self.sample_weight[train_y.index],
            )
            y_pred = pd.Series(trained.predict(X=test_X), index=test_X.index)
            y_prob = pd.DataFrame(trained.predict_proba(X=test_X), index=test_X.index)
            model_tuples.append((trained, test_y.copy(), y_pred, y_prob))
        return model_tuples

    def current_best_ratio(self):
        r_best = (
            self.subset_scores[tuple(self.best_subset)]
            - self.subset_scores[tuple(self.current_features)]
        ) / self.subset_scores[tuple(self.best_subset)]
        return r_best

    def _prelim_sfs(self):
        maxed_out = len(self.current_features) >= self.params["max_features_out"]
        above_min = len(self.current_features) >= self.params["features_min_sfs"]
        if False and above_min and (maxed_out or self.clean_up):
            while self._over_sfs_thresh():
                original_size = len(self.current_features)
                subset_score, elim_feat = self.sequential_elimination()
                if (
                    len(
                        set(self.current_features).symmetric_difference(
                            self.best_subset
                        )
                    )
                    == 0
                ):
                    self.clean_up = False
                    break
            return False
        else:
            self.clean_up = False
        self.choose_next_feature()
        exceeded = self.score_drop_exceeded(
            new_score=self.subset_scores[tuple(self.current_features)],
            set_size=len(self.current_features),
            replace_current=False,
        )
        while exceeded and len(self.current_features) > self.params["features_min_sfs"]:
            if len(self.current_features) <= self.params["features_min_sfs"]:
                self.current_features = copy.deepcopy(list(self.best_subset))
                exceeded = False
                continue
            subset_score, elim_feat = self.sequential_elimination()
            exceeded = self.score_drop_exceeded(
                new_score=subset_score,
                set_size=len(self.current_features) - 1,
                replace_current=False,
            )
        return True
