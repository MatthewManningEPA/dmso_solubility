import os
import pickle

import pandas as pd


class FuzzyFileHandler:

    def __init__(
        self,
        working_dir,
        estimator_name,
        estimator_type,
        primary_output_type,
        index_label="INCHI_KEY",
    ):
        """

        Parameters
        ----------
        working_dir : str
        estimator_name : str
        estimator_type : str | bool
        primary_output_type : str
        index_label : str
        """
        self.working_dir = working_dir
        self.estimator_name = estimator_name
        if (
            estimator_type == "clf"
            or estimator_type == "classifier"
            or estimator_type is True
        ):
            self.is_classifier = True
        else:
            self.is_classifier = False
        if index_label is not None:
            self.index_label = index_label
        else:
            self.index_label = None
        self.output_type = primary_output_type

    def get_paths(self, subset_dir=None, i=None, label_name="original"):
        # method_name = "predict_proba"
        i = str(i)
        subset_paths = dict()
        if subset_dir is None:
            assert i is not None
            subset_dir = self.get_subset_dir(str(i))
        print(subset_dir)
        # TODO: Compare i to len(blah_blah_dict[self.estimatolabel_name]) to see if a fit is needed.
        os.makedirs(subset_dir, exist_ok=True)

        subset_paths["best_features"] = "{}{}_{}_best_features.csv".format(
            subset_dir, self.estimator_name, i
        )
        subset_paths["best_model"] = "{}{}_{}_best_model.pkl".format(
            subset_dir, self.estimator_name, i
        )
        subset_paths["weighted_label"] = "{}weighted_label_corr.csv".format(subset_dir)
        subset_paths["weighted_cross"] = "{}weighted_cross_corr.csv".format(subset_dir)
        return subset_paths

    def _get_pkl_path(self, i):
        return self.get_predict_paths(
            i, method_name=None, file_type="pkl", label_name=None, split_type=None
        )

    def get_subset_dir(self, i):
        subset_dir = "{}subset_{}/".format(self.working_dir, i)
        return subset_dir

    def get_predict_paths(
        self, i, method_name, file_type, label_name="original", split_type="test"
    ):
        subset_dir = self.get_subset_dir(i)
        os.makedirs(subset_dir, exist_ok=True)
        fmt_path = "{}{}".format(subset_dir, self.estimator_name)
        for id in (i, method_name, label_name, split_type):
            if id is not None and id != "":
                fmt_path += "_{}".format(str(id))
        fmt_path += ".{}".format(file_type)
        # fmt_path = "{}_{}_{}_{}.{}}".format(predict_stub, i, method_name, label_name, file_type)
        return fmt_path

    def load_cv_results(
        self,
        i,
    ):
        """
        Loads nested dictionary of results from i-th subset.

        Parameters
        ----------
        i
        Returns
        -------
        cv_predicts_loaded : dict[str, dict[str, dict[str, pd.Series | pd.DataFrame]]]
        Results from previous submodel search. CV results are concatenated.
        """
        with open(self._get_pkl_path(i), "rb") as f:
            cv_predicts_loaded = pickle.load(f)
        return cv_predicts_loaded

    def save_model_results(
        self,
        save_dir=None,
        i=None,
        best_features=None,
        frozen_model=None,
        label_corr=None,
        pair_corr=None,
    ):
        if save_dir is None or isinstance(save_dir, int):
            subset_paths = self.get_paths(i=i)
        else:
            subset_paths = self.get_paths(subset_dir=save_dir, i=i)
        if isinstance(best_features, (list, tuple, set)):
            pd.Series(best_features).to_csv(subset_paths["best_features"])
        elif isinstance(best_features, pd.Series):
            best_features.to_csv(subset_paths["best_features"])
        if frozen_model is not None:
            with open(subset_paths["best_model"], "wb") as f:
                pickle.dump(frozen_model, f)
        if label_corr is not None:
            label_corr.to_csv(subset_paths["weighted_label"])
        if pair_corr is not None:
            pair_corr.to_csv(subset_paths["weighted_cross"])

    def save_cv_results(
        self,
        i,
        cv_predicts,
        output_type=("predict", "predict_proba"),
        split_type=("test",),
        label_names=("original",),
    ):
        df_list = list()
        filtered_dict = dict()
        for r_name, r_dict in cv_predicts.items():
            if r_name not in label_names:
                continue
            filtered_dict[r_name] = dict()
            for output_name, score_dict in r_dict.items():
                if output_name not in output_type:
                    continue
                filtered_dict[r_name][output_name] = dict()
                for cv_type, results_df in score_dict.items():
                    if cv_type not in split_type:
                        continue
                    csv_path = self.get_predict_paths(
                        i,
                        method_name=output_name,
                        file_type="csv",
                        label_name=r_name,
                        split_type=cv_type,
                    )
                    if isinstance(results_df, (pd.Series, pd.DataFrame)):
                        results = results_df
                    elif isinstance(results_df, (list, tuple)):
                        results = pd.concat(results_df)
                    else:
                        raise TypeError
                    if results.empty:
                        raise ValueError
                    results.to_csv(csv_path, index_label=self.index_label)
                    filtered_dict[r_name][output_name][cv_type] = results
        pkl_path = self._get_pkl_path(i)
        with open(pkl_path, "wb") as f:
            pickle.dump(filtered_dict, f)
        return filtered_dict
