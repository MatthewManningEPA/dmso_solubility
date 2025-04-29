class FuzzyApplicator:

    def __init__(self, base_regressor):
        self.annealer_list = None
        self.predictor_type = None
        self.predictors = None
        self.label_type = None
        self.labels = None
        self.base_regressor = base_regressor
        self.annealer_weights = dict()

    def fit(
        self,
        predictors,
        labels,
        predictor_type=None,
        label_type=None,
        annealer_wts=None,
    ):
        self.predictors = predictors
        self.labels = labels
        self.predictor_type = predictor_type
        self.label_type = label_type
        self.annealer_weights = annealer_wts
        pass

    def add_annealer(self):
        pass

    def build_regressor(self):
        # Instantiate regressor function to predict self.labels using self.predictors (and self.weights? but not annealer weights?)
        pass
