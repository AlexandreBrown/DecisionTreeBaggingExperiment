import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BaggedTreeRegressor:
    def __init__(self, number_of_sampling):
        self.models = []
        self.number_of_sampling = number_of_sampling

    def fit(self, X, Y):

        number_of_samples = len(X)

        for sample in range(self.number_of_sampling):
            x_ids = np.random.choice(number_of_samples, size=number_of_samples, replace=True)
            x_bootstrap = X[x_ids]
            y_bootstrap = Y[x_ids]

            model = DecisionTreeRegressor()
            model.fit(x_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):

        predictions = np.zeros(len(X))

        for model in self.models:
            predictions += model.predict(X)

        return predictions / self.number_of_sampling

    def score(self, X, Y):

        difference_between_target_and_prediction = Y - self.predict(X)
        difference_between_target_and_mean_of_target = Y - Y.mean()

        r2_score = 1 - difference_between_target_and_prediction.dot(
            difference_between_target_and_prediction) / difference_between_target_and_mean_of_target.dot(
            difference_between_target_and_mean_of_target)

        return r2_score
