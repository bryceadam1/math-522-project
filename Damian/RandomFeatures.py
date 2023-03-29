import numpy as np
class random_features:
    def __init__(self):
        pass

    def generate_features(self, feature_funcs, coeff_stds, error, num_features):
        self.num_features = num_features
        self.feature_funcs = feature_funcs
        self.coeff = np.random.normal(0, coeff_stds, (num_features, len(feature_funcs)))
        self.error = error

    def get_xs(self, ys):
        xs = []
        for y in ys:
            func_vals = [f(y) for f in self.feature_funcs]
            xs.append(self.coeff @ func_vals)
        return np.array(xs)

    