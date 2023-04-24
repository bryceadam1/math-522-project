import numpy as np
class random_features:
    def __init__(self, func, num_params, num_features, num_rands = 1):
        """func: should have several inputs
        @param
        func should take in several inputs.
        The first should be y,
        the second should be a random normally distributed number, this should reflect noise.
        The rest of the numbers are randomly generated.
        Each feature will have a different set of random numbers reflecting what the feature is measuring."""
        self.func = func
        self.params = np.random.normal(0, 1, (num_features, num_params))
        self.num_rands = num_rands

    def get_xs(self, ys):
        all_outputs = []
        for y in ys:
            y_outputs = []
            for feature_params in self.params:
                y_outputs.append(self.func(y, *np.random.normal(0, 1, self.num_rands), *feature_params))
            all_outputs.append(y_outputs)
        return np.array(all_outputs)

    