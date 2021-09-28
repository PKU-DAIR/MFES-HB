import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter


class CountingOnes(object):
    """
    Proposed by BOHB
    """

    def __init__(self, n_cat, n_cont, max_samples=729, seed=47, **kwargs):
        self.dim = n_cat+n_cont
        self.n_cat = n_cat
        self.n_cont = n_cont
        self.max_samples = max_samples
        self._optimal_value = -(n_cat + n_cont)
        self.optimal_value = 0
        self.rng = np.random.RandomState(seed)

    def evaluate_config(self, config, fidelity):
        x_cat = [config['cat%d' % i] for i in range(self.n_cat)]
        x_cont = [config['cont%d' % i] for i in range(self.n_cont)]

        result = -np.sum(x_cat)
        # draw samples to approximate the expectation (Bernoulli distribution)
        n_samples = int(self.max_samples * fidelity)
        for x in x_cont:
            result -= self.rng.binomial(n_samples, p=x) / n_samples
        return abs(result - self._optimal_value)

    def get_configspace(self):
        cs = ConfigurationSpace()
        for i in range(self.n_cat):
            x_cat = CategoricalHyperparameter("cat%d" % i, choices=[0, 1])
            cs.add_hyperparameter(x_cat)
        for i in range(self.n_cont):
            x_cont = UniformFloatHyperparameter("cont%d" % i, 0, 1)
            cs.add_hyperparameter(x_cont)
        return cs
