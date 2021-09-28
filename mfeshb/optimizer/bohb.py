from openbox.utils.config_space import ConfigurationSpace
from mfeshb.optimizer.hyperband import Hyperband
from mfeshb.optimizer.base.utils import sample_configurations, expand_configurations
from mfeshb.acq_maximizer.ei_optimization import RandomSampling

import numpy as np
from openbox.utils.util_funcs import get_types
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.utils.config_space.util import convert_configurations_to_array


class BOHB(Hyperband):
    """ The implementation of BOHB.
        The paper can be found in https://arxiv.org/abs/1807.01774 .
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='BOHB',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        super().__init__(objective_func, config_space, R, eta=eta, num_iter=num_iter,
                         random_state=random_state, method_id=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        types, bounds = get_types(config_space)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_function = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                            n_samples=max(5000, 50 * len(bounds)))
        self.rng = np.random.RandomState(self.seed)

    def choose_next(self, num_config):
        # Sample n configurations according to BOHB strategy.
        self.logger.info('Sample %d configs in choose_next. rand_prob is %f.' % (num_config, self.rand_prob))

        if len(self.incumbent_configs) < self.bo_init_num:
            self.logger.info('len(self.incumbent_configs) = %d. Return all random configs.'
                             % (len(self.incumbent_configs), ))
            return sample_configurations(self.config_space, num_config, excluded_configs=self.incumbent_configs)

        config_candidates = []

        # BO
        num_bo_config = num_config - int(num_config * self.rand_prob)
        self.surrogate.train(convert_configurations_to_array(self.incumbent_configs),
                             np.array(self.incumbent_perfs, dtype=np.float64))
        best_index = np.argmin(self.incumbent_perfs)
        best_config = self.incumbent_configs[best_index]
        best_perf = self.incumbent_perfs[best_index]
        # Update surrogate model in acquisition function.
        self.acquisition_function.update(model=self.surrogate, eta=best_perf,
                                         num_data=len(self.incumbent_configs))
        bo_candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
        for config in bo_candidates:
            if config not in config_candidates + self.incumbent_configs:
                config_candidates.append(config)
                if len(config_candidates) == num_bo_config:
                    break
        self.logger.info('len bo configs = %d.' % len(config_candidates))

        # sample random configs
        config_candidates = expand_configurations(config_candidates, self.config_space, num_config,
                                                  excluded_configs=self.incumbent_configs)
        self.logger.info('len total configs = %d.' % len(config_candidates))
        assert len(config_candidates) == num_config
        return config_candidates

    def update_incumbent_before_reduce(self, T, val_losses, n_iteration):
        if int(n_iteration) < self.R:
            return
        self.incumbent_configs.extend(T)
        self.incumbent_perfs.extend(val_losses)
        self.logger.info('%d observations updated. %d incumbent configs total.' % (len(T), len(self.incumbent_configs)))

    def update_incumbent_after_reduce(self, T, incumbent_loss):
        return
