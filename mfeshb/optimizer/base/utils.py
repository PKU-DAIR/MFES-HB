from typing import List
import numpy as np
from openbox.utils.config_space import Configuration, ConfigurationSpace


def sample_configuration(configuration_space: ConfigurationSpace, excluded_configs=None,
                         max_sample_cnt=1000):
    """
    sample one config not in excluded_configs
    """
    if excluded_configs is None:
        excluded_configs = set()
    if isinstance(excluded_configs, set):
        excluded_configs_set = excluded_configs
    else:
        excluded_configs_set = set(excluded_configs)

    sample_cnt = 0
    while True:
        config = configuration_space.sample_configuration()
        sample_cnt += 1
        if config not in excluded_configs_set:
            break
        if sample_cnt >= max_sample_cnt:
            raise ValueError('Cannot sample non duplicate configuration after %d iterations. '
                             'len of excluded configs set/list = %d/%d.'
                             % (max_sample_cnt, len(excluded_configs_set), len(excluded_configs)))
    return config


def sample_configurations(configuration_space: ConfigurationSpace, num,
                          excluded_configs=None, max_sample_cnt=1000) -> List[Configuration]:
    if excluded_configs is None:
        excluded_configs = set()
    if isinstance(excluded_configs, set):
        excluded_configs_set = excluded_configs
    else:
        excluded_configs_set = set(excluded_configs)

    result = []
    result_set = set()  # speedup checking
    max_sample_cnt = max(max_sample_cnt, 3 * num)
    sample_cnt = 0
    while len(result) < num:
        config = configuration_space.sample_configuration(1)
        sample_cnt += 1
        if config not in result_set and config not in excluded_configs_set:
            result.append(config)
            result_set.add(config)
        if sample_cnt >= max_sample_cnt:
            raise ValueError('Cannot sample non duplicate configuration after %d iterations. '
                             'len of excluded configs set/list = %d/%d.'
                             % (max_sample_cnt, len(excluded_configs_set), len(excluded_configs)))
    return result


def expand_configurations(configs: List[Configuration], configuration_space: ConfigurationSpace, num: int,
                          excluded_configs=None, max_sample_cnt=1000):
    if excluded_configs is None:
        excluded_configs = set()
    if isinstance(excluded_configs, set):
        excluded_configs_set = excluded_configs
    else:
        excluded_configs_set = set(excluded_configs)

    max_sample_cnt = max(max_sample_cnt, 3 * num)
    sample_cnt = 0
    while len(configs) < num:
        config = configuration_space.sample_configuration(1)
        sample_cnt += 1
        if config not in configs and config not in excluded_configs_set:
            configs.append(config)
        if sample_cnt >= max_sample_cnt:
            raise ValueError('Cannot sample non duplicate configuration after %d iterations. '
                             'len of excluded configs set/list = %d/%d.'
                             % (max_sample_cnt, len(excluded_configs_set), len(excluded_configs)))
    return configs


def minmax_normalization(x):
    min_value = min(x)
    delta = max(x) - min(x)
    if delta == 0:
        return [1.0] * len(x)
    return [(float(item) - min_value) / float(delta) for item in x]


def std_normalization(x):
    _mean = np.mean(x)
    _std = np.std(x)
    if _std == 0:
        return np.array([0.] * len(x))
    return (np.array(x) - _mean) / _std


def norm2_normalization(x):
    z = np.array(x)
    normalized_z = z / np.linalg.norm(z)
    return normalized_z
