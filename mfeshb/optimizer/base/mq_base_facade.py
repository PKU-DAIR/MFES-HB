import time
import os
import numpy as np
import pickle as pkl
from openbox.utils.logging_utils import get_logger, setup_logger
from openbox.core.message_queue.master_messager import MasterMessager

PLOT = False
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    PLOT = True
except Exception as e:
    pass


class mqBaseFacade(object):
    def __init__(self, objective_func,
                 restart_needed=True,
                 need_lc=False,
                 method_name='default_method_name',
                 log_directory='logs',
                 data_directory='data',
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 max_queue_len=300,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 sleep_time=0.1,):
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        self.logger = self._get_logger(method_name)

        self.objective_func = objective_func
        self.trial_statistics = []
        self.recorder = []

        self.global_start_time = time.time()
        self.runtime_limit = None
        self.global_incumbent = 1e10
        self.global_incumbent_configuration = None
        self.global_trial_counter = 0
        self.restart_needed = restart_needed
        self.record_lc = need_lc
        self.method_name = method_name

        self.save_intermediate_record = False
        self.save_intermediate_record_id = 0
        self.save_intermediate_record_path = None

        if self.method_name is None:
            raise ValueError('Method name must be specified! NOT NONE.')

        self.time_limit_per_trial = time_limit_per_trial
        self.runtime_limit = runtime_limit

        max_queue_len = max(1000, max_queue_len)
        self.master_messager = MasterMessager(ip, port, authkey, max_queue_len, max_queue_len)
        self.sleep_time = sleep_time

    def run_in_parallel(self, configurations, n_iteration, extra_info=None, initial_run=True):
        n_configuration = len(configurations)
        performance_result = []
        early_stops = []

        # TODO: need systematic tests.
        # check configurations, whether it exists the same configs
        count_dict = dict()
        for i, config in enumerate(configurations):
            if config not in count_dict:
                count_dict[config] = 0
            count_dict[config] += 1

        # incorporate ref info.
        conf_list = []
        for index, config in enumerate(configurations):
            extra_conf_dict = dict()
            if count_dict[config] > 1:
                extra_conf_dict['uid'] = count_dict[config]
                count_dict[config] -= 1

            if extra_info is not None:
                extra_conf_dict['reference'] = extra_info[index]
            extra_conf_dict['need_lc'] = self.record_lc
            extra_conf_dict['method_name'] = self.method_name
            extra_conf_dict['initial_run'] = initial_run    # for loading from checkpoint in DL
            conf_list.append((config, extra_conf_dict))

        # Add batch configs to masterQueue.
        for config, extra_conf in conf_list:
            msg = [config, extra_conf, self.time_limit_per_trial, n_iteration, self.global_trial_counter]
            self.master_messager.send_message(msg)
            self.global_trial_counter += 1
        self.logger.info('Master: %d configs sent.' % (len(conf_list)))
        # Get batch results from workerQueue.
        result_num = 0
        result_needed = len(conf_list)
        while True:
            if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
                break
            observation = self.master_messager.receive_message()    # return_info, time_taken, trial_id, config
            if observation is None:
                # Wait for workers.
                time.sleep(self.sleep_time)
                continue
            # Report result.
            result_num += 1
            global_time = time.time() - self.global_start_time
            self.trial_statistics.append((observation, global_time))
            self.logger.info('Master: Get the [%d] observation %s. Global time=%.2fs.'
                             % (result_num, str(observation), global_time))
            if result_num == result_needed:
                break

        # sort by trial_id. FIX BUG
        self.trial_statistics.sort(key=lambda x: x[0][2])

        # get the evaluation statistics
        for observation, global_time in self.trial_statistics:
            return_info, time_taken, trial_id, config = observation

            performance = return_info['loss']
            if performance < self.global_incumbent:
                self.global_incumbent = performance
                self.global_incumbent_configuration = config

            performance_result.append(return_info)
            early_stops.append(return_info.get('early_stop', False))
            self.recorder.append({'trial_id': trial_id, 'time_consumed': time_taken,
                                  'configuration': config, 'n_iteration': n_iteration,
                                  'return_info': return_info, 'global_time': global_time})

        self.trial_statistics.clear()

        self.save_intermediate_statistics()
        if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
            raise ValueError('Runtime budget meets!')
        return performance_result, early_stops

    def set_save_intermediate_record(self, dir_path, file_name):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass
        self.save_intermediate_record = True
        if file_name.endswith('.pkl'):
            file_name = file_name[:-4]
        self.save_intermediate_record_path = os.path.join(dir_path, file_name)
        self.logger.info('set save_intermediate_record to True. path: %s.' % (self.save_intermediate_record_path,))

    def save_intermediate_statistics(self):
        if self.save_intermediate_record:
            self.save_intermediate_record_id += 1
            path = '%s_%d.pkl' % (self.save_intermediate_record_path, self.save_intermediate_record_id)
            with open(path, 'wb') as f:
                pkl.dump(self.recorder, f)
            global_time = time.time() - self.global_start_time
            self.logger.info('Intermediate record %s saved! global_time=%.2fs.' % (path, global_time))

        # file_name = '%s.npy' % self.method_name
        # x = np.array(self._history['time_elapsed'])
        # y = np.array(self._history['performance'])
        # np.save(os.path.join(self.data_directory, file_name), np.array([x, y]))
        #
        # if PLOT:
        #     plt.plot(x, y)
        #     plt.xlabel('Time elapsed (sec)')
        #     plt.ylabel('Validation error')
        #     plt.savefig("data/%s.png" % self.method_name)
        return

    def _get_logger(self, name):
        logger_name = name
        setup_logger(os.path.join(self.log_directory, '%s.log' % str(logger_name)), None)
        return get_logger(self.__class__.__name__)

    def run(self):
        raise NotImplementedError
