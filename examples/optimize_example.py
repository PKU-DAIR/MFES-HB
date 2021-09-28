"""
example cmdline:

[master node]
python examples/optimize_example.py --role master --port 13579 --n_workers 2 --R 27

[worker nodes]
python examples/optimize_example.py --role worker --ip 127.0.0.1 --port 13579 --R 27

"""

import time
import argparse
from functools import partial
import numpy as np
import sys
sys.path.insert(0, '.')
from mfeshb.optimizer import Worker, MFESHB
from examples.problem import CountingOnes

parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int)

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port
R = args.R
eta = args.eta
n_workers = args.n_workers

problem = CountingOnes(4, 4)
config_space = problem.get_configspace()


def objective_function(config, n_resource, extra_conf, total_resource, eta, problem, continue_training):
    print('objective extra conf:', extra_conf)

    fidelity = n_resource / total_resource
    perf = problem.evaluate_config(config, fidelity=fidelity)
    print('config: %s, resource: %f/%f, perf=%f'
          % (config, n_resource, total_resource, perf))

    # simulate
    eval_time = 27 * n_resource / total_resource
    if continue_training and not extra_conf['initial_run']:
        eval_time -= 27 * n_resource / eta / total_resource
    sleep_factor = 0.5
    print('sleep %.2fs.' % (eval_time * sleep_factor))
    time.sleep(eval_time * sleep_factor)
    print('end sleep.')

    result = dict(
        objective_value=perf,  # minimize
    )
    return result


if role == 'master':
    method_name = 'mfeshb'
    problem_name = 'countingones'
    seed = 5162
    method_id = '%s-n%d-%s-%d' % (method_name, n_workers, problem_name, seed)
    optimizer = MFESHB(
        None, config_space, R, eta=eta,
        num_iter=999999, random_state=seed,
        method_id=method_id, restart_needed=True,
        time_limit_per_trial=999999,
        runtime_limit=300,
        ip='', port=port, authkey=b'abc',
    )
    optimizer.run()
    print('===== Optimization Finished =====')
    print('> last 3 records:')
    print(optimizer.recorder[-3:])
    print('> incumbent configuration and performance:')
    print(optimizer.get_incumbent())
else:
    obj_func = partial(objective_function, total_resource=R, eta=eta, problem=problem, continue_training=True)
    worker = Worker(obj_func, ip, port, authkey=b'abc')
    worker.run()
