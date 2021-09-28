# MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements

Implementation of MFES-HB (AAAI-21) along with Hyperband and BOHB.

Support distributed evaluation using message queue.

## Links
+ Paper Download: <https://ojs.aaai.org/index.php/AAAI/article/view/17031/16838>
+ Paper Homepage: <https://ojs.aaai.org/index.php/AAAI/article/view/17031>

## Reference
```
@article{mfeshb, 
  title={MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements}, 
  author={Li, Yang and Shen, Yu and Jiang, Jiawei and Gao, Jinyang and Zhang, Ce and Cui, Bin}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  volume={35}, 
  number={10},
  year={2021}, 
  month={May}, 
  pages={8491-8500}
}
```

## Project Directory

+ **mfeshb/optimizer/**: Hyperband, BOHB, MFES-HB optimizers.
+ **mfeshb/surrogate/**: multi-fidelity ensemble surrogate.
+ **examples**/: Optimization examples.

## Install Requirements

Python 3.7 is recommended.

First, install SWIG3.0:
```shell script
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

For MacOS and Windows users, please refer to the 
[SWIG Installation Guide](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

Then, run the following commands:
```shell script
cat requirements.txt | xargs -n 1 -L 1 pip install
```

If you encounter problems installing `openbox`, please refer to the 
[OpenBox Installation Guide](https://open-box.readthedocs.io/en/latest/installation/installation_guide.html).

## Quick Start

We provide an implementation of MFES-HB based on message queue to support distributed evaluation.

In this example, we optimize the multi-fidelity Counting Ones function using MFES-HB. \[[code](./examples/optimize_example.py)\]

### Run Optimizer

Start the optimizer first. The master will listen on the `port` and wait for workers to connect.
```
python examples/optimize_example.py --role master --port 13579 --n_workers 2 --R 27
```

### Run Workers

After the optimizer starts, run workers to get jobs and evaluate configurations. Please set master `ip` and `port`.
```shell script
python examples/optimize_example.py --role worker --ip 127.0.0.1 --port 13579 --R 27
```
