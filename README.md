# A Critical Assessment of State-of-the-Art in Entity Alignment

[![Arxiv](https://img.shields.io/badge/arXiv-2010.16314-b31b1b)](https://arxiv.org/abs/2010.16314)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the paper
```
A Critical Assessment of State-of-the-Art in Entity Alignment
Max Berrendorf, Ludwig Wacker, and Evgeniy Faerman
https://arxiv.org/abs/2010.16314
```

# Installation
Setup and activate virtual environment:
```shell script
python3.8 -m venv ./venv
source ./venv/bin/activate
```

Install requirements (in this virtual environment):
```shell script
pip install -U pip
pip install -U -r requirements.txt
```

In order to run the DGMC scripts, you additionally need to setup 
its requirements as described in the corresponding GitHub repository's 
[README](https://github.com/rusty1s/deep-graph-matching-consensus/blob/a25f89751f4a3a0d509baa6bbada8b4153c635f6/README.md).
We do not include them into [`requirements.txt`](./requirements.txt), 
since their installation is a bit more involved, including non-Python dependencies. 

# Preparation

## MLFlow
In order to track results to a MLFlow server, start it first by running
```shell script
mlflow server
```
_Note: When storing the result for many configurations, we recommend to setup a
database backend following the [instructions](https://mlflow.org/docs/latest/tracking.html)._
For the following examples, we assume that the server is running at
```shell script
TRACKING_URI=http://localhost:5000
```

## OpenEA RDGCN embeddings
Please download the RDGCN embeddings extracted with the [OpenEA codebase](https://github.com/nju-websoft/OpenEA/tree/2a6e0b03ec8cdcad4920704d1c38547a3ad72abe)
from [here](https://www.dbs.ifi.lmu.de/~berrendorf/ea-sota-comparison/openea_rdgcn_embeddings/)
and place them in `~/.kgm/openea_rdgcn_embeddings`.
They require around 160MiB storage.

## BERT initialization
To generate data for the BERT-based initialization, run
```shell script
(venv) PYTHONPATH=./src python3 executables/prepare_bert.py
```

We also provide preprocessed files at [this url](https://www.dbs.ifi.lmu.de/~berrendorf/ea-sota-comparison/bert_prepared/).
If you prefer to use those, please download and place them in `~/.kgm/bert_prepared`. 
They require around 6.1GiB storage. 

# Experiments

For all experiments the results are logged to the running MLFlow instance.

_Note: The hyperparameter searches takes a significant amount of time (~multiple days),
 and requires access to GPU(s). You can abort the script at any time, and inspect the
  current results via the web interface of MLFlow._


## Zero-Shot
For the zero-shot evaluation run
```shell script
(venv) PYTHONPATH=./src python3 executables/zero_shot.py --tracking_uri=${TRACKING_URI} 
```

## GCN-Align
To run the hyperparameter search run
```shell script
(venv) PYTHONPATH=./src python3 executables/tune_gcn_align.py --tracking_uri=${TRACKING_URI} 
```

## RDGCN
To run the hyperparameter search run
```shell script
(venv) PYTHONPATH=./src python3 executables/tune_rdgcn.py --tracking_uri=${TRACKING_URI} 
```

## DGMC
To run the hyperparameter search run
```shell script
(venv) PYTHONPATH=./src python3 executables/tune_dgmc.py  --tracking_uri=${TRACKING_URI} 
```

# Evaluation
To summarize the dataset statistics run
```shell script
(venv) PYTHONPATH=./src python3 executables/summarize.py --target datasets --force
```

To summarize all experiments run
```shell script
(venv) PYTHONPATH=./src python3 executables/summarize.py --target results --tracking_uri=${TRACKING_URI} --force
```

To generate the ablation study table run
```shell script
(venv) PYTHONPATH=./src python3 executables/summarize.py --target ablation --tracking_uri=${TRACKING_URI} --force
```
