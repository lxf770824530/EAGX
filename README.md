# EAGX

  A novel GNN explanation framework towards embedding ambiguity-sensitive Graph Neural Network explainability

## Requirement

Python 3.9

Pytorch 1.13.0

Pytorch-Geometric 2.3.0

network 2.5.1

## Data

There four datasets are used in our work, namely, BA-3motif, Mutagenicity, BBBP, and AIDS. The generation code of BA-3motif and Mutagenicity datasets can be found in the [code repository](https://github.com/Wuyxin/ReFine) of ReFine[^1]. BBBP and AIDS are available on [pyG](https://pytorch-geometric.readthedocs.io/en/latest/).

| Dataset      | Task                  | Data class     |
|--------------|-----------------------|----------------|
| BA-3motif    | Graph classification  | Synthetic data |
| Mutagenicity | Graph classification  | Real-word data |
| BBBP         | Graph classification  | Real-word data |
| AIDS         | Graph classification  | Real-word data |


## How to use

1. Training GNN models by runing 'GNNs/GNN.py', you need to configure the parameters related to GNN training in the 'GNN.py'.

2. Configuring the parameters related to our EAGX in the file ‘config.py’, and then execute ‘Run_fuzzyexplainer.py’

3. The explanation results will be saved in the folder '/result'.  




[^1]:Wang X,Wu Y, Zhang A, et al. Towards multi-grained explainability for graph neural networks. Advances in Neural Information Processing Systems, 2021, 34: 18446-18458.]
