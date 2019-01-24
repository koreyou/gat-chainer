# Text GCN on Chainer

This project implements GCN ([Kipf and Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. ICLR.](https://arxiv.org/abs/1609.02907)) and GAT ([Veličković et al. 2018. Graph Attention Networks](https://arxiv.org/abs/1710.10903)) with [Chainer](https://chainer.org/).
The project includes codes to reproduce the experiments on multiple graph classification datasets. **This is NOT an official implementation by the authors.**

I referenced [@takyamamoto's implementation of GCN](https://github.com/takyamamoto/Graph-Convolution-Chainer) to implement this project.

# How to Run

## Prerequisite

I have only tested the code on Python 3.6.4. Install dependent library by running:

```
pip install -r requirements.txt
```

You need to install `cupy` to enable GPU.

## Running training and evaluation

Run:

```
python train.py
```

Refer to `python train.py -h` for the options.
Note that you can enable early stopping by `--early-stopping` flag, but the overhead for saving intermediate models is quite large.


# Reproducing the paper

## GCN

Run training in with same parameters as the original paper.

```
python train.oy --model gcn --lr 0.01 --unit 16 --dataset $DATASET
```

Following table shows the average (min/max) over 10 runs.


| Dataset  | Reported in the paper | My implementation |
|----------|-----------------------|-------------------|
| Cora | 81.5 | 81.6 (81.2/82.3) |
| Citeseer | 70.3 | 71.2 (70.9/71.7) |
| Pubmed | 79.0 | 78.8 (78.4/79.2) |

My implementation is comparable with the reported result.

## GAT

```
python train.oy --dataset $DATASET --early-stopping
```

Average test accuracy was 83.7 (from 82.6 to 84.7) over 10 runs.
The test accuracy reported in the paper was 83.0, so my implementation is comparable with the reported result.

My implementation took 0.3214s per epoch on K80 GPU (Google Colabolatory), which is slightly slower than 0.3002s (non-sparse) and 0.2043 (sparse) of the tensorflow implementation by the authors.


# Licensing

`load_data` module and all files under `data/` directory have been derived from [Dr. Kipf's repository](https://github.com/tkipf/gcn/tree/98357bded82fdc19595aa5b1448ee0e76557a399), so refer to the original repository for licensing.
Other files are distributed under [CC0](./LICENSE).
