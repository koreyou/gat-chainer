# Text GCN on Chainer

This project implements [Yao et al. 2018. Graph Convolutional Networks for Text Classification. ArXiv.](https://arxiv.org/abs/1809.05679) in [Chainer](https://chainer.org/).
The project includes codes to reproduce the text classification experiment on the 20 news groups dataset. **This is NOT an official implementation by the authors.**

This project adopts hyperparamters specified in the original paper.
I have greatly simplified text preprocessing to just splitting words with white spaces.

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

Running this project with the original adjacency matrix normalization method (`python train.py -g 0 --normalization gcn`) yields 0.8412 accuracy in the 20 News groups dataset.
The test accuracy was 0.8634 in the original paper.

Running `python train.py -g 0 --normalization pygcn` which uses normalization method proposed in [GCN authors' PyTorch implementation](https://github.com/tkipf/pygcn/issues/11) yields much better result of 0.8687 (comparable with the original paper).

# Licensing

`load_data` module and all files under `data/` directory have been derived from [Dr. Kipf's repository](https://github.com/tkipf/gcn/tree/98357bded82fdc19595aa5b1448ee0e76557a399), so refer to the original repository for licensing.
Other files are distributed under [CC0](./LICENSE).