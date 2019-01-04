import chainer


def sparse_to_gpu(x, device):
    x.data.data = chainer.backends.cuda.to_gpu(x.data.data, device=device)
    x.row = chainer.backends.cuda.to_gpu(x.row, device=device)
    x.col = chainer.backends.cuda.to_gpu(x.col, device=device)
    return x


def sparse_to_cpu(x):
    x.data.data = chainer.backends.cuda.to_cpu(x.data.data)
    x.row = chainer.backends.cuda.to_cpu(x.row)
    x.col = chainer.backends.cuda.to_cpu(x.col)
    return x