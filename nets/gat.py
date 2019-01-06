import copy

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import initializers
from chainer import reporter
from chainer.utils.sparse import CooMatrix

from nets.common import sparse_to_gpu, sparse_to_cpu


class GAT(chainer.Chain):
    def __init__(self, adj, features, labels, feat_size, dropout=0.5):
        super(GAT, self).__init__()
        n_class = np.max(labels) + 1
        with self.init_scope():
            self.gconv1 = GraphAttentionConvolution(8, features.shape[1], feat_size)
            self.gconv2 = GraphAttentionConvolution(1, self.gconv1.out_size, n_class)
        self.adj = adj
        self.features = features
        self.labels = labels
        self.dropout = dropout

    def _forward(self):
        h = F.elu(self.gconv1(self.features, self.adj))
        out = self.gconv2(h, self.adj)
        return out

    def __call__(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        return loss

    def evaluate(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        return float(loss.data), float(accuracy.data)

    def predict(self, idx):
        out = self._forward()
        out = out[idx]
        pred = self.xp.argmax(out.data)
        return pred

    def predict_proba(self, idx):
        out = self._forward()
        out = out[idx]
        return out.data

    def to_gpu(self, device=None):
        self.adj = sparse_to_gpu(self.adj, device=device)
        self.features = sparse_to_gpu(self.features, device=device)
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(GAT, self).to_gpu(device=device)

    def to_cpu(self):
        self.adj = sparse_to_cpu(self.adj)
        self.features = sparse_to_cpu(self.features)
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(GAT, self).to_cpu()


class GraphAttentionConvolution(chainer.Chain):
    def __init__(self, n_heads, in_size, out_size=None, nobias=False, initialW=None,
                 initial_bias=None):
        super(GraphAttentionConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = n_heads * out_size
        self.n_heads = n_heads

        with self.init_scope():
            if initialW is None:
                initialW = initializers.GlorotUniform()
            self.W = chainer.Parameter(initialW, (1, self.n_heads, in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, self.out_size)
            self.attention_W = chainer.Parameter(
                initialW, (1, self.n_heads, out_size, 2))

    def attention(self, x, adj):
        # x's shape: (nodes, n_heads, out_size)
        att = copy.deepcopy(adj)
        # e's shape: (nodes, n_heads, 2)
        e = F.squeeze(F.matmul(x[:, :, None, :], self.attention_W), 2)
        # e_row's shape: (nodes, n_heads, 1)
        e_row, e_col = F.split_axis(e, 2, axis=2)
        # Linear(concat(x_row, x_col)) = Linear_row(x_row) + Linear_col(x_col)
        h = F.squeeze(e_row[adj.row] + e_col[adj.col], 2)
        att_data = F.leaky_relu(h, 0.2)
        # Scaling trick for numerical stability
        att_data -= self.xp.max(att_data.data)
        att_data = F.exp(att_data)
        x = F.dropout(x, 0.6)

        output = []
        for att_data_i, xi in zip(F.split_axis(att_data, self.n_heads, axis=1), F.split_axis(x, self.n_heads, axis=1)):
            att.data = F.squeeze(att_data_i, 1)
            rowsum = F.sparse_matmul(
                att, self.xp.ones([att.shape[1], 1], dtype=att.data.dtype))
            rowsum = 1. / F.squeeze(rowsum, 1)
            # We could've just converted rowsum to diagonal matrix and do sparse_matmul
            # but current sparse_matmul does not support two sparse matrix inputs
            att.data = att.data * rowsum[att.row]
            att.data = F.dropout(att.data, 0.6)
            output.append(F.sparse_matmul(att, F.squeeze(xi, 1)))
        output = F.concat(output, axis=1)
        return output

    def __call__(self, x, adj):
        if isinstance(x, chainer.utils.CooMatrix):
            x = copy.deepcopy(x)
            x_data = x.data
            z = []
            for i in range(self.n_heads):
                x.data = F.dropout(x_data, 0.6)
                z.append(F.sparse_matmul(x, self.W[0, i])[:, None, :])
            z = F.concat(z, axis=1)
        else:
            x = F.tile(x[:, None, :], (1, self.n_heads, 1))
            x = F.dropout(x, 0.6)
            z = F.squeeze(F.matmul(x[:, :, None, :], self.W), 2)
        output = self.attention(z, adj)

        if self.b is not None:
            output += self.b

        return output
