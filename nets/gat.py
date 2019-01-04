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
            self.gconv1 = GraphAttentionConvolution(features.shape[1], feat_size)
            self.gconv2 = GraphAttentionConvolution(feat_size, n_class)
        self.adj = adj
        self.features = features
        self.labels = labels
        self.dropout = dropout

    def _forward(self):
        if isinstance(self.features, CooMatrix):
            features = copy.deepcopy(self.features)
            features.data = F.dropout(features.data, self.dropout)
        else:
            features = F.dropout(self.features)
        h = F.relu(self.gconv1(features, self.adj))
        h = F.dropout(h, self.dropout)
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
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(GAT, self).to_gpu(device=device)

    def to_cpu(self):
        self.adj = sparse_to_cpu(self.adj)
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(GAT, self).to_cpu()


class GraphAttentionConvolution(chainer.Link):
    def __init__(self, in_size, out_size=None, nobias=True, initialW=None,
                 initial_bias=None):
        super(GraphAttentionConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            if initialW is None:
                initialW = initializers.GlorotUniform()
            self.W = chainer.Parameter(initialW, (in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, out_size)
            self.attention_linear = L.Linear(out_size * 2, 1)

    def attention(self, x, adj):
        att = copy.deepcopy(adj)
        h = F.hstack((x[adj.col], x[adj.row]))
        h = F.squeeze(self.attention_linear(h), -1)
        att.data = F.leaky_relu(h, 0.2)
        # Scaling trick for numerical stability
        att.data -= F.max(att.data)
        att.data = F.exp(att.data)
        rowsum = F.sparse_matmul(
            att, self.xp.ones([att.shape[1], 1], dtype=att.data.dtype))
        rowsum = 1. / F.squeeze(rowsum, 1)
        # We could've just converted rowsum to diagonal matrix and do sparse_matmul
        # but current sparse_matmul does not support two sparse matrix inputs
        att.data = att.data * rowsum[att.row]
        return att

    def __call__(self, x, adj):
        if isinstance(x, chainer.utils.CooMatrix):
            x = F.sparse_matmul(x, self.W)
        else:
            x = F.matmul(x, self.W)
        att = self.attention(x, adj)
        output = F.sparse_matmul(att, x)

        if self.b is not None:
            output += self.b

        return output
