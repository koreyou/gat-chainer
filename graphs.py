import pickle as pkl
import sys

import chainer
import networkx as nx
import numpy as np
import scipy.sparse as sp


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, normalization='gcn'):
    """
    Loads input data from gcn/data directory
    This function was adopted from https://github.com/tkipf/gcn/blob/98357b/gcn/utils.py

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.

    Arguments:
        dataset_str (str): Name of the dataset
        normalization (str): Variant of normalization method to use.

    Returns:
        adj (chainer.utils.sparse.CooMatrix): (Node, Node) shape
            normalized adjency matrix.
        features (chainer.utils.sparse.CooMatrix): (Node, feature size) shape
            normalized feature matrix.
        labels (np.ndarray): (Node, ) shape labels array
        idx_train (np.ndarray): Indices of the train
        idx_val (np.ndarray): Indices of val array
        idx_test (np.ndarray): Indices of test array
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = to_chainer_sparse_variable(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32)

    labels = np.vstack((ally, ty))
    # in citeseer dataset, there are all-none node, which we need to remove
    zero_indices = np.where(1 - labels.sum(1))[0]
    labels = np.argmax(labels, axis=1)
    if dataset_str == 'citeseer':
        labels[zero_indices] = -1
    labels[test_idx_reorder] = labels[test_idx_range]
    labels = labels.astype(np.int32)

    idx_test = np.array(test_idx_range.tolist(), np.int32)
    idx_train = np.array(list(range(len(y))), np.int32)
    idx_val = np.array(list(range(len(y), len(y)+500)), np.int32)

    if normalization == 'gcn':
        adj = normalize(adj)
    else:
        adj = normalize_pygcn(adj)

    adj = to_chainer_sparse_variable(adj)

    return adj, features, labels, idx_train, idx_val, idx_test


def preprocess_features(features):
    """ Row-normalize feature matrix and convert to tuple representation
    This function was adopted from https://github.com/tkipf/gcn/blob/98357b/gcn/utils.py
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_pygcn(a):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    a += sp.eye(a.shape[0])
    rowsum = np.array(a.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(a)


def normalize(adj):
    """ normalize adjacency matrix with normalization-trick that is faithful to
    the original paper.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ~D in the GCN paper
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


def to_chainer_sparse_variable(mat):
    mat = mat.tocoo().astype(np.float32)
    ind = np.argsort(mat.row)
    data = mat.data[ind]
    row = mat.row[ind]
    col = mat.col[ind]
    shape = mat.shape
    # check that adj's row indices are sorted
    assert np.all(np.diff(row) >= 0)
    return chainer.utils.CooMatrix(data, row, col, shape, order='C')
