# -*- coding: utf-8 -*-
"""Bottom-up Tree GRU (Python 3 / PyTorch port of the original Theano model)."""
import numpy as np
import torch


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


################################# tree structure ##############################
def gen_nn_inputs(root_node, max_degree=None):
    """Returns (X_word, X_index, tree).
    Nodes are numbered leaves first, then internal nodes bottom-up (root last).
    tree[i] = [child_1, ..., child_degree, own_idx] with -1 for missing children."""
    _clear_indices(root_node)
    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word), max_degree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int64'),
            np.array(tree, dtype='int64'))


def _clear_indices(root_node):
    root_node.idx = None
    for child in root_node.children:
        _clear_indices(child)


def _get_leaf_vals(root_node):
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend(node.children[::-1])
        layer = next_layer
    X_word, X_index = [], []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)
    return X_word, X_index


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        for node in layer:
            next_layer.extend(node.children)
        layer = next_layer
    tree, internal_word, internal_index = [], [], []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:  # leaf, already numbered
                continue
            child_idxs = [child.idx for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(i is None for i in child_idxs)
            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word)
            internal_index.append(node.index)
            idx += 1
    return tree, internal_word, internal_index


################################ model ########################################
def hard_sigmoid(x):  # same definition as Theano's T.nnet.hard_sigmoid
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class RvNN(object):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4, degree=2, momentum=0.9):
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim, self.hidden_dim, self.Nclass = word_dim, hidden_dim, Nclass

        def mat(*shape):
            return torch.nn.Parameter(torch.randn(*shape) * 0.1)

        def vec(n):
            return torch.nn.Parameter(torch.zeros(n))

        H = hidden_dim
        self.E = mat(H, word_dim)
        self.W_z, self.U_z, self.b_z = mat(H, H), mat(H, H), vec(H)
        self.W_r, self.U_r, self.b_r = mat(H, H), mat(H, H), vec(H)
        self.W_h, self.U_h, self.b_h = mat(H, H), mat(H, H), vec(H)
        self.W_out, self.b_out = mat(Nclass, H), vec(Nclass)
        self.params = [self.E, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h, self.W_out, self.b_out]

    def _forward(self, x_word, x_index, tree):
        word = torch.from_numpy(x_word)
        index = torch.from_numpy(x_index)
        # embedding of every node at once: sum_k freq_k * E[:, index_k]
        xe = (self.E.t()[index] * word.unsqueeze(-1)).sum(dim=1)          # [n, H]
        a_z = xe @ self.W_z.t() + self.b_z
        a_r = xe @ self.W_r.t() + self.b_r
        a_h = xe @ self.W_h.t() + self.b_h
        num_parents = tree.shape[0]
        num_leaves = xe.shape[0] - num_parents

        # leaves: children states are zero
        z = hard_sigmoid(a_z[:num_leaves])
        c = torch.tanh(a_h[:num_leaves])
        states = list(((1 - z) * c).unbind(0))

        # internal nodes, bottom-up
        for t, row in enumerate(tree.tolist()):
            i = num_leaves + t
            kids = [k for k in row[:-1] if k > -1]
            h_tilde = torch.stack([states[k] for k in kids]).sum(dim=0)
            z = hard_sigmoid(a_z[i] + self.U_z @ h_tilde)
            r = hard_sigmoid(a_r[i] + self.U_r @ h_tilde)
            c = torch.tanh(a_h[i] + self.U_h @ (h_tilde * r))
            states.append(z * h_tilde + (1 - z) * c)
        final_state = states[-1]                                            # root
        return torch.softmax(self.W_out @ final_state + self.b_out, dim=-1).unsqueeze(0)  # [1, 4]

    def train_step_up(self, x_word, x_index, tree, y, lr):
        pred = self._forward(x_word, x_index, tree)
        y_t = torch.tensor(y, dtype=torch.float32)
        loss = ((y_t - pred) ** 2).sum()
        grads = torch.autograd.grad(loss, self.params)
        sgd_clip_step(self.params, grads, lr)
        return loss.item(), pred.detach().numpy()

    def predict_up(self, x_word, x_index, tree):
        with torch.no_grad():
            return self._forward(x_word, x_index, tree).numpy()


def sgd_clip_step(params, grads, lr, clip=5.0):
    """Gradient-norm clipping + SGD, as in the original gradient_descent().
    (The original 'momentum' was never applied: its velocity was a constant 0.)"""
    with torch.no_grad():
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
        finite = bool(torch.isfinite(norm))
        scale = clip / max(clip, float(norm)) if finite else 0.0
        for p, g in zip(params, grads):
            g = g * scale if finite else 0.1 * p
            p -= lr * g
