# -*- coding: utf-8 -*-
"""Top-down Tree GRU (Python 3 / PyTorch port of the original Theano model)."""
import numpy as np
import torch
from BU_RvNN import hard_sigmoid, sgd_clip_step


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def gen_nn_inputs(root_node, ini_word):
    """tree[k] = [parent_idx, child_idx] in breadth-first order; tree[0] = [0, root_idx]
    (node 0 is a dummy all-zero state). Node idx values are the 1-based tweet indices."""
    tree = [[0, root_node.idx]]
    X_word, X_index = [root_node.word], [root_node.index]
    internal_tree, internal_word, internal_index = _get_tree_path(root_node)
    tree.extend(internal_tree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)  # extra row: only gives the state table its size (N+1)
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int64'),
            np.array(tree, dtype='int64'))


def _get_tree_path(root_node):
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
    tree, word, index = [], [], []
    for layer in layers:
        for node in layer:
            for child in node.children:
                tree.append([node.idx, child.idx])
                word.append(child.word)
                index.append(child.index)
    return tree, word, index


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

    def _forward(self, x_word, x_index, num_parent, tree):
        word = torch.from_numpy(x_word[:-1])
        index = torch.from_numpy(x_index)
        xe = (self.E.t()[index] * word.unsqueeze(-1)).sum(dim=1)            # [steps, H]
        a_z = xe @ self.W_z.t() + self.b_z
        a_r = xe @ self.W_r.t() + self.b_r
        a_h = xe @ self.W_h.t() + self.b_h

        zero = torch.zeros(self.hidden_dim)
        states = [zero] * x_word.shape[0]          # one slot per node idx (0..N)
        child_hs = []
        for k, (p, c) in enumerate(tree.tolist()):
            parent_h = states[p]
            z = hard_sigmoid(a_z[k] + self.U_z @ parent_h)
            r = hard_sigmoid(a_r[k] + self.U_r @ parent_h)
            cand = torch.tanh(a_h[k] + self.U_h @ (parent_h * r))
            h = z * parent_h + (1 - z) * cand
            states[c] = h
            child_hs.append(h)
        final_state = torch.stack(child_hs[num_parent - 1:]).max(dim=0).values
        return torch.softmax(self.W_out @ final_state + self.b_out, dim=-1).unsqueeze(0)  # [1, 4]

    def train_step_up(self, x_word, x_index, num_parent, tree, y, lr):
        pred = self._forward(x_word, x_index, num_parent, tree)
        y_t = torch.tensor(y, dtype=torch.float32)
        loss = ((y_t - pred) ** 2).sum()
        grads = torch.autograd.grad(loss, self.params)
        sgd_clip_step(self.params, grads, lr)
        return loss.item(), pred.detach().numpy()

    def predict_up(self, x_word, x_index, num_parent, tree):
        with torch.no_grad():
            return self._forward(x_word, x_index, num_parent, tree).numpy()
