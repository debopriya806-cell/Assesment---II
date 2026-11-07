# -*- coding: utf-8 -*-
"""Bottom-up RvNN for rumor detection (4 classes) - Python 3 / PyTorch port.
Usage: python Main_BU_RvNN.py --obj Twitter15 --fold 3 --epochs 20
"""
import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch

import BU_RvNN
from common import (ROOT, read_labels, read_trees, load_split, str2matrix)
from evaluate import evaluation_4class

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="Twitter15", choices=["Twitter15", "Twitter16"])
ap.add_argument("--fold", default="3", choices=["0", "1", "2", "3", "4"])
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--lr", type=float, default=0.005)
ap.add_argument("--hidden_dim", type=int, default=100)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--max_samples", type=int, default=0, help="limit trees per split (quick test)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
vocabulary_size, Nclass = 5000, 4
lr = args.lr


def constructTree(tree):
    """tree: {idx: {'parent', 'max_degree', 'maxL', 'vec'}} -> (x_word, x_index, tree)"""
    index2node = {i: BU_RvNN.Node_tweet(idx=i) for i in tree}
    root = None
    for j in tree:
        nodeC = index2node[j]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'], tree[j]['maxL'])
        nodeC.index, nodeC.word = wordIndex, wordFreq
        if tree[j]['parent'] != 'None':
            nodeP = index2node[int(tree[j]['parent'])]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            root = nodeC
    degree = max(tree[j]['max_degree'], max(len(n.children) for n in index2node.values()))
    return BU_RvNN.gen_nn_inputs(root, max_degree=degree)


################################## load data ##################################
print("loading tree labels:", end=" ")
labelDic = read_labels(args.obj)
print(len(labelDic))
print("reading trees:", end=" ")
treeDic = read_trees(os.path.join(ROOT, "resource", "data.BU_RvNN.vol_%d.txt" % vocabulary_size), 'max_degree')
print(len(treeDic))
data = load_split(args.obj, args.fold, treeDic, labelDic, constructTree,
                  min_nodes=2, max_samples=args.max_samples)
train_x, y_train = data["train"]
test_x, y_test = data["test"]

################################## model ######################################
t0 = time.time()
model = BU_RvNN.RvNN(vocabulary_size, args.hidden_dim, Nclass)
print("Recursive model established,", (time.time() - t0) / 60)

################################## training ###################################
losses_5, losses, num_examples_seen = [], [], 0
for epoch in range(args.epochs):
    order = list(range(len(y_train)))
    random.shuffle(order)
    for i in order:
        xw, xi, tr = train_x[i]
        loss, _ = model.train_step_up(xw, xi, tr, y_train[i], lr)
        losses.append(loss)
        num_examples_seen += 1
    print("epoch=%d: loss=%f" % (epoch, np.mean(losses)))
    sys.stdout.flush()

    if epoch % args.eval_every == 0:
        losses_5.append((num_examples_seen, np.mean(losses)))
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (now, num_examples_seen, epoch, np.mean(losses)))
        prediction = [model.predict_up(*test_x[j][:3]) for j in range(len(y_test))]
        res = evaluation_4class(prediction, y_test)
        print("results:", res)
        if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
            lr = lr * 0.5
            print("Setting learning rate to %f" % lr)
        sys.stdout.flush()
    losses = []
