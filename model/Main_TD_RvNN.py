# -*- coding: utf-8 -*-
"""Top-down RvNN for rumor detection (4 classes) - Python 3 / PyTorch port.
Usage: python Main_TD_RvNN.py --obj Twitter15 --fold 3 --epochs 20
"""
import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch

import TD_RvNN
from common import (ROOT, read_labels, read_trees, load_split, str2matrix)
from evaluate import evaluation_4class

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="Twitter15", choices=["Twitter15", "Twitter16"])
ap.add_argument("--fold", default="2", choices=["0", "1", "2", "3", "4"])
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--lr", type=float, default=0.005)
ap.add_argument("--hidden_dim", type=int, default=100)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--max_samples", type=int, default=0, help="limit trees per split (quick test)")
ap.add_argument("--shuffle", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
vocabulary_size, Nclass = 5000, 4
lr = args.lr


def constructTree(tree):
    """tree: {idx: {'parent', 'parent_num', 'maxL', 'vec'}} -> (x_word, x_index, tree, parent_num)"""
    index2node = {i: TD_RvNN.Node_tweet(idx=i) for i in tree}
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
    parent_num = tree[j]['parent_num']
    ini_x, _ = str2matrix("0:0", tree[j]['maxL'])
    x_word, x_index, tr = TD_RvNN.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tr, parent_num


################################## load data ##################################
print("loading tree labels:", end=" ")
labelDic = read_labels(args.obj)
print(len(labelDic))
print("reading trees:", end=" ")
treeDic = read_trees(os.path.join(ROOT, "resource", "data.TD_RvNN.vol_%d.txt" % vocabulary_size), 'parent_num')
print(len(treeDic))
data = load_split(args.obj, args.fold, treeDic, labelDic, constructTree,
                  min_nodes=1, max_samples=args.max_samples)
train_x, y_train = data["train"]
test_x, y_test = data["test"]

################################## model ######################################
t0 = time.time()
model = TD_RvNN.RvNN(vocabulary_size, args.hidden_dim, Nclass)
print("Recursive model established,", (time.time() - t0) / 60)

################################## training ###################################
losses_5, losses, num_examples_seen = [], [], 0
for epoch in range(args.epochs):
    order = list(range(len(y_train)))
    # (the original TD script keeps the shuffle disabled; pass --shuffle to enable it)
    if args.shuffle:
        random.shuffle(order)
    for i in order:
        xw, xi, tr, pn = train_x[i]
        loss, _ = model.train_step_up(xw, xi, pn, tr, y_train[i], lr)
        losses.append(round(loss, 2))
        num_examples_seen += 1
    print("epoch=%d: loss=%f" % (epoch, np.mean(losses)))
    sys.stdout.flush()

    if epoch % args.eval_every == 0:
        losses_5.append((num_examples_seen, np.mean(losses)))
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (now, num_examples_seen, epoch, np.mean(losses)))
        prediction = [model.predict_up(test_x[j][0], test_x[j][1], test_x[j][3], test_x[j][2])
                      for j in range(len(y_test))]
        res = evaluation_4class(prediction, y_test)
        print("results:", res)
        if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
            lr = lr * 0.5
            print("Setting learning rate to %f" % lr)
        sys.stdout.flush()
    losses = []
