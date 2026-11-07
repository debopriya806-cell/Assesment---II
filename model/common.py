# -*- coding: utf-8 -*-
"""Shared data helpers (Python 3 port of the loading code in Main_*_RvNN.py)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def str2matrix(Str, MaxL):  # Str = "index:wordfreq index:wordfreq ..."
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        idx, freq = pair.split(':')
        wordFreq.append(float(freq))
        wordIndex.append(int(idx))
    pad = [0] * (MaxL - len(wordFreq))
    return wordFreq + pad, wordIndex + pad


def loadLabel(label):
    """Returns a 4-class one-hot list, or None for unknown labels."""
    if label in ('news', 'non-rumor'):
        return [1, 0, 0, 0]
    if label == 'false':
        return [0, 1, 0, 0]
    if label == 'true':
        return [0, 0, 1, 0]
    if label == 'unverified':
        return [0, 0, 0, 1]
    return None


def read_labels(obj):
    path = os.path.join(ROOT, "resource", obj + "_label_All.txt")
    labelDic = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            labelDic[parts[2]] = parts[0].lower()
    return labelDic


def read_trees(tree_file, third_col_name):
    """tree_file columns: eid, parent idx, own idx, <third_col_name>, maxL, 'idx:freq ...'."""
    treeDic = {}
    with open(tree_file, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').rstrip('\r').split('\t')
            eid, indexP, indexC = p[0], p[1], int(p[2])
            treeDic.setdefault(eid, {})[indexC] = {
                'parent': indexP, third_col_name: int(p[3]),
                'maxL': int(p[4]), 'vec': p[5]}
    return treeDic


def load_split(obj, fold, treeDic, labelDic, build_fn, min_nodes, max_samples=None):
    """Builds (inputs, y) lists for the train and test split of one fold."""
    out = {}
    for split in ("train", "test"):
        path = os.path.join(ROOT, "nfold", "RNN%sSet_%s%s_tree.txt" % (split, obj, fold))
        samples, ys, counts = [], [], [0, 0, 0, 0]
        with open(path, encoding='utf-8') as f:
            for eid in f:
                eid = eid.strip()
                if eid not in labelDic or eid not in treeDic:
                    continue
                if len(treeDic[eid]) < min_nodes:
                    continue
                y = loadLabel(labelDic[eid])
                if y is None:
                    continue
                counts[y.index(1)] += 1
                ys.append(y)
                samples.append(build_fn(treeDic[eid]))
                if max_samples and len(ys) >= max_samples:
                    break
        print("%s set: %d trees, class counts (non-rumor/false/true/unverified) = %s"
              % (split, len(ys), counts))
        out[split] = (samples, ys)
    return out
