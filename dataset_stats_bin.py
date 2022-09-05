import os
import pickle

from collections import Counter, defaultdict
from pprint import pprint
from typing import DefaultDict


def load_dataset(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        text = pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '_indexed.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
    return text, X, y


if __name__ == '__main__':
    for l in [
        ['toefl11'],
        ['reddit500k'],
        ['EFCAMDAT2'],
        ['LOCNESS'],
        ['reddit500kEN'],
        ['toefl11', 'LOCNESS'],
        ['EFCAMDAT2', 'LOCNESS'],
        ['reddit500k', 'reddit500kEN'],
    ]:
        print(l)
        feat_sets = defaultdict(set)
        for d in l:
            text, X, y = load_dataset(d)
            for x in X:
                for k, v in x.items():
                    feat_sets[k].update(v)
        for k,v in feat_sets.items():
            print(k, len(v))
