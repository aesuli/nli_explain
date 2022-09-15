import os
import pickle
from collections import Counter, defaultdict
from pprint import pprint

from tqdm.auto import tqdm


def get_token_count(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        _ = pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '_indexed_T1.pkl'), mode='rb') as outputfile:
        indexed = pickle.load(outputfile)
    token_count = 0
    y_token_count = defaultdict(int)
    labels = defaultdict(int)
    for y, doc in zip(y, indexed):
        t1_count = len(doc)
        token_count += t1_count
        y_token_count[y] += t1_count
        labels[y] += 1
    print(dataset, len(indexed))
    pprint(labels)
    print('token_count', dataset, token_count)
    pprint(y_token_count)


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'reddit',
        'EFCAMDAT2',
        'LOCNESS',
        'redditEN',
        'EFCAMDAT2_L1',
        'EFCAMDAT2_L2',
        'EFCAMDAT2_L3'
    ]:
        get_token_count(dataset)
