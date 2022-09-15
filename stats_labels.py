import os
import pickle

from collections import Counter


def load_dataset(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        _ = pickle.load(inputfile)
        y = pickle.load(inputfile)
    return y


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'reddit',
        'EFCAMDAT2',
        'LOCNESS',
        'redditEN',
        'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'
    ]:
        y = load_dataset(dataset)
        labels = set(y)
        print(dataset, len(y), len(labels))
        counts = Counter(y)
        for label in labels:
            print(label, counts[label])
