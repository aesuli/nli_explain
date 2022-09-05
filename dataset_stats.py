import os
import pickle

from collections import Counter


def load_dataset(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        text = pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '_indexed.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
    return text, X, y


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'reddit500k',
        'EFCAMDAT2',
        'LOCNESS',
        'reddit500kEN',
        'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'
    ]:
        text, X, y = load_dataset(dataset)
        labels = set(y)
        print(dataset, len(X), len(labels))
        counts = Counter(y)
        for label in labels:
            print(label, counts[label])
