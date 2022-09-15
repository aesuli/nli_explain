import os
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def plot_features_stats(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    stats_dir = 'feature_stats'

    os.makedirs(stats_dir, exist_ok=True)

    to_test = ['WL', 'SL', 'DD']

    for feat in to_test:
        with open(os.path.join('data', dataset + '_indexed_' + feat + '.pkl'), mode='rb') as inputfile:
            X = pickle.load(inputfile)
        counter = defaultdict(Counter)
        for doc, label in zip(X, y):
            counter[label].update(doc)

        min_val = 0
        if feat == 'WL':
            max_val = 20
        elif feat == 'SL':
            max_val = 100
        elif feat == 'DD':
            max_val = 20

        print(dataset, feat, 'LANG', sep='\t', end='\t')
        labels = list()
        for value in range(min_val, max_val + 1):
            print(value, end='\t')
            labels.append(value)
        print()
        for label in sorted(counter):
            series = list()
            den = sum([value for value in counter[label].values()])
            print(dataset, feat, label, sep='\t', end='\t')
            for value in range(min_val, max_val + 1):
                series.append(counter[label][feat + "_" + str(value)] / den)
                print(f'{counter[label][feat + "_" + str(value)] / den:3.3}', end='\t')
            plt.semilogy(labels, series, label=label)
            print()
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(stats_dir, dataset + '_' + feat + '.pdf'))
        plt.close()


if __name__ == '__main__':
    for dataset in ['toefl11', 'reddit',
                    'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'
                    ]:
        plot_features_stats(dataset)
