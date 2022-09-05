import os
import pickle
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def feture_stats(dataset, dataset_en):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '_indexed.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)

    with open(os.path.join('data', dataset_en + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y_en = pickle.load(inputfile)

    with open(os.path.join('data', dataset_en + '_indexed.pkl'), mode='rb') as inputfile:
        X_en = pickle.load(inputfile)

    stats_dir = 'feature_stats_vs_l1en'

    os.makedirs(stats_dir, exist_ok=True)

    to_test = ['WL', 'SL', 'DD']

    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker',
                                                                                                ['^', ',', '.']))

    for feat in to_test:
        counter_en = Counter()
        for doc in X_en:
            counter_en.update(doc[feat])

        counter = defaultdict(Counter)
        for doc, label in zip(X, y):
            counter[label].update(doc[feat])

        diffs = defaultdict(dict)
        for label in counter:
            feats = set(counter_en.keys())
            feats.update(set(counter[label].keys()))
            for value in feats:
                diffs[label][value] = counter[label][value]-counter_en[value]

        min_val = 100000
        max_val = -100000
        for label in diffs:
            min_val = min(min_val, *[int(key[len(feat) + 1:]) for key in diffs[label].keys()])
            max_val = max(max_val, *[int(key[len(feat) + 1:]) for key in diffs[label].keys()])

        max_val = min(max_val, 200)

        print(dataset, feat, 'LANG', sep='\t', end='\t')
        labels = list()
        for value in range(min_val, max_val + 1):
            print(value, end='\t')
            labels.append(value)
        print()
        plt.cycler(monochrome)
        for label in diffs:
            series = list()
            # den = sum([np.log(value) for value in counter[label].values()])
            print(dataset, feat, label, sep='\t', end='\t')
            for value in range(min_val, max_val + 1):
                try:
                    series.append(diffs[label][feat + "_" + str(value)])
                    print(f'{diffs[label][feat + "_" + str(value)]}', end='\t')
                except KeyError:
                    series.append(0)
                    print('0', end='\t')
            plt.plot(labels, series, label=label)
            print()
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(stats_dir, dataset + '_' + feat + '.pdf'))
        plt.close()


if __name__ == '__main__':
    for dataset in ['toefl11', 'reddit500k', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3']:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            dataset_en = 'LOCNESS'
        else:
            dataset_en = 'reddit500kEN'
        feture_stats(dataset, dataset_en)
