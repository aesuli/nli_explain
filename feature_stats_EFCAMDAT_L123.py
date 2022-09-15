import os
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def feture_stats():
    dX = defaultdict(dict)
    dy = dict()
    to_test = ['WL', 'SL', 'DD']
    for level in ['EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3']:
        with open(os.path.join('data', level + '.pkl'), mode='rb') as inputfile:
            pickle.load(inputfile)
            dy[level[-2:]] = pickle.load(inputfile)

        for feat_type in to_test:
            with open(os.path.join('data', level + '_indexed_'+feat_type+'.pkl'), mode='rb') as inputfile:
                dX[level[-2:]][feat_type] = pickle.load(inputfile)

    stats_dir = 'feature_stats_L123'

    os.makedirs(stats_dir, exist_ok=True)


    for feat in to_test:
        counter = defaultdict(Counter)
        languages = set()
        for level in dX:
            for doc, label in zip(dX[level][feat], dy[level]):
                counter[level + '_' + label].update(doc)
                languages.add(label)

        min_val = 1

        if feat == 'WL':
            max_val = 20
        elif feat == 'SL':
            max_val = 100
        elif feat == 'DD':
            max_val = 20

        print(feat, 'LANG', sep='\t', end='\t')
        labels = list()
        for value in range(min_val, max_val + 1):
            labels.append(value)
        print()
        per_lang_lab_vals = dict()
        for language in languages:
            for label in sorted(counter):
                if not label.endswith(language):
                    continue
                series = list()
                den = sum([value for value in counter[label].values()])
                for value in range(min_val, max_val + 1):
                    series.append(counter[label][feat + "_" + str(value)] / den)
                per_lang_lab_vals[label] = series

        if feat in ['WL', 'SL', 'DD']:
            per_lang_lab_dens = dict()
            per_lang_lab_sums = dict()
            per_level_lab_dens = defaultdict(int)
            per_level_lab_sums = defaultdict(int)
            for language in languages:
                for label in sorted(counter):
                    if not label.endswith(language):
                        continue
                    per_lang_lab_dens['EFCamDat2_G' + label[1:]] = sum([value for value in counter[label].values()])
                    per_lang_lab_sums['EFCamDat2_G' + label[1:]] = sum(
                        [value * int(key[key.index('_') + 1:]) for key, value in counter[label].items()])
                    per_level_lab_dens['EFCamDat2_G' + label[1:2]] += sum([value for value in counter[label].values()])
                    per_level_lab_sums['EFCamDat2_G' + label[1:2]] += sum(
                        [value * int(key[key.index('_') + 1:]) for key, value in counter[label].items()])

            print('************')
            for label in per_lang_lab_sums:
                print(feat, label, per_lang_lab_sums[label], per_lang_lab_dens[label],
                      per_lang_lab_sums[label] / per_lang_lab_dens[label], sep='\t')

            print('************')
            for label in per_level_lab_sums:
                print(feat, label, per_level_lab_sums[label], per_level_lab_dens[label],
                      per_level_lab_sums[label] / per_level_lab_dens[label], sep='\t')

        plt.tight_layout()
        monochrome = (cycler('color', ['k']) * cycler('marker', ['', '^', ',', '.']) * cycler('linestyle',
                                                                                              ['-', '--', ':', '-.']))
        plt.rcParams['axes.prop_cycle'] = monochrome
        plt.cycler(monochrome)
        for level in dX:
            avg = [0.01] * len(labels)
            for label in per_lang_lab_vals:
                if label.startswith(level):
                    series = per_lang_lab_vals[label]
                    avg = [a + b for a, b in zip(avg, series)]
            avg = [a / (len(per_lang_lab_vals) / len(dX)) for a in avg]
            plt.semilogy(labels, avg, label='EFCamDat2_G' + level[1:])
        plt.legend(loc='upper right')
        if feat == 'WL':
            plt.title('Word Length (characters)')
        elif feat == 'SL':
            plt.title('Sentence Length (words)')
        elif feat == 'DD':
            plt.title('Depencency Depth (distance from root)')
        plt.ylabel('Normalized distribution (log scale)')
        plt.xticks(list(np.arange(0, max_val + 1, max_val / 5)))
        plt.savefig(os.path.join(stats_dir, 'efcamdat_123_' + feat + '_AVG_bw.pdf'))
        plt.close()


if __name__ == '__main__':
    feture_stats()
