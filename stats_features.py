import os
import pickle
from collections import defaultdict

from custom_tokenization import feature_types


def load_dataset(dataset):
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        text = pickle.load(inputfile)
        y = pickle.load(inputfile)

    X = dict()
    for feature_type in feature_types:
        with open(os.path.join('data', dataset + '_indexed_' + feature_type + '.pkl'), mode='rb') as inputfile:
            X[feature_type] = pickle.load(inputfile)
    return text, X, y


if __name__ == '__main__':
    for source_datasets in [
        ['toefl11'],
        ['reddit'],
        ['EFCAMDAT2'],
        ['LOCNESS'],
        ['redditEN'],
        ['toefl11', 'LOCNESS'],
        ['EFCAMDAT2', 'LOCNESS'],
        ['reddit', 'redditEN'],
    ]:
        print(source_datasets)
        feat_sets = defaultdict(set)
        for dataset in source_datasets:
            text, X, y = load_dataset(dataset)
            for feature_type in X:
                for features in X[feature_type]:
                    feat_sets[feature_type].update(features)
        for feature_type, distinct_features in feat_sets.items():
            print(feature_type, len(distinct_features))
