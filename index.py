import os
import pickle
from collections import defaultdict

from tqdm.auto import tqdm

from custom_tokenization import feature_types


def index(dataset):
    print('indexing', dataset)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
        y = pickle.load(inputfile)

        print(len(X),len(y))

    from custom_tokenization import spacy_tokenizer

    for feature_type in feature_types:
        print(f'\t{feature_type}')
        for text in tqdm(X):
            X_indexed =spacy_tokenizer(text, feature_type)
            with open(os.path.join('data', dataset + '_indexed_' + feature_type + '.pkl'), mode='wb') as outputfile:
                pickle.dump(X_indexed, outputfile)
    print('indexed', dataset)


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
        index(dataset)
