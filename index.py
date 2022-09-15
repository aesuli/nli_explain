import os
import pickle
from collections import defaultdict

from tqdm.auto import tqdm


def index(dataset):
    print('indexing', dataset)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
        y = pickle.load(inputfile)

        print(len(X),len(y))

    from custom_tokenization import spacy_tokenizer

    indexed = defaultdict(list)
    for text in tqdm(X):
        features_dict =spacy_tokenizer(text)
        for feature_type in features_dict:
            indexed[feature_type].append(features_dict[feature_type])

    for feature_type in indexed:
        with open(os.path.join('data', dataset + '_indexed_' + feature_type + '.pkl'), mode='wb') as outputfile:
            pickle.dump(indexed[feature_type], outputfile)
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
