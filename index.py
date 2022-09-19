import os
import pickle

from tqdm.auto import tqdm

from custom_tokenization import feature_types


def index(dataset, feature_type):
    print('indexing', dataset, feature_type)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
        y = pickle.load(inputfile)

        print(len(X), len(y))

    from custom_tokenization import spacy_tokenizer

    X_indexed = list()
    for text in tqdm(X):
        X_indexed.append(spacy_tokenizer(text, feature_type))
    with open(os.path.join('data', dataset + '_indexed_' + feature_type + '.pkl'), mode='wb') as outputfile:
        pickle.dump(X_indexed, outputfile)


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
        for feature_type in feature_types:
            index(dataset, feature_type)
