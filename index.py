import os
import pickle

import spacy
from tqdm.auto import tqdm

from custom_tokenization import feature_types


def index(dataset, feature_types):
    print('indexing', dataset)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
        y = pickle.load(inputfile)

        print(len(X), len(y))

    from custom_tokenization import spacy_tokenizer

    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    X_docs = list()
    for text in tqdm(X):
        X_docs.append(nlp(text))

    for feature_type in feature_types:
        print(f'\t{feature_type}')
        X_indexed = list()
        for doc in tqdm(X_docs):
            X_indexed.append(spacy_tokenizer(doc, feature_type))
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
        index(dataset, feature_types)
