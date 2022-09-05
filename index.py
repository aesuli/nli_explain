import os
import pickle

from tqdm.auto import tqdm


def index(dataset):
    print('indexing', dataset)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)
        y = pickle.load(inputfile)

        print(len(X),len(y))

    from custom_tokenization import spacy_tokenizer

    indexed = list()
    for text in tqdm(X):
        indexed.append(spacy_tokenizer(text))

    with open(os.path.join('data', dataset + '_indexed.pkl'), mode='wb') as outputfile:
        pickle.dump(indexed, outputfile)
    print('indexed', dataset)


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'reddit500k',
        'EFCAMDAT2',
        'LOCNESS',
        'reddit500kEN',
        'EFCAMDAT2_L1',
        'EFCAMDAT2_L2',
        'EFCAMDAT2_L3'
    ]:
        index(dataset)
