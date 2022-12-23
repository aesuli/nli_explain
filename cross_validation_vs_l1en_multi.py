import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from cross_validation import MIN_DF
from custom_tokenization import dummy_tokenizer
from learn_vs_l1en_multi import EnsembleNativeNonNative


def kfold(texts, labels, pipeline, native_label, folds=10, seed=0):
    """
    Performs a cross-validation experiment with a dataset of labeled documents
    :param texts: list of documents
    :param labels: list of labels assigned to documents
    :param pipeline: scikit-learn pipeline
    :param folds: number of fold to split the dataset into
    :param seed: seed for random number generator
    :return:
    """
    # generator of cross-validation folds
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_idx = 0
    cv_true_labels = list()
    cv_predicted_labels = list()
    # for each fold
    bin_labels = np.asarray(labels)
    print(set(bin_labels))
    bin_labels[bin_labels != native_label] = 'nn'
    print(set(bin_labels))
    for traincv, testcv in cv.split(np.zeros(len(labels)), list(bin_labels)):
        fold_idx += 1
        print('fold %d/%d' % (fold_idx, folds))

        # fold construction
        training_texts = [texts[i] for i in traincv]
        training_labels = [labels[i] for i in traincv]
        test_texts = [texts[i] for i in testcv]
        test_labels = [labels[i] for i in testcv]
        cv_true_labels.extend(test_labels)

        # learning
        model = pipeline.fit(training_texts, training_labels)

        # classification
        predicted_labels = model.predict(test_texts)
        cv_predicted_labels.extend(predicted_labels)

    bin_cv_true_labels = np.asarray(cv_true_labels)
    bin_cv_true_labels[bin_cv_true_labels != native_label] = 'nn'
    cm = confusion_matrix(bin_cv_true_labels, cv_predicted_labels, labels=pipeline.named_steps['class'].classes_)

    output = 'Confusion matrix\n'

    for row, label in enumerate(pipeline.named_steps['class'].classes_):
        output += label + '     ' + str(cm[row]) + '\n'

    output += '\n\n'

    output += classification_report(bin_cv_true_labels, cv_predicted_labels, digits=3,
                                    target_names=pipeline.named_steps['class'].classes_)

    print(output)
    return output


def cv(dataset, dataset_en, native_label):
    print('cv', dataset)
    with open(os.path.join('data', dataset_en + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y_en = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    ratio = len(y) // len(y_en)
    y = y[::ratio]
    print(len(y), len(y_en), set(y))

    to_test = [
        ['T1'],  # solo token
        ['PHO3', 'PHO2', 'PHO1', 'PH3', 'PH2', 'PH1', 'PHO', 'T1'],  # tutte le features fonetiche + token
        ['PHO3', 'PHO2', 'PHO1', 'PH3', 'PH2', 'PH1', 'PHO'],  # tutte le feature fonetiche
        ['PHO3', 'T1'],  # token + trigrams fonetici
        ['T1', 'PHO'],  # token + parola fonetica
        ['PH3', 'PH2', 'PH1'],
        ['PHO3', 'PHO2', 'PHO1'],
        ['PHO3', 'PH3'],  # trigrams con e senza spazio
        ['PHO2', 'PH2'],  # bigrams con e senza spazio
        ['PH1'],  # solo unigrams senza spazi
        ['PHO'],  # solo parola fonetica
        ['PHO3', 'PHO2', 'PHO1', 'PHO', 'PH3', 'PH2', 'PH1', 'T1', 'T2', 'T3'],
        # tutte le features fonetiche + ngrams token
        # ['T2'],
        # ['T3'],
        # ['L1'],
        # ['L2'],
        # ['L3'],
        # ['Tn1'],
        # ['Tn2'],
        # ['Tn3'],
        # ['Ln1'],
        # ['Ln2'],
        # ['Ln3'],
        # ['Tp1'],
        # ['Tp2'],
        # ['Tp3'],
        # ['Lp1'],
        # ['Lp2'],
        # ['Lp3'],
        # ['Ms1'],
        # ['Ms2'],
        # ['Ms3'],
        # ['P1'],
        # ['P2'],
        # ['P3'],
        # ['D1'],
        # ['D2'],
        # ['D3'],
        # ['WL'],
        # ['SL'],
        # ['DD'],
        # ['T1', 'T2', 'T3'],
        # ['Tn1', 'Tn2', 'Tn3'],
        # ['Tp1', 'Tp2', 'Tp3'],
        # ['L1', 'L2', 'L3'],
        # ['Ln1', 'Ln2', 'Ln3'],
        # ['Lp1', 'Lp2', 'Lp3'],
        # ['P1', 'P2', 'P3'],
        # ['D1', 'D2', 'D3'],
        # ['Ms1', 'Ms2', 'Ms3'],
        # ['WL', 'SL', 'DD'],
        # ['T1', 'T2', 'T3', 'Tn1', 'Tn2', 'Tn3', 'L1', 'L2', 'L3', 'Ln1', 'Ln2', 'Ln3', 'P1', 'P2', 'P3', 'D1', 'D2',
        #  'D3', 'WL', 'SL', 'DD', 'Tp1', 'Tp2', 'Tp3', 'Lp1', 'Lp2', 'Lp3', 'Ms1', 'Ms2', 'Ms3'],
    ]

    pathdir = 'kfold_res_vs_l1en_multi'
    os.makedirs(pathdir, exist_ok=True)

    for feat_mask in to_test:
        mask = ''
        X = None
        for feat_type in feat_mask:
            mask += '_' + feat_type
            with open(os.path.join('data', dataset + '_indexed_' + feat_type + '.pkl'), mode='rb') as inputfile:
                feat_X = pickle.load(inputfile)[::ratio]
                if X is None:
                    X = feat_X
                else:
                    if len(X) != len(feat_X):
                        raise Exception(
                            f'Mismatch in the number of indexed documents {len(X)}!={len(feat_X)} ({feat_type})')
                    for vect, feat_vect in zip(X, feat_X):
                        vect.extend(feat_vect)

        X_en = None
        for feat_type in feat_mask:
            with open(os.path.join('data', dataset_en + '_indexed_' + feat_type + '.pkl'), mode='rb') as inputfile:
                feat_X = pickle.load(inputfile)
                if X_en is None:
                    X_en = feat_X
                else:
                    if len(X_en) != len(feat_X):
                        raise Exception(
                            f'Mismatch in the number of indexed documents {len(X_en)}!={len(feat_X)} ({feat_type})')
                    for vect, feat_vect in zip(X_en, feat_X):
                        vect.extend(feat_vect)

        if len(mask) == 0:
            continue

        with open(os.path.join(pathdir, dataset + '_' + mask + '_vs_l1en_kfold_multi.txt'), mode='w',
                  encoding='utf-8') as outputfile:
            X_vs = list(X)
            y_vs = list(y)
            X_vs.extend(X_en)
            y_vs.extend(y_en)

            print(dataset + '_' + mask + '_' + '_vs_l1en_kfold_multi')

            pipeline = Pipeline([
                ('vect', CountVectorizer(analyzer=dummy_tokenizer, lowercase=False, min_df=MIN_DF)),
                # ('select', SelectPercentile(chi2, percentile=50)),
                ('weight', TfidfTransformer()),
                ('class', EnsembleNativeNonNative(native_label))
            ])

            print(dataset + '_' + mask + '_' + '_vs_l1en_kfold_multi', file=outputfile)
            print(kfold(X_vs, y_vs, pipeline, native_label), file=outputfile)
            print('----------------------', file=outputfile)
    print('cv\'ed', dataset)


if __name__ == '__main__':
    for dataset in [
        'reddit',
        # 'toefl11',
        # 'EFCAMDAT2',
        # 'EFCAMDAT2_L1',
        # 'EFCAMDAT2_L2',
        # 'EFCAMDAT2_L3',
        'openaire_en_nonnative'
    ]:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            dataset_en = 'LOCNESS'
        elif dataset == 'openaire_en_nonnative':
            dataset_en = 'openaire_en_native'
            native_label = 'en'
        else:
            dataset_en = 'redditEN'
            native_label = 'UK'

        cv(dataset, dataset_en, native_label)
