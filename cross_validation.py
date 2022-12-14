import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from custom_tokenization import dummy_tokenizer

MIN_DF = 10

def kfold(texts, labels, pipeline, folds=10, seed=0):
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
    for traincv, testcv in cv.split(np.zeros(len(labels)), labels):
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

    cm = confusion_matrix(cv_true_labels, cv_predicted_labels, labels=pipeline.named_steps['class'].classes_)

    output = 'Confusion matrix\n'

    for row, label in enumerate(pipeline.named_steps['class'].classes_):
        output += label + '     ' + str(cm[row]) + '\n'

    output += '\n\n'

    output += classification_report(cv_true_labels, cv_predicted_labels, digits=3,
                                    target_names=pipeline.named_steps['class'].classes_)

    print(output)
    return output


def cv(dataset, algo, only_indexing=False):
    print('cv', dataset, algo)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    to_test = [
        ['T1'],
        ['T2'],
        ['T3'],
        ['L1'],
        ['L2'],
        ['L3'],
        ['Tn1'],
        ['Tn2'],
        ['Tn3'],
        ['Ln1'],
        ['Ln2'],
        ['Ln3'],
        ['Tp1'],
        ['Tp2'],
        ['Tp3'],
        ['Lp1'],
        ['Lp2'],
        ['Lp3'],
        ['Ms1'],
        ['Ms2'],
        ['Ms3'],
        ['P1'],
        ['P2'],
        ['P3'],
        ['D1'],
        ['D2'],
        ['D3'],
        ['WL'],
        ['SL'],
        ['DD'],
        ['T1', 'T2', 'T3'],
        ['Tn1', 'Tn2', 'Tn3'],
        ['Tp1', 'Tp2', 'Tp3'],
        ['L1', 'L2', 'L3'],
        ['Ln1', 'Ln2', 'Ln3'],
        ['Lp1', 'Lp2', 'Lp3'],
        ['P1', 'P2', 'P3'],
        ['D1', 'D2', 'D3'],
        ['Ms1', 'Ms2', 'Ms3'],
        ['WL', 'SL', 'DD'],
        ['T1', 'T2', 'T3', 'Tn1', 'Tn2', 'Tn3', 'L1', 'L2', 'L3', 'Ln1', 'Ln2', 'Ln3', 'P1', 'P2', 'P3', 'D1', 'D2',
         'D3', 'WL', 'SL', 'DD', 'Tp1', 'Tp2', 'Tp3', 'Lp1', 'Lp2', 'Lp3', 'Ms1', 'Ms2', 'Ms3'],
    ]

    if only_indexing:
        kfold_dir = 'kfold_stats'
    else:
        kfold_dir = 'kfold_res'
    os.makedirs(kfold_dir, exist_ok=True)

    with open(os.path.join(kfold_dir, dataset + '_' + algo + '_kfold.txt'), mode='w', encoding='utf-8') as outputfile:
        for feat_mask in to_test:
            mask = ''
            X = None
            for feat_type in feat_mask:
                mask += '_' + feat_type
                with open(os.path.join('data', dataset + '_indexed_' + feat_type + '.pkl'), mode='rb') as inputfile:
                    feat_X = pickle.load(inputfile)
                    if X is None:
                        X = feat_X
                    else:
                        if len(X) != len(feat_X):
                            raise Exception(f'Mismatch in the number of indexed documents {len(X)}!={len(feat_X)}')
                        for vect, feat_vect in zip(X, feat_X):
                            vect.extend(feat_vect)

            if len(mask) == 0:
                continue

            print(dataset + '_' + algo + mask)
            if algo == 'svm':
                learner = LinearSVC()
            elif algo == 'dt':
                learner = DecisionTreeClassifier()
            elif algo == 'dtm':
                learner = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3))

            if only_indexing:
                pipeline = Pipeline([
                    ('vect', CountVectorizer(analyzer=dummy_tokenizer, lowercase=False, min_df=MIN_DF)),
                    # ('select', SelectPercentile(chi2, percentile=50)),
                    ('weight', TfidfTransformer()),
                ])

                print(dataset + '_' + algo + mask, file=outputfile)
                Xt = pipeline.fit_transform(X, y)
                print(Xt.shape, file=outputfile)
                print('----------------------', file=outputfile)
            else:
                pipeline = Pipeline([
                    ('vect', CountVectorizer(analyzer=dummy_tokenizer, lowercase=False, min_df=MIN_DF)),
                    # ('select', SelectPercentile(chi2, percentile=50)),
                    ('weight', TfidfTransformer()),
                    ('class', learner)
                ])

                print(dataset + '_' + algo + mask, file=outputfile)
                print(kfold(X, y, pipeline), file=outputfile)
                print('----------------------', file=outputfile)
    print('cv\'ed', dataset, algo)


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'EFCAMDAT2',
        'reddit',
        'openaire_en_nonnative',
    ]:
        # only_indexing = True
        only_indexing = False

        for algo in ['svm']:  # , 'dt', 'dtm']:
            cv(dataset, algo, only_indexing=only_indexing)
