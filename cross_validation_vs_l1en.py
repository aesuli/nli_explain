import os
import pickle
from functools import partial

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from custom_tokenization import list_tokenizer


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


def cv(dataset, dataset_en, algo, only_stats=False):
    print('cv', dataset, algo)
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

    if only_stats:
        pathdir = 'kfold_stats_vs_l1en'
    else:
        pathdir = 'kfold_res_vs_l1en'
    os.makedirs(pathdir, exist_ok=True)

    with open(os.path.join(pathdir, dataset + '_' + algo + 'vs_l1en_kfold.txt'), mode='w', encoding='utf-8') as outputfile:
        for feat_mask in to_test:
            mask = ''
            for feat_type in feat_mask:
                mask += '_' + feat_type

            if len(mask) == 0:
                continue

            X_vs = list()
            y_vs = list()
            for doc, label in zip(X, y):
                X_vs.append(doc)
                y_vs.append(label)
            X_vs.extend(X_en)
            y_vs.extend(y_en)

            print(dataset + '_bin_' + algo + mask)
            if algo == 'svm':
                learner = LinearSVC()
            elif algo == 'dt':
                learner = DecisionTreeClassifier()
            elif algo == 'dtm':
                learner = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3))

            if only_stats:
                pipeline = Pipeline([
                    ('vect', CountVectorizer(analyzer=partial(list_tokenizer, feat_mask), lowercase=False, min_df=2)),
                    # ('select', SelectPercentile(chi2, percentile=50)),
                    ('weight', TfidfTransformer()),
                ])

                print(dataset + '_' + algo + mask, file=outputfile)
                Xt = pipeline.fit_transform(X_vs,y_vs)
                print(Xt.shape, file=outputfile)
                print('----------------------', file=outputfile)
            else:
                pipeline = Pipeline([
                    ('vect', CountVectorizer(analyzer=partial(list_tokenizer, feat_mask), lowercase=False, min_df=2)),
                    # ('select', SelectPercentile(chi2, percentile=50)),
                    ('weight', TfidfTransformer()),
                    ('class', learner)
                ])

                print(dataset + '_bin_' + algo + mask, file=outputfile)
                print(kfold(X_vs, y_vs, pipeline), file=outputfile)
                print('----------------------', file=outputfile)
    print('cv\'ed', dataset, algo)


if __name__ == '__main__':
    for dataset in [
        'reddit500k',
        'toefl11',
        'EFCAMDAT2',
        # 'EFCAMDAT2_L1',
        # 'EFCAMDAT2_L2',
        # 'EFCAMDAT2_L3'
    ]:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            dataset_en = 'LOCNESS'
        else:
            dataset_en = 'reddit500kEN'

        only_stats = True
        # only_stats = False

        for algo in ['svm']:  # , 'dt', 'dtm']:
            cv(dataset, dataset_en, algo, only_stats=only_stats)
