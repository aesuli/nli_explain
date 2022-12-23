import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from cross_validation import MIN_DF
from custom_tokenization import dummy_tokenizer


def learn(dataset, dataset_en, algo):
    print('learn', dataset, dataset_en, algo)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset_en + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y_en = pickle.load(inputfile)

    model_dir = os.path.join('models_vs_l1en', algo)

    os.makedirs(model_dir, exist_ok=True)

    to_test = [
        ['T1'],
        ['T2'],
        ['T3'],
        ['Tn1'],
        ['Tn2'],
        ['Tn3'],
        ['L1'],
        ['L2'],
        ['L3'],
        ['Ln1'],
        ['Ln2'],
        ['Ln3'],
        ['P1'],
        ['P2'],
        ['P3'],
        ['D1'],
        ['D2'],
        ['D3'],
        ['Tp1'],
        ['Tp2'],
        ['Tp3'],
        ['Lp1'],
        ['Lp2'],
        ['Lp3'],
        ['Ms1'],
        ['Ms2'],
        ['Ms3'],
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
        pipelines = dict()
        for lang in set(y):

            X_vs = list()
            y_vs = list()
            for doc, label in zip(X, y):
                if label == lang:
                    X_vs.append(doc)
                    y_vs.append(label)
            X_vs.extend(X_en)
            y_vs.extend(y_en)

            print(dataset + '_' + lang + '_' + algo + mask)
            if algo == 'svm':
                learner = LinearSVC()
            elif algo == 'dt':
                learner = DecisionTreeClassifier(max_depth=3)

            pipeline = Pipeline([
                ('vect', CountVectorizer(analyzer=dummy_tokenizer, lowercase=False, min_df=MIN_DF)),
                # ('select', SelectPercentile(chi2, percentile=50)),
                ('weight', TfidfTransformer()),
                ('class', learner)
            ])
            pipeline.fit(X_vs, y_vs)
            pipelines[lang] = pipeline

        with open(os.path.join(model_dir, dataset + '_' + dataset + '_' + algo + '_model_spacy' + mask + '.pkl'),
                  mode='wb') as outputfile:
            pickle.dump(pipelines, outputfile)
    print('learned', dataset, dataset_en, algo)


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'EFCAMDAT2',
        'reddit',
        'EFCAMDAT2_L1',
        'EFCAMDAT2_L2',
        'EFCAMDAT2_L3',
        'openaire_en_nonnative',
    ]:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            dataset_en = 'LOCNESS'
        elif dataset == 'openaire_en_nonnative':
            dataset_en = 'openaire_en_native'
        else:
            dataset_en = 'redditEN'
        for algo in ['svm']:
            # 'dt']:
            learn(dataset, dataset_en, algo)
