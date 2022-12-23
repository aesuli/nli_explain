import os
import pickle
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from custom_tokenization import dummy_tokenizer

class EnsembleNativeNonNative(BaseEstimator):

    def __init__(self, native_label ,estimator_m=LogisticRegressionCV(multi_class='multinomial',n_jobs=-1), estimator_b=LogisticRegressionCV(multi_class='multinomial', n_jobs=-1)):
        self._native_label = native_label
        self._nn_label = None
        self.estimator_m = estimator_m
        self.estimator_b = estimator_b

    def fit(self, X, y=None, **kwargs):
        self.estimator_m.fit(X, y)
        m_probs = self.estimator_m.predict_proba(X)
        b_y = np.asarray(y)
        b_y[b_y!=self._native_label]='nn'
        self.estimator_b.fit(m_probs, list(b_y))
        return self

    def predict(self, X, y=None):
        m_probs = self.estimator_m.predict_proba(X)
        return self.estimator_b.predict(m_probs)

    def predict_proba(self, X):
        m_probs = self.estimator_m.predict_proba(X)
        return self.estimator_b.predict_proba(m_probs)

    def score(self, X, y):
        m_probs = self.estimator_m.predict_proba(X)
        b_y = y.copy()
        b_y[b_y!='en']='nn'
        return self.estimator_b.score(m_probs, b_y)

    @property
    def classes_(self):
        return self.estimator_b.classes_




def learn(dataset, dataset_en, native_label):
    print('learn', dataset, dataset_en, 'multi')
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset_en + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y_en = pickle.load(inputfile)

    model_dir = 'models_vs_l1en_multi'

    os.makedirs(model_dir, exist_ok=True)

    to_test = [
        ['T1'], #solo token
        ['PHO3', 'PHO2', 'PHO1', 'PH3', 'PH2', 'PH1', 'PHO', 'T1'], #tutte le features fonetiche + token
        ['PHO3', 'PHO2', 'PHO1', 'PH3', 'PH2', 'PH1', 'PHO'], #tutte le feature fonetiche
        ['PHO3', 'T1'], #token + trigrams fonetici
        ['T1', 'PHO'], #token + parola fonetica
        ['PH3','PH2','PH1'],
        ['PHO3','PHO2','PHO1'],
        ['PHO3', 'PH3'], # trigrams con e senza spazio
        ['PHO2', 'PH2'], # bigrams con e senza spazio
        ['PH1'], # solo unigrams senza spazi
        ['PHO'], # solo parola fonetica
        ['PHO3', 'PHO2', 'PHO1', 'PHO', 'PH3', 'PH2', 'PH1', 'T1', 'T2', 'T3'], #tutte le features fonetiche + ngrams token
        # ['T2'],
        # ['T3'],
        # ['Tn1'],
        # ['Tn2'],
        # ['Tn3'],
        # ['L1'],
        # ['L2'],
        # ['L3'],
        # ['Ln1'],
        # ['Ln2'],
        # ['Ln3'],
        # ['P1'],
        # ['P2'],
        # ['P3'],
        # ['D1'],
        # ['D2'],
        # ['D3'],
        # ['Tp1'],
        # ['Tp2'],
        # ['Tp3'],
        # ['Lp1'],
        # ['Lp2'],
        # ['Lp3'],
        # ['Ms1'],
        # ['Ms2'],
        # ['Ms3'],
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

        X_vs = list(X)
        y_vs = list(y)
        X_vs.extend(X_en)
        y_vs.extend(y_en)

        print(dataset + '_multi' + mask)

        pipeline = Pipeline([
            ('vect', CountVectorizer(analyzer=dummy_tokenizer, lowercase=False, min_df=2)),
            # ('select', SelectPercentile(chi2, percentile=50)),
            ('weight', TfidfTransformer()),
            ('class', EnsembleNativeNonNative(native_label))
        ])
        pipeline.fit(X_vs, y_vs)
        
        with open(os.path.join(model_dir, dataset + '_' + dataset + '_multi' + '_model_spacy' + mask + '.pkl'),
                  mode='wb') as outputfile:
            pickle.dump(pipeline, outputfile)
    print('learned', dataset, dataset_en, 'multi')


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

        learn(dataset, dataset_en, native_label)
