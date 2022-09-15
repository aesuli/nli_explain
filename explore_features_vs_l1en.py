import os
import pickle

from tqdm.auto import tqdm

from custom_tokenization import feature_types


def explore(dataset, dataset_en):
    print('explore', dataset, dataset_en)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    X = dict()
    for feature_type in feature_types:
        with open(os.path.join('data', dataset + '_indexed_'+feature_type+'.pkl'), mode='rb') as inputfile:
            X[feature_type] = pickle.load(inputfile)

    with open(os.path.join('data', dataset_en + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y_en = pickle.load(inputfile)

    X_en = dict()
    for feature_type in feature_types:
        with open(os.path.join('data', dataset_en + '_indexed_'+feature_type+'.pkl'), mode='rb') as inputfile:
            X_en[feature_type] = pickle.load(inputfile)

    output_dir = os.path.join('explore_vs_l1en')

    os.makedirs(output_dir, exist_ok=True)

    top_to_explore = 10
    cases_to_find = 10
    window = 5

    svm_feats_path = 'svm_feats_vs_l1en'
    filename = 'spacy_' + dataset + '_svm.txt'
    with open(os.path.join(svm_feats_path, filename), mode='r', encoding='utf-8') as inputfile:
        lines = inputfile.readlines()
    with open(os.path.join(output_dir, filename), mode='w', encoding='utf-8') as outputfile:
        for line in tqdm(lines):
            fields = line.split('\t')
            lang = fields[1]
            feats_w_weights = list(zip(fields[3::2], [float(f) for f in fields[4::2]]))
            for rank, (feat, weight) in enumerate(feats_w_weights[:top_to_explore]):
                feat_type = feat.split('_')[0]

                if weight > 0:
                    rel = 'yes'
                else:
                    rel = 'no'

                if feat_type in ['DD', 'SL', 'WL']:
                    continue
                ngram = int(feat_type[-1])

                found = 0
                if rel=='yes':
                    for doc,doc_feat, label in zip(X['T1'],X[feat_type], y):
                        if label == lang:
                            idxs = [i for i, x in enumerate(doc_feat) if x == feat]
                            for idx in idxs:
                                print(fields[0], lang, rel, rank + 1, feat_type, feat, weight, sep='\t', end='\t',
                                      file=outputfile)
                                print('\t'.join([token[3:] for token in doc[idx - window:idx]]), end='\t',
                                      file=outputfile)
                                print('\t'.join([token[3:].upper() for token in doc[idx:idx + ngram]]), end='\t',
                                      file=outputfile)
                                print('\t'.join([token[3:] for token in doc[idx + ngram:idx + window + ngram]]),
                                      end='\t', file=outputfile)
                                print('', file=outputfile)
                                found += 1
                                if found == cases_to_find:
                                    break

                        if found == cases_to_find:
                            break
                else:
                    for doc,doc_feat in zip(X_en['T1'],X[feat_type]):
                        idxs = [i for i, x in enumerate(doc_feat) if x == feat]
                        for idx in idxs:
                            print(fields[0], lang, rel, rank + 1, feat_type, feat, weight, sep='\t', end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:] for token in doc[idx - window:idx]]), end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:].upper() for token in doc[idx:idx + ngram]]), end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:] for token in doc[idx + ngram:idx + window + ngram]]),
                                  end='\t', file=outputfile)
                            print('', file=outputfile)
                            found += 1
                            if found == cases_to_find:
                                break

                        if found == cases_to_find:
                            break

    print('explored', dataset, dataset_en)


if __name__ == '__main__':
    for dataset in ['toefl11', 'EFCAMDAT2', 'reddit',
                    'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3']:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            dataset_en = 'LOCNESS'
        else:
            dataset_en = 'redditEN'
        explore(dataset, dataset_en)
