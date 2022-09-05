import os
import pickle


def explore(dataset):
    print('explore', dataset)
    with open(os.path.join('data', dataset + '.pkl'), mode='rb') as inputfile:
        pickle.load(inputfile)
        y = pickle.load(inputfile)

    with open(os.path.join('data', dataset + '_indexed.pkl'), mode='rb') as inputfile:
        X = pickle.load(inputfile)

    output_dir = os.path.join('explore_features')

    os.makedirs(output_dir, exist_ok=True)

    top_to_explore = 10
    cases_to_find = 10
    window = 5

    svm_feats_path = 'svm_feats'
    filename = 'spacy_' + dataset + '_svm.txt'
    with open(os.path.join(svm_feats_path, filename), mode='r', encoding='utf-8') as inputfile, \
            open(os.path.join(output_dir, filename), mode='w', encoding='utf-8') as outputfile:
        for line in inputfile:
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
                for doc, label in zip(X, y):
                    if (label == lang and rel == 'yes') or (label != lang and rel == 'no'):
                        idxs = [i for i, x in enumerate(doc[feat_type]) if x == feat]
                        for idx in idxs:
                            print(fields[0], lang, rel, rank + 1, feat_type, feat, weight, sep='\t', end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:] for token in doc['T1'][idx - window:idx]]), end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:].upper() for token in doc['T1'][idx:idx + ngram]]), end='\t',
                                  file=outputfile)
                            print('\t'.join([token[3:] for token in doc['T1'][idx + ngram:idx + window + ngram]]),
                                  end='\t', file=outputfile)
                            print('', file=outputfile)
                            found += 1
                            if found == cases_to_find:
                                break

                    if found == cases_to_find:
                        break

    print('explored', dataset)


if __name__ == '__main__':
    for dataset in ['toefl11', 'EFCAMDAT2', 'reddit500k']:
        explore(dataset)
