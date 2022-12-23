import os
import pickle

from learn_vs_l1en_multi import EnsembleNativeNonNative


def inspect(dataset, native_label):
    print('inspect', dataset)
    model_dir = 'models_vs_l1en_multi'

    to_test = list()
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue

        if not filename.startswith(dataset + '_' + dataset + '_multi'):
            continue

        to_test.append(filename)

    multi_dir = 'multi_feats_vs_l1en'
    os.makedirs(multi_dir, exist_ok=True)
    outputfile = open(os.path.join(multi_dir, 'spacy_' + dataset + '_multi.txt'), mode='w',
                      encoding='utf-8')

    for filename in sorted(to_test, key=lambda x: f'{len(x):3}' + x):
        print(filename)

        with open(os.path.join(model_dir, filename), mode='rb') as inputfile:
            pipeline = pickle.load(inputfile)

        tokenizer = pipeline.named_steps['vect']
        # selector = pipeline.named_steps['select']
        classifier:EnsembleNativeNonNative = pipeline.named_steps['class']

        feature_names = tokenizer.get_feature_names_out()

        # feats_w_score = list()
        # for index, (selected, score) in enumerate(zip(selector.get_support(), selector.scores_)):
        #     feats_w_score.append((score, selected, feature_names[index]))
        #
        # print(sorted(feats_w_score, reverse=True)[:100])

        count = 50

        label = classifier.classes_[0]
        feats_w_classifier_weight = list()
        # for index, weight in enumerate(selector.inverse_transform(classifier.coef_[i].reshape(1, -1))[0]):
        for index, weight in enumerate(classifier.estimator_m.coef_[0]):
            if weight != 0:
                feats_w_classifier_weight.append((weight, feature_names[index]))

        if label == native_label:
            print(filename, native_label, 'yes',
                  '\t'.join(
                      [f'{label}\t{-value:3.3}' for value, label in
                       sorted(feats_w_classifier_weight)[:count]]),
                  file=outputfile, sep='\t')

            print(filename, native_label, 'no',
                  '\t'.join(
                      [f'{label}\t{-value:3.3}' for value, label in
                       sorted(feats_w_classifier_weight, reverse=True)[:count]]),
                  file=outputfile, sep='\t')
        else:
            print(filename, native_label, 'yes',
                  '\t'.join(
                      [f'{label}\t{value:3.3}' for value, label in
                       sorted(feats_w_classifier_weight, reverse=True)[:count]]),
                  file=outputfile, sep='\t')

            print(filename, native_label, 'no',
                  '\t'.join(
                      [f'{label}\t{value:3.3}' for value, label in sorted(feats_w_classifier_weight)[:count]]),
                  file=outputfile, sep='\t')

    if outputfile:
        outputfile.close()
    print('inspected', dataset)


if __name__ == '__main__':
    for dataset in [
        # 'toefl11',
        'reddit',
        # 'EFCAMDAT2',
        # 'EFCAMDAT2_L1',
        # 'EFCAMDAT2_L2',
        # 'EFCAMDAT2_L3',
        'openaire_en_nonnative'
    ]:
        if dataset in {'toefl11', 'EFCAMDAT2', 'EFCAMDAT2_L1', 'EFCAMDAT2_L2', 'EFCAMDAT2_L3'}:
            native_label = None
        elif dataset == 'openaire_en_nonnative':
            native_label = 'en'
        else:
            native_label = 'UK'

        inspect(dataset, native_label)
