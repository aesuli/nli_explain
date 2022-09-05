import os
import pickle

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))


def prune_duplicate_leaves(mdl):
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)


def inspect(dataset, algo):
    print('inspect', dataset, algo)
    model_dir = os.path.join('models_vs_l1en', algo)

    to_test = list()
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue

        if not filename.startswith(dataset + '_' + dataset + '_' + algo):
            continue

        to_test.append(filename)

    outputfile = None
    if algo == 'svm':
        svm_dir = 'svm_feats_vs_l1en'
        os.makedirs(svm_dir, exist_ok=True)
        outputfile = open(os.path.join(svm_dir, 'spacy_' + dataset + '_' + algo + '.txt'), mode='w',
                          encoding='utf-8')
    elif algo == 'dt':
        plot_dir = os.path.join('plots_vs_l1en', algo)
        os.makedirs(plot_dir, exist_ok=True)

    # plot_format = '.svg'
    plot_format = '.png'

    for filename in sorted(to_test, key=lambda x: f'{len(x):3}' + x):
        print(filename)

        with open(os.path.join(model_dir, filename), mode='rb') as inputfile:
            pipelines = pickle.load(inputfile)

        for lang in pipelines:
            pipeline = pipelines[lang]

            tokenizer = pipeline.named_steps['vect']
            # selector = pipeline.named_steps['select']
            classifier = pipeline.named_steps['class']

            feature_names = tokenizer.get_feature_names()

            # feats_w_score = list()
            # for index, (selected, score) in enumerate(zip(selector.get_support(), selector.scores_)):
            #     feats_w_score.append((score, selected, feature_names[index]))
            #
            # print(sorted(feats_w_score, reverse=True)[:100])

            if algo == 'svm':
                count = 50

                label = classifier.classes_[0]
                feats_w_classifier_weight = list()
                # for index, weight in enumerate(selector.inverse_transform(classifier.coef_[i].reshape(1, -1))[0]):
                for index, weight in enumerate(classifier.coef_[0]):
                    if weight != 0:
                        feats_w_classifier_weight.append((weight, feature_names[index]))

                if label == lang:
                    print(filename, lang, 'yes',
                          '\t'.join(
                              [f'{label}\t{-value:3.3}' for value, label in
                               sorted(feats_w_classifier_weight)[:count]]),
                          file=outputfile, sep='\t')

                    print(filename, lang, 'no',
                          '\t'.join(
                              [f'{label}\t{-value:3.3}' for value, label in
                               sorted(feats_w_classifier_weight, reverse=True)[:count]]),
                          file=outputfile, sep='\t')
                else:
                    print(filename, lang, 'yes',
                          '\t'.join(
                              [f'{label}\t{value:3.3}' for value, label in
                               sorted(feats_w_classifier_weight, reverse=True)[:count]]),
                          file=outputfile, sep='\t')

                    print(filename, lang, 'no',
                          '\t'.join(
                              [f'{label}\t{value:3.3}' for value, label in sorted(feats_w_classifier_weight)[:count]]),
                          file=outputfile, sep='\t')

            elif algo == 'dtm':
                raise NotImplementedError()
            elif algo == 'dt':
                prune_duplicate_leaves(classifier)
                fig = plt.figure(figsize=(10, 10))
                try:
                    tree.plot_tree(classifier, feature_names=feature_names,
                                   class_names=[classifier.classes_[1], classifier.classes_[0]], filled=True,
                                   rounded=True, ax=fig.gca())
                    fig.savefig(os.path.join(plot_dir, filename[:-4] + '_' + lang + plot_format), bbox_inches='tight')
                except ValueError:
                    print(lang, classifier)
                plt.close()

    if outputfile:
        outputfile.close()
    print('inspected', dataset, algo)


if __name__ == '__main__':
    for dataset in [
        'toefl11',
        'reddit500k',
        'EFCAMDAT2'
        'EFCAMDAT2_L1',
        'EFCAMDAT2_L2',
        'EFCAMDAT2_L3'
    ]:
        for algo in ['svm']:  # , 'dt']:
            inspect(dataset, algo)
