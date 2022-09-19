import nltk
import spacy

nlp = None

feature_types = [
    'T1',
    'T2',
    'T3',
    'Tn1',
    'Tn2',
    'Tn3',
    'L1',
    'L2',
    'L3',
    'Ln1',
    'Ln2',
    'Ln3',
    'P1',
    'P2',
    'P3',
    'D1',
    'D2',
    'D3',
    'Tp1',
    'Tp2',
    'Tp3',
    'Lp1',
    'Lp2',
    'Lp3',
    'Ms1',
    'Ms2',
    'Ms3',
    'WL',
    'SL',
    'DD',
]


def spacy_tokenizer(text, feature_type):
    global nlp
    if type(text) == str:
        if nlp is None:
            try:
                nlp = spacy.load('en_core_web_sm')
            except:
                spacy.cli.download('en_core_web_sm')
                nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
    else:
        doc = text

    # tokens
    if feature_type in ['T1', 'T2', 'T3']:
        tokens = [token.text.strip() for token in doc if len(token.text.strip()) > 0]
        if feature_type == 'T1':
            return ['T1_' + token for token in tokens]
        elif feature_type == 'T2':
            return ['T2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(tokens, 2)]
        elif feature_type == 'T3':
            return ['T3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(tokens, 3)]

    # tokens with NER filter
    ner_to_filter = {'PERSON', 'PER', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'LANGUAGE', 'MONEY'}

    if feature_type in ['Tn1', 'Tn2', 'Tn3']:
        ner_tokens = list()
        for token in doc:
            if len(token.text.strip()) > 0:
                if token.ent_type_ in ner_to_filter:
                    ner_tokens.append(token.ent_type_)
                else:
                    ner_tokens.append(token.text.strip())
        if feature_type == 'Tn1':
            return ['Tn1_' + token for token in ner_tokens]
        elif feature_type == 'Tn2':
            return ['Tn2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(ner_tokens, 2)]
        elif feature_type == 'Tn3':
            return ['Tn3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(ner_tokens, 3)]

    # lemmatized
    if feature_type in ['L1', 'L2', 'L3']:
        lemmas = [token.lemma_.strip() for token in doc if len(token.lemma_.strip()) > 0]
        if feature_type == 'L1':
            return ['L1_' + w for w in lemmas]
        elif feature_type == 'L2':
            return ['L2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(lemmas, 2)]
        elif feature_type == 'L3':
            return ['L3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(lemmas, 3)]

    # lemmatized tokens with NER filter
    if feature_type in ['Ln1', 'Ln2', 'Ln3']:
        ner_lemmas = list()
        for token in doc:
            if len(token.lemma_.strip()) > 0:
                if token.ent_type_ in ner_to_filter:
                    ner_lemmas.append(token.ent_type_)
                else:
                    ner_lemmas.append(token.lemma_.strip())
        if feature_type == 'Ln1':
            return ['Ln1_' + w for w in ner_lemmas]
        elif feature_type == 'Ln2':
            return ['Ln2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(ner_lemmas, 2)]
        elif feature_type == 'Ln3':
            return ['Ln3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(ner_lemmas, 3)]

    # pos tagging features
    if feature_type in ['P1', 'P2', 'P3']:
        poss = [token.tag_.strip() for token in doc if len(token.tag_.strip()) > 0]

        if feature_type == 'P1':
            return ['P1_' + w for w in poss]
        elif feature_type == 'P2':
            return ['P2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(poss, 2)]
        elif feature_type == 'P3':
            return ['P3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(poss, 3)]

    to_mask = {'ADD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'XX'}

    # tokens with POS filter
    if feature_type in ['Tp1', 'Tp2', 'Tp3']:

        mposs = list()
        for token in doc:
            if token.tag_.strip() in to_mask:
                if len(token.tag_.strip()) > 0:
                    mposs.append(token.tag_.strip())
            else:
                if len(token.text.strip()):
                    mposs.append(token.text.strip())

        if feature_type == 'Tp1':
            return ['Tp1_' + w for w in mposs]
        elif feature_type == 'Tp2':
            return ['Tp2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(mposs, 2)]
        elif feature_type == 'Tp3':
            return ['Tp3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(mposs, 3)]

    # lemmatized tokens with POS filter
    if feature_type in ['Lp1', 'Lp2', 'Lp3']:
        mlposs = list()
        for token in doc:
            if token.tag_.strip() in to_mask:
                if len(token.tag_.strip()) > 0:
                    mlposs.append(token.tag_.strip())
            else:
                if len(token.lemma_.strip()):
                    mlposs.append(token.lemma_.strip())

        if feature_type == 'Lp1':
            return ['Lp1_' + w for w in mlposs]
        elif feature_type == 'Lp2':
            return ['Lp2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(mlposs, 2)]
        elif feature_type == 'Lp3':
            return ['Lp3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(mlposs, 3)]

    # mixed word-pos tag representation with suffix marking
    if feature_type in ['Ms1', 'Ms2', 'Ms3']:
        suff_dic = {
            'ADJ': ['able', 'ac', 'al', 'an', 'ian', 'ar', 'ary', 'ate', 'ative', 'ent', 'ern', 'ese', 'esque', 'etic',
                    'ful', 'gonic', 'ial', 'ian', 'iatric', 'ible', 'ic', 'ical', 'ile', 'ine', 'ious', 'ish', 'ive',
                    'less', 'like', 'ous', 'ose', 'plegic', 'some', 'sophic', 'ular', 'uous', 'ward', 'wise', 'y'],
            'NOUN': ['ade', 'age', 'acity', 'ancy', 'ard', 'art', 'cade', 'drome', 'ery', 'ocity', 'aholic', 'oholic',
                     ' algia', 'ance', 'ant', 'ard', 'arian', 'arium', 'orium', 'ation', 'cide', 'cracy', 'crat', 'cy',
                     'cycle', 'dom', 'dox', 'ectomy', 'ee', 'eer', 'emia', 'ence', 'ency', 'er', 'escence', 'ess',
                     'ette',
                     'gon', 'hood', 'iasis', 'ion', 'ism', 'ist', 'ite', 'itis', 'ity', 'isation', 'ization', 'let',
                     'ling',
                     'loger', 'logist', 'log', 'logue', 'ment', 'ness', 'oid', 'ology', 'oma', 'onym', 'opia', 'opsy',
                     'or',
                     'ory', 'osis', 'ostomy', 'otomy', 'path', 'pathy', 'phile', 'phobia', 'phone', 'phyte', 'plegia',
                     'pnea', 'scopy', 'scope', 'script', 'ship', 'sion', 'sophy', 'th', 'tion', 'tome', 'tomy',
                     'trophy',
                     'tude', 'ty', 'ure', 'ware', 'iatry', 'ice'],
            'VERB': ['en', 'fy', 'ize', 'ise', 'scribe', 'sect'],
            'ADV': ['ily', 'ly', 'ward', 'wise', 'fold']}

        msposs = list()
        for token in doc:
            if token.tag_.strip() in to_mask:
                if token.pos_ in suff_dic and len(token.lemma_) > 4:
                    search_suffix = [token.lemma_.endswith(suffix) for suffix in suff_dic[token.pos_]]
                    if any(search_suffix):
                        idx = search_suffix.index(True)
                        msposs.append(token.tag_ + '_' + suff_dic[token.pos_][idx])
                else:
                    msposs.append(token.tag_.strip())
            else:
                if len(token.text.strip()):
                    msposs.append(token.text.strip())

        if feature_type == 'Ms1':
            return ['Ms1_' + w for w in msposs]
        elif feature_type == 'Ms2':
            return ['Ms2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(msposs, 2)]
        elif feature_type == 'Ms3':
            return ['Ms3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(msposs, 3)]

    # dependency parsing features
    if feature_type in ['D1', 'D2', 'D3']:
        deps = [token.dep_.strip() for token in doc if len(token.dep_.strip()) > 0]
        if feature_type == 'D1':
            return ['D1_' + w for w in deps]
        if feature_type == 'D1':
            return ['D2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(deps, 2)]
        if feature_type == 'D1':
            return ['D3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(deps, 3)]

    if feature_type == 'WL':
        tokens = [token.text.strip() for token in doc if len(token.text.strip()) > 0]
        return ['WL_' + str(len(token)) for token in tokens]

    if feature_type == 'SL':
        return ['SL_' + str(len(sent)) for sent in doc.sents]

    # dependency parse tree complexity
    if feature_type == 'DD':
        depdepths = list()
        for token in doc:
            depth = 0
            while token != token.head:
                token = token.head
                depth += 1
            depdepths.append(depth)
        return ['DD_' + str(dd) for dd in depdepths]


def dummy_tokenizer(tokenized_doc):
    return tokenized_doc


if __name__ == '__main__':
    for feature_type in feature_types:
        print(feature_type)
        print(spacy_tokenizer('I like cookies that contain butter', feature_type))
