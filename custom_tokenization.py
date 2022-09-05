import nltk
import spacy

nlp = None


def spacy_tokenizer(text):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load('en_core_web_sm')
        except:
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    features = dict()

    # tokens
    tokens = [token.text.strip() for token in doc if len(token.text.strip()) > 0]
    features['T1'] = ['T1_' + token for token in tokens]
    features['T2'] = ['T2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(tokens, 2)]
    features['T3'] = ['T3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(tokens, 3)]

    # tokens with NER filter
    ner_to_filter = {'PERSON', 'PER', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'LANGUAGE', 'MONEY'}
    ner_tokens = list()
    for token in doc:
        if len(token.text.strip()) > 0:
            if token.ent_type_ in ner_to_filter:
                ner_tokens.append(token.ent_type_)
            else:
                ner_tokens.append(token.text.strip())
    features['Tn1'] = ['Tn1_' + token for token in ner_tokens]
    features['Tn2'] = ['Tn2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(ner_tokens, 2)]
    features['Tn3'] = ['Tn3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(ner_tokens, 3)]

    # lemmatized
    lemmas = [token.lemma_.strip() for token in doc if len(token.lemma_.strip()) > 0]
    features['L1'] = ['L1_' + w for w in lemmas]
    features['L2'] = ['L2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(lemmas, 2)]
    features['L3'] = ['L3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(lemmas, 3)]

    # lemmatized tokens with NER filter
    ner_lemmas = list()
    for token in doc:
        if len(token.lemma_.strip()) > 0:
            if token.ent_type_ in ner_to_filter:
                ner_lemmas.append(token.ent_type_)
            else:
                ner_lemmas.append(token.lemma_.strip())
    features['Ln1'] = ['Ln1_' + w for w in ner_lemmas]
    features['Ln2'] = ['Ln2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(ner_lemmas, 2)]
    features['Ln3'] = ['Ln3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(ner_lemmas, 3)]

    # pos tagging features
    poss = [token.tag_.strip() for token in doc if len(token.tag_.strip()) > 0]

    features['P1'] = ['P1_' + w for w in poss]
    features['P2'] = ['P2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(poss, 2)]
    features['P3'] = ['P3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(poss, 3)]

    # tokens with POS filter
    to_mask = set(['ADD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'XX'])

    mposs = list()
    for token in doc:
        if token.tag_.strip() in to_mask:
            if len(token.tag_.strip()) > 0:
                mposs.append(token.tag_.strip())
        else:
            if len(token.text.strip()):
                mposs.append(token.text.strip())

    features['Tp1'] = ['Tp1_' + w for w in mposs]
    features['Tp2'] = ['Tp2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(mposs, 2)]
    features['Tp3'] = ['Tp3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(mposs, 3)]

    # lemmatized tokens with POS filter
    mlposs = list()
    for token in doc:
        if token.tag_.strip() in to_mask:
            if len(token.tag_.strip()) > 0:
                mlposs.append(token.tag_.strip())
        else:
            if len(token.lemma_.strip()):
                mlposs.append(token.lemma_.strip())

    features['Lp1'] = ['Lp1_' + w for w in mlposs]
    features['Lp2'] = ['Lp2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(mlposs, 2)]
    features['Lp3'] = ['Lp3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(mlposs, 3)]

    # mixed word-pos tag representation with suffix marking
    suff_dic = {
        'ADJ': ['able', 'ac', 'al', 'an', 'ian', 'ar', 'ary', 'ate', 'ative', 'ent', 'ern', 'ese', 'esque', 'etic',
                'ful', 'gonic', 'ial', 'ian', 'iatric', 'ible', 'ic', 'ical', 'ile', 'ine', 'ious', 'ish', 'ive',
                'less', 'like', 'ous', 'ose', 'plegic', 'some', 'sophic', 'ular', 'uous', 'ward', 'wise', 'y'],
        'NOUN': ['ade', 'age', 'acity', 'ancy', 'ard', 'art', 'cade', 'drome', 'ery', 'ocity', 'aholic', 'oholic',
                 ' algia', 'ance', 'ant', 'ard', 'arian', 'arium', 'orium', 'ation', 'cide', 'cracy', 'crat', 'cy',
                 'cycle', 'dom', 'dox', 'ectomy', 'ee', 'eer', 'emia', 'ence', 'ency', 'er', 'escence', 'ess', 'ette',
                 'gon', 'hood', 'iasis', 'ion', 'ism', 'ist', 'ite', 'itis', 'ity', 'isation', 'ization', 'let', 'ling',
                 'loger', 'logist', 'log', 'logue', 'ment', 'ness', 'oid', 'ology', 'oma', 'onym', 'opia', 'opsy', 'or',
                 'ory', 'osis', 'ostomy', 'otomy', 'path', 'pathy', 'phile', 'phobia', 'phone', 'phyte', 'plegia',
                 'pnea', 'scopy', 'scope', 'script', 'ship', 'sion', 'sophy', 'th', 'tion', 'tome', 'tomy', 'trophy',
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

    features['Ms1'] = ['Ms1_' + w for w in msposs]
    features['Ms2'] = ['Ms2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(msposs, 2)]
    features['Ms3'] = ['Ms3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(msposs, 3)]

    # dependency parsing features
    deps = [token.dep_.strip() for token in doc if len(token.dep_.strip()) > 0]
    features['D1'] = ['D1_' + w for w in deps]
    features['D2'] = ['D2_' + w1 + '_' + w2 for w1, w2 in nltk.ngrams(deps, 2)]
    features['D3'] = ['D3_' + w1 + '_' + w2 + '_' + w3 for w1, w2, w3 in nltk.ngrams(deps, 3)]

    features['WL'] = ['WL_' + str(len(token)) for token in tokens]

    features['SL'] = ['SL_' + str(len(sent)) for sent in doc.sents]

    # dependency parse tree complexity
    depdepths = list()
    for token in doc:
        depth = 0
        while token != token.head:
            token = token.head
            depth += 1
        depdepths.append(depth)
    features['DD'] = ['DD_' + str(dd) for dd in depdepths]

    return features


def list_tokenizer(fields, text):
    tokens = list()
    for key in fields:
        tokens.extend(text[key])
    return tokens


if __name__ == '__main__':
    print(spacy_tokenizer('I like cookies that contain butter'))