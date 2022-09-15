import os
import pickle
from collections import Counter

import bs4
import pandas as pd

os.makedirs('data',exist_ok=True)
if not os.path.exists(os.path.join('data', 'EFCAMDAT2_df.pkl')):
    xmlfile = r'/media/datasets/EFCAMDAT2/EF201403_selection121.xml'

    print('Loading xml', flush=True)
    with open(xmlfile, mode='r', encoding='utf-8') as inputfile:
        soup = bs4.BeautifulSoup(inputfile, 'xml')

    print('Filling dataframe', flush=True)
    writings = soup.find('writings')
    writing = writings.find_next('writing')
    count = 0
    data = list()
    while writing:
        count += 1
        if count % 1000 == 0:
            print(count, flush=True)
        data.append([writing['id'], writing['level'], writing['unit'], writing.find('learner')['nationality'],
                     writing.find('topic')['id'], writing.find('grade').get_text(),
                     ' '.join([txt.strip() for txt in writing.find('text').find_all(text=True)])])
        writing = writing.find_next('writing')
    df = pd.DataFrame(data, columns=['id', 'level', 'unit', 'nation', 'topic', 'grade', 'text'])
    print('Saving dataframe', flush=True)
    df.to_pickle(os.path.join('data', 'EFCAMDAT2_df.pkl'))

print('Loading dataframe', flush=True)
df = pd.read_pickle(os.path.join('data', 'EFCAMDAT2_df.pkl'))
selected_nations = {'ae', 'cn', 'it', 'fr', 'es', 'de', 'tw', 'jp', 'kr', 'tr', 'ru'}

print('Selecting data', flush=True)
X = list()
y = list()

per_nation = 2000
counter = Counter()
for index, row in df.iterrows():
    nation = row['nation']
    if nation in selected_nations and counter[nation] <= per_nation:
        X.append(row['text'])
        y.append(nation)
        counter.update([nation])

print('Saving selected data', flush=True)
with open(os.path.join('data', 'EFCAMDAT2.pkl'), mode='wb') as outputfile:
    pickle.dump(X, outputfile)
    pickle.dump(y, outputfile)
