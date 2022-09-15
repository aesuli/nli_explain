import os
import pickle
import sys
from collections import Counter

import bs4
import pandas as pd

if __name__ == '__main__':
    os.makedirs('data',exist_ok=True)
    if not os.path.exists(os.path.join('data', 'EFCAMDAT2_df.pkl')):
        xmlfile = sys.argv[1]+r'/EFCAMDAT2/EF201403_selection121.xml'

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
    X1 = list()
    y1 = list()
    X2 = list()
    y2 = list()
    X3 = list()
    y3 = list()

    levels = set()
    per_nation = 2000
    counter = Counter()
    for index, row in df.iterrows():
        nation = row['nation']
        level = int(row['level'])
        if level < 6:
            level = '1'
        elif level < 13:
            level = '2'
        else:
            level = '3'
        levels.add(level)
        if nation in selected_nations and counter[nation + '_' + level] <= per_nation:
            if level == '1':
                X1.append(row['text'])
                y1.append(nation)
            elif level == '2':
                X2.append(row['text'])
                y2.append(nation)
            elif level == '3':
                X3.append(row['text'])
                y3.append(nation)
            counter.update([nation + '_' + level])

    print('Saving selected data', flush=True)
    with open(os.path.join('data', 'EFCAMDAT2_L1.pkl'), mode='wb') as outputfile:
        pickle.dump(X1, outputfile)
        pickle.dump(y1, outputfile)

    with open(os.path.join('data', 'EFCAMDAT2_L2.pkl'), mode='wb') as outputfile:
        pickle.dump(X2, outputfile)
        pickle.dump(y2, outputfile)

    with open(os.path.join('data', 'EFCAMDAT2_L3.pkl'), mode='wb') as outputfile:
        pickle.dump(X3, outputfile)
        pickle.dump(y3, outputfile)
