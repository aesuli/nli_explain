import os
import pickle

import re
import sys
from collections import defaultdict
from pprint import pprint

if __name__ == '__main__':
    dataset_path = sys.argv[1]+r'/LOCNESS-corpus-files'

    dataset_files = list()

    for filename in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, filename)) and filename.endswith('.txt'):
            dataset_files.append(filename)

    X = list()
    y = list()
    counts = defaultdict(int)
    for filename in dataset_files:
        with open(os.path.join(dataset_path, filename), mode='r', encoding='utf-8', errors='ignore') as inputfile:
            text = ''
            for line in inputfile:
                line = line.strip()
                if line.startswith('<'):
                    text = ''
                    continue
                line = re.sub('<.{1,40}?>', '  ', line)
                if len(line):
                    text += ' ' + line
                else:
                    if len(text) > 60:
                        counts[filename[0]] += 1
                        X.append(text)
                        y.append('en')
                    text = ''

    print(len(X))
    pprint(counts)

    os.makedirs('data',exist_ok=True)
    with open(os.path.join('data', 'LOCNESS.pkl'), mode='wb') as outputfile:
        pickle.dump(X, outputfile)
        pickle.dump(y, outputfile)
