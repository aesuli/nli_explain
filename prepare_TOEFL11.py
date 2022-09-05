import os
import pickle

import pandas as pd

label_file = r'e:\Users\Andrea Esuli\Documents\Corpora\nli-shared-task-2017\data\labels\train\labels.train.csv'

df = pd.read_csv(label_file)

train_label = dict(zip(list(df['test_taker_id']), list(df['L1'])))

print(len(train_label), set(train_label.values()))

dataset_path = r'e:\Users\Andrea Esuli\Documents\Corpora\nli-shared-task-2017\data\essays\train\tokenized'

dataset_files = list()

for filename in os.listdir(dataset_path):
    if os.path.isfile(os.path.join(dataset_path, filename)):
        dataset_files.append(filename)

train_text = dict()
for filename in dataset_files:
    with open(os.path.join(dataset_path, filename), mode='r', encoding='utf-8') as inputfile:
        train_text[int(filename[0:filename.rfind('.')])] = inputfile.read()

X = list()
y = list()
for key in train_label:
    X.append(train_text[key])
    y.append(train_label[key])

with open(os.path.join('data', 'toefl11.pkl'), mode='wb') as outputfile:
    pickle.dump(X, outputfile)
    pickle.dump(y, outputfile)
