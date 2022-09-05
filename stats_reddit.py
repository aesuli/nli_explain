import os
import pickle
from collections import defaultdict, Counter
from pprint import pprint

import pandas as pd
from tqdm.auto import tqdm


dataset_file = "D:/Users/esuli/Documents/Software/NLI/data/reddit500k.pkl"

with open(dataset_file, mode='rb') as inputfile:
    X = pickle.load(inputfile)
    y =  pickle.load(inputfile)

token_counts = defaultdict(int)
for text,nation in tqdm(zip(X,y)):
    token_counts[nation] += len(text.split())

label_counts = Counter(y)

pprint(label_counts)
pprint(token_counts)

dataset_file = "D:/Users/esuli/Documents/Software/NLI/data/reddit500kEN.pkl"

with open(dataset_file, mode='rb') as inputfile:
    X = pickle.load(inputfile)
    y =  pickle.load(inputfile)

token_counts = defaultdict(int)
for text,nation in tqdm(zip(X,y)):
    token_counts[nation] += len(text.split())

label_counts = Counter(y)

pprint(label_counts)
pprint(token_counts)
