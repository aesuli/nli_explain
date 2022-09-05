import os
import pickle
import random
from collections import Counter

dataset_path = r'd:\Users\esuli\Documents\Corpora\reddit.l2\reddit_full_posts_data\non_europe_data'

dataset_files = list()

for filename in os.listdir(dataset_path):
    if os.path.isfile(os.path.join(dataset_path, filename)):
        dataset_files.append(filename)

languages = [filename[7:filename.find('.', 7)] for filename in dataset_files]

prefix = 'reddit.'
suffix = '.tok.clean.csv'

print(languages)

min_len = 320

counts = Counter()

to_keep = {
'Germany',
'Netherlands',
'Sweden',
'France',
'Finland',
'Poland',
'Norway',
'Spain',
'Portugal',
'Romania',
'Italy',
}

X = list()
y = list()

max_size = 10000
counts = Counter()
for language in to_keep:
    with open(os.path.join(dataset_path, prefix + language + suffix), mode='r', encoding='utf-8') as inputfile:
        for line in inputfile:
            text = line.strip()
            if len(text) >= min_len and counts[language] <= max_size:
                if random.randint(0, 10) == 0:
                    X.append(text[text.find(',',text.find(',')+1)+1:])
                    y.append(language)
                    counts.update([language])
            if counts[language] == max_size:
                break

print(counts)

with open(os.path.join('data', 'reddit500k.pkl'), mode='wb') as outputfile:
    pickle.dump(X, outputfile)
    pickle.dump(y, outputfile)
