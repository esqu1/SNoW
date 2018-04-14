from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from collections import Counter
import re
import sys
import random
import pickle
import sqlite3

# List of target words, and words that will be incorporated into confusion set
target_words = ['government', 'university', 'church', 'people', 'american', 'turkish', 'college', 'division', 'israeli', 'clipper',
                'military']

confusion_set = ['computer', 'information', 'version', 'evidence', 'president', 'algorithm' , 'hardware', 'software', 'technical', 'numbers',
                 'package', 'network', 'driver', 'graphics', 'internet', 'display', 'server', 'engineering', 'machine', 'memory']

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

p = PorterStemmer()
vocabulary = []

with open('vocabulary.txt') as f:
    vocabulary = f.read().split()

vocabulary = list(set(map(p.stem, vocabulary))) + target_words

inverse_vocab = {s: i for i, s in enumerate(vocabulary)}

def sparse(n, i):
    zeros = [0] * n
    if i == -1:
        return zeros
    zeros[i] = 1
    return zeros

def calculate_error(y, yhat):
    return len(y[y != yhat]) * 1.0 / len(y)

def popular_words():
    # generate a list of all nouns
    nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
    s = map(lambda x: x.split(), newsgroups_train.data)
    words = []
    for split in s:
        words += map(lambda x: x.lower().encode('utf8'), split)
    cnt = Counter(words)
    for word in stopwords.words('english'):
        if word in cnt.keys():
            del cnt[word]
    b = cnt.keys()
    for word in b:
        if len(word) == 1 or not word.isalpha() or not word in nouns or len(word) < 6:
            del cnt[word]
    return cnt.most_common(100)

def extract_features(data, word):
    examples = []
    filtered = [re.sub(r'[^\w\s]','',s) for s in data if word in map(lambda x: re.sub(r'[^\w\s]','',x), s.split())]
    for q, example in enumerate(filtered):
        example_split = example.split()
        # do something that converts example into a boolean vector
        k = example_split.index(word)
        #print example_split[k-3:k+4]
        example_split = map(lambda x: x.lower(), map(p.stem, example_split))
        counter, l = 0, k - 1
        window = []
        while counter < 3:
            if l < 0:
                window = [-1] * (3 - counter) + window
                break
            elif example_split[l] not in vocabulary:
                l -= 1
            else:
                window = [example_split[l]] + window
                l -= 1
                counter += 1
        counter, l = 0, k + 1
        while counter < 3:
            if l >= len(example_split):
                window = window + [-1] * (3 - counter)
                break
            elif example_split[l] not in vocabulary:
                l += 1
            else:
                window = window + [example_split[l]]
                l += 1
                counter += 1
        indices = []
        for index in window:
            try:
                indices.append(inverse_vocab[index])
            except:
                indices.append(-1)
        # convert to a sparse vector
        example_features = []
        for j in indices:
            example_features += sparse(len(vocabulary), j)
        examples.append(example_features)
    return {word: examples}

def pickle_words():
    for target in target_words:
        d = {}
        d.update(extract_features(newsgroups_train.data, target))
        with open('%s.pkl' % target, 'wb') as f:
            pickle.dump(d, f)
    for c in confusion_set:
        d = {}
        d.update(extract_features(newsgroups_train.data, c))
        with open('%s.pkl' % c,'wb') as f:
            pickle.dump(d, f)

def something_else(data, word, size): 

    #T = target_words[:size]
    T = random.sample(confusion_set, size - 1)
    T.append(word)
    examples = []
    y = []

    for i in T:
        filtered = [re.sub(r'[^\w\s]','',s) for s in data if i in map(lambda x: re.sub(r'[^\w\s]','',x), s.split())]
        for q, example in enumerate(filtered):
            example_split = example.split()
            # do something that converts example into a boolean vector
            k = example_split.index(i)
            #print example_split[k-3:k+4]
            example_split = map(lambda x: x.lower(), map(p.stem, example_split))
            counter, l = 0, k - 1
            window = []
            while counter < 3:
                if l < 0:
                    window = [-1] * (3 - counter) + window
                    break
                elif example_split[l] not in vocabulary:
                    l -= 1
                else:
                    window = [example_split[l]] + window
                    l -= 1
                    counter += 1
            counter, l = 0, k + 1
            while counter < 3:
                if l >= len(example_split):
                    window = window + [-1] * (3 - counter)
                    break
                elif example_split[l] not in vocabulary:
                    l += 1
                else:
                    window = window + [example_split[l]]
                    l += 1
                    counter += 1
            indices = []
            for index in window:
                try:
                    indices.append(inverse_vocab[index])
                except:
                    indices.append(-1)
            # convert to a sparse vector
            example_features = []
            for j in indices:
                example_features += sparse(len(vocabulary), j)
            examples.append(example_features)
            #y.append(inverse_vocab[i])
            if i != word:
                y.append(-1)
            else:
                y.append(1)

    return examples, y

def main():
    X_train, y_train = extract_features(newsgroups_train.data, int(sys.argv[1]))
    X_test, y_test = extract_features(newsgroups_test.data, int(sys.argv[1]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf = svm.SVC(kernel='linear')
    clf = clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)
    y_hat_train = clf.predict(X_train)

    print 'Confusion Set Size: %s' % sys.argv[1]
    print 'Training Error: %.4f' % calculate_error(y_train, y_hat_train)
    print 'Test Error: %.4f' % calculate_error(y_test, y_hat)

#main()

print pickle_words()