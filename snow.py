from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from collections import Counter
import re
import pickle
import random 

# List of target words, and words that will be incorporated into confusion set
target_words = ['government', 'university', 'church', 'people', 'american', 'turkish', 'college', 'division', 'israeli', 'clipper',
                'military']

confusion_set = ['computer', 'information', 'version', 'evidence', 'president', 'algorithm' , 'hardware', 'software', 'technical', 'numbers',
                 'package', 'network', 'driver', 'graphics', 'internet', 'display', 'server', 'engineering', 'machine', 'memory']

# Import the 20 newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# Lemmatizer
p = PorterStemmer()

def sparse(n, i):
    zeros = [0] * n
    if i == -1:
        return zeros
    zeros[i] = 1
    return zeros

def create_sparse(n, arr):
    X = np.zeros(n * len(arr))
    for i, index in enumerate(arr):
        X[index + n * (i - 1)] = 1
    return X

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

def extract_features(data, word, with_conj=False):
    vocabulary = set({})
    examples = []
    filtered = [re.sub(r'[^\w\s]','',s) for s in data if word in map(lambda x: re.sub(r'[^\w\s]','',x), s.split())]
    for q, example in enumerate(filtered):
        example_split = example.split()
        k = example_split.index(word)
        example_split = map(lambda x: x.lower(), map(p.stem, example_split))
        counter, l = 0, k - 1
        window = []
        while counter < 3:
            if l < 0:
                window = [-1] * (3 - counter) + window
                break
            else:
                window = [example_split[l]] + window
                vocabulary.update([example_split[l]])
                l -= 1
                counter += 1
        counter, l = 0, k + 1
        while counter < 3:
            if l >= len(example_split):
                window = window + [-1] * (3 - counter)
                break
            else:
                window = window + [example_split[l]]
                vocabulary.update([example_split[l]])
                l += 1
                counter += 1
        indices = []
        for index in window:
            indices.append(vocabulary.find(index))
        examples.append(indices)
    return {word: examples}, vocabulary

def pickle_words():
    d = {}
    vocabulary = set([])
    for target in target_words:
        print target
        features, v = extract_features(newsgroups_train.data, target)
        d.update(features)
        vocabulary.update(v)
    with open('target_train.pkl', 'wb') as f:
        pickle.dump(d, f)
    d = {}
    for c in confusion_set:
        print c
        features, v = extract_features(newsgroups_train.data, c)
        d.update(features)
        vocabulary.update(v)
    with open('confusion.pkl','wb') as f:
        pickle.dump(d, f)
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print "Extraction complete. Length of vocabulary: %d" % len(vocabulary)
    #conn.commit()

def main():
    with open('target_train.pkl', 'rb') as f:
        target_train_indices = pickle.load(f)
    with open('confusion.pkl', 'rb') as f:
        confusion_train_indices = pickle.load(f)
    with open('vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    target_train, confusion_train = {}, {}
    for key in target_train_indices.keys():
        #wrong
        target_train[key] = map(lambda x: create_sparse(len(vocabulary), x), target_train_indices[key])
    for key in confusion_train_indices.keys():
        confusion_train[key] = map(lambda x: create_sparse(len(vocabulary), x), confusion_train_indices[key])

    models = []
    for target in target_words:
        for i in range(2,20):
            clf = svm.SVC()
            X = target_train[target]
            y = np.array([1] * len(X))
            confusion_words = random.sample(confusion_set, i)
            for word in confusion_words:
                X += confusion_train[word]
                y = np.append(y, [-1] * len(confusion_train[word]))
            clf = clf.fit(X, y)
            print "success: %d" % i


    # for target in target_words:
    #      += sparse(len(vocabulary), j)
    # examples.append(example_features)


# def main():
#     X_train, y_train = extract_features(newsgroups_train.data, int(sys.argv[1]))
#     X_test, y_test = extract_features(newsgroups_test.data, int(sys.argv[1]))

#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)

#     clf = svm.SVC(kernel='linear')
#     clf = clf.fit(X_train, y_train)

#     y_hat = clf.predict(X_test)
#     y_hat_train = clf.predict(X_train)

#     print 'Confusion Set Size: %s' % sys.argv[1]
#     print 'Training Error: %.4f' % calculate_error(y_train, y_hat_train)
#     print 'Test Error: %.4f' % calculate_error(y_test, y_hat)

main()

#print pickle_words()