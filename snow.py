from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re
import pickle
import random
import sys


# List of target words, and words that will be incorporated into confusion set
target_words = ['government', 'university', 'church', 'people', 'american',
                'turkish', 'college', 'division', 'israeli', 'clipper',
                'military']

confusion_set = ['computer', 'information', 'version', 'evidence', 'president',
                 'algorithm', 'hardware', 'software', 'technical', 'numbers',
                 'package', 'network', 'driver', 'graphics', 'internet',
                 'display', 'server', 'engineering', 'machine', 'memory']

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
        if index != -1:
            X[index + n * (i - 1)] = 1
    return X


def calculate_error(y, yhat):
    return len(y[y != yhat]) * 1.0 / len(y)


def popular_words():
    s = [x.split() for x in newsgroups_train.data]
    words = []
    for split in s:
        words += [x.lower().encode('utf8') for x in split]
    cnt = Counter(words)
    for word in stopwords.words('english'):
        if word in list(cnt.keys()):
            del cnt[word]
    b = list(cnt.keys())
    for word in b:
        if len(word) == 1 or not word.isalpha() or len(word) < 6:
            del cnt[word]
    return cnt.most_common(100)


def extract_features(data, word, with_conj=True):
    examples = []
    filtered = [re.sub(r'[^\w\s]', '', s) for s in data
                if word in [re.sub(r'[^\w\s]', '', x) for x in s.split()]]
    for q, example in enumerate(filtered):
        example_split = example.split()
        k = example_split.index(word)

        # Place padding in word
        example_split = ['START', 'START', 'START'] + \
                        [x.lower() for x in list(map(p.stem, example_split))] + \
                        ['END', 'END', 'END']
        window = example_split[k-3:k] + example_split[k+1:k+4]
        pos = list(range(-3,0)) + list(range(1, 4))
        features = {'w%d=%s' % (p, w) : 1 for p, w in zip(pos, window)}
        if with_conj:
            conjunctions = ['%s %s' % (window[i], window[i+1]) for i in range(len(window) - 1)]
            conj_pos = [(pos[i], pos[i+1]) for i in range(len(pos) - 1)]
            conj_features = {}
            for i, x in enumerate(conjunctions):
                conj_features['w%d%d=%s' % (conj_pos[i][0], conj_pos[i][1], x)] = 1
            features.update(conj_features)
        examples.append(features)
    return examples


def pickle_words(with_conj=False):
    # Save the pre-processed feature vectors to a pickle
    d = {}
    for target in target_words:
        print(target)
        d[target] = extract_features(newsgroups_train.data, target, with_conj=with_conj)
    with open('data/target_train.pkl', 'wb') as f:
        pickle.dump(d, f)
    d = {}
    for con in confusion_set:
        print(con)
        d[con] = extract_features(newsgroups_train.data, con, with_conj=with_conj)
    with open('data/confusion_train.pkl', 'wb') as f:
        pickle.dump(d, f)

    d = {}
    for target in target_words:
        print(target)
        d[target] = extract_features(newsgroups_test.data, target, with_conj=with_conj)
    with open('data/target_test.pkl', 'wb') as f:
        pickle.dump(d, f)
    d = {}
    for con in confusion_set:
        print(con)
        d[con] = extract_features(newsgroups_test.data, con, with_conj=with_conj)
    with open('data/confusion_test.pkl', 'wb') as f:
        pickle.dump(d, f)


def main():
    with open('data/target_train.pkl', 'rb') as f:
        target_train = pickle.load(f)
    with open('data/confusion_train.pkl', 'rb') as f:
        confusion_train = pickle.load(f)
    with open('data/target_test.pkl', 'rb') as f:
        target_test = pickle.load(f)
    with open('data/confusion_test.pkl', 'rb') as f:
        confusion_test = pickle.load(f)

    full_models = []
    errors = []

    # Convert features to sparse matrices
    v = DictVectorizer(sparse=True)

    all_features = list(target_train.values()) + list(confusion_train.values()) + \
                   list(target_test.values()) + list(confusion_test.values())
    all_features = [x for i in all_features for x in i]
    v.fit_transform(all_features)


    for target in target_words:
        print(target)
        models = []
        test_errors = []
        for i in range(2, 20):
            clf = svm.SVC(kernel='linear')
            confusion_words = random.sample(confusion_set, i)
            X = [confusion_train[k] for k in confusion_words]
            y = np.array([[-1] * len([x for j in X for x in j])])
            X.append(target_train[target])
            X = [x for j in X for x in j]
            y = np.append(y, [[1] * len(target_train[target])])
            X = v.transform(X)
            print(X.shape)

            X_test = [confusion_test[k] for k in confusion_words]
            y_test = np.array([[-1] * len([x for j in X_test for x in j])])
            X_test.append(target_test[target])
            X_test = [x for j in X_test for x in j]
            y_test = np.append(y_test, [1] * len(target_test[target]))
            X_test = v.transform(X_test)

            print(cross_val_score(clf, X, y, cv=3))

            clf = clf.fit(X, y)
            models.append(clf)
            train_error = clf.score(X, y)
            test_error = clf.score(X_test, y_test)
            # print(y_test)
            # print(clf.predict(X_test))
            print('Training Error: %.03f' % train_error)
            print('Test Error: %.03f' % test_error)
            test_errors.append(test_error)
            print("success: %d" % i)
        full_models.append(models)
        errors.append(test_errors)

    # Write the models to a pickle for future use
    with open('results/models.pkl', 'wb') as f:
        pickle.dump(full_models, f)

    with open('results/errors.pkl', 'wb') as f:
        pickle.dump(errors, f)

if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        pickle_words(False)
    elif sys.argv[1] == 'extractconj':
        pickle_words(True)
    elif sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'popular':
        print(popular_words())
    else:
        print('Invalid input.')