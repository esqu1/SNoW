from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re
import pickle
import random


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
    vocabulary = []
    examples = []
    conjs = []
    filtered = [re.sub(r'[^\w\s]', '', s) for s in data
                if word in [re.sub(r'[^\w\s]', '', x) for x in s.split()]]
    for q, example in enumerate(filtered):
        example_split = example.split()
        k = example_split.index(word)
        example_split = [x.lower() for x in list(map(p.stem, example_split))]
        counter, l = 0, k - 1
        window = []
        while counter < 3:
            if l < 0:
                window = [-1] * (3 - counter) + window
                break
            else:
                window = [example_split[l]] + window
                if example_split[l] not in vocabulary:
                    vocabulary.append(example_split[l])
                l -= 1
                counter += 1
        counter, l = 0, k + 1
        while counter < 3:
            if l >= len(example_split):
                window = window + [-1] * (3 - counter)
                break
            else:
                window = window + [example_split[l]]
                if example_split[l] not in vocabulary:
                    vocabulary.append(example_split[l])
                l += 1
                counter += 1
        indices = []
        for index in window:
            try:
                indices.append(vocabulary.index(index))
            except ValueError:
                indices.append(-1)

        if with_conj:
            conj_indices = []
            for i in range(len(window) - 1):
                try:
                    c = ' '.join(window[i:i + 2])
                    if c not in conjs:
                        conjs.append(c)
                    conj_indices.append(conjs.index(c))
                except TypeError:
                    conj_indices.append(-1)
            examples.append(indices + conj_indices)
        else:
            examples.append(indices)

    return {word: examples}, vocabulary, conjs


def pickle_words():
    # Save the pre-processed feature vectors to a pickle
    d = {}
    vocabulary = []
    conjs = []
    for target in target_words:
        print(target)
        features, v, c = extract_features(newsgroups_train.data, target)
        d[target] = np.array(features[target])
        for word in v:
            if word not in vocabulary:
                vocabulary.append(word)
        for conj in c:
            if conj not in conjs:
                conjs.append(conj)
    with open('data/target_train.pkl', 'wb') as f:
        pickle.dump(d, f)
    d = {}
    for con in confusion_set:
        print(con)
        features, v, c = extract_features(newsgroups_train.data, con)
        d[con] = np.array(features[con])
        for word in v:
            if word not in vocabulary:
                vocabulary.append(word)
        for conj in c:
            if conj not in conjs:
                conjs.append(conj)
    with open('data/confusion_train.pkl', 'wb') as f:
        pickle.dump(d, f)

    d = {}
    for target in target_words:
        print(target)
        features, v, c = extract_features(newsgroups_test.data, target)
        d[target] = np.array(features[target])
        for word in v:
            if word not in vocabulary:
                vocabulary.append(word)
        for conj in c:
            if conj not in conjs:
                conjs.append(conj)
    with open('data/target_test.pkl', 'wb') as f:
        pickle.dump(d, f)
    d = {}
    for con in confusion_set:
        print(con)
        features, v, c = extract_features(newsgroups_test.data, con)
        d[con] = np.array(features[con])
        for word in v:
            if word not in vocabulary:
                vocabulary.append(word)
        for conj in c:
            if conj not in conjs:
                conjs.append(conj)
    with open('data/confusion_test.pkl', 'wb') as f:
        pickle.dump(d, f)

    with open('data/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    with open('data/conjunctions.pkl', 'wb') as f:
        pickle.dump(conjs, f)
    print("Extraction complete. Length of vocabulary: %d" % len(vocabulary))


def main():
    with open('data/target_train.pkl', 'rb') as f:
        target_train_indices = pickle.load(f)
    with open('data/confusion_train.pkl', 'rb') as f:
        confusion_train_indices = pickle.load(f)
    with open('data/target_test.pkl', 'rb') as f:
        target_test_indices = pickle.load(f)
    with open('data/confusion_test.pkl', 'rb') as f:
        confusion_test_indices = pickle.load(f)
    with open('data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    with open('data/conjunctions.pkl', 'rb') as f:
        conjunctions = pickle.load(f)

    target_train, confusion_train = {}, {}
    target_test, confusion_test = {}, {}

    for key in list(target_train_indices.keys()):
        first = [create_sparse(len(vocabulary), x)
                 for x in target_train_indices[key][:, :6]]
        second = [create_sparse(len(conjunctions), x)
                  for x in target_train_indices[key][:, 6:]]
        target_train[key] = np.hstack((first, second))
    for key in list(confusion_train_indices.keys()):
        first = [create_sparse(len(vocabulary), x)
                 for x in confusion_train_indices[key][:, :6]]
        second = [create_sparse(len(conjunctions), x)
                  for x in confusion_train_indices[key][:, 6:]]
        confusion_train[key] = np.hstack((first, second))
    for key in list(target_test_indices.keys()):
        first = [create_sparse(len(vocabulary), x)
                 for x in target_test_indices[key][:, :6]]
        second = [create_sparse(len(conjunctions), x)
                  for x in target_test_indices[key][:, 6:]]
        target_test[key] = np.hstack((first, second))
    for key in list(confusion_train_indices.keys()):
        first = [create_sparse(len(vocabulary), x)
                 for x in confusion_test_indices[key][:6]]
        second = [create_sparse(len(conjunctions), x)
                  for x in confusion_test_indices[key][6:]]
        confusion_test[key] = np.hstack((first, second))

    full_models = []
    errors = []

    for target in target_words:
        models = []
        test_errors = []
        for i in range(2, 20):
            clf = svm.SVC()
            X = target_train[target]
            y = np.array([1] * len(X))
            X_test = target_test[target]
            y_test = np.array([1] * len(X_test))
            confusion_words = random.sample(confusion_set, i)
            print(X.shape)
            for word in confusion_words:
                X = np.append(X, confusion_train[word])
                y = np.append(y, [-1] * len(confusion_train[word]))
                X_test = np.append(X_test, confusion_test[word])
                y_test = np.append(y_test, [-1] * len(confusion_test[word]))
            clf = clf.fit(X, y)
            models.append(clf)
            test_error = clf.score(X_test, y_test)
            test_errors.append(test_error)
            print("success: %d" % i)
        full_models.append(models)
        errors.append(test_errors)

    # Write the models to a pickle for future use
    with open('results/models.pkl', 'wb') as f:
        pickle.dump(full_models, f)

    with open('results/errors.pkl', 'wb') as f:
        pickle.dump(errors, f)

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
#pickle_words()
#print popular_words()