from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from statistics import mode
import matplotlib.pyplot as plt
import re
import pickle
import random
import sys


# List of target words, and words that will be incorporated into confusion set
# target_words = ['government', 'university', 'church', 'people', 'american',
#                 'turkish', 'college', 'division', 'israeli', 'clipper',
#                 'military', 'public']

target_words = ['turkish', 'christian', 'israeli', 'american', 'national',
                'government', 'armenian', 'jewish', 'military', 'public',
                'western', 'society', 'russian', 'washington', 'administration',
                'country', 'president', 'political', 'toronto', 'pittsburgh']

confusion_set = ['computer', 'information', 'version', 'evidence', 'communications',
                 'algorithm', 'hardware', 'software', 'technical', 'numbers',
                 'package', 'network', 'driver', 'graphics', 'internet',
                 'display', 'server', 'engineering', 'machine', 'memory']

# Import the 20 newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='all')
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
    return cnt.most_common(400)


def extract_features(data, word, with_conj=True):
    examples = []
    filtered = [re.sub(r'[^\w\s]', '', s) for s in data
                if word in [re.sub(r'[^\w\s]', '', x) for x in s.split()]]
    for q, example in enumerate(filtered):
        example_split = example.split()

        # Place padding in word
        example_split = ['START', 'START', 'START'] + \
                        [x.lower() for x in list(map(p.stem, example_split))] + \
                        ['END', 'END', 'END']
        k = example_split.index(p.stem(word))
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
    accuracies = []

    # Convert features to sparse matrices
    v = DictVectorizer(sparse=True)

    # all_features = list(target_train.values()) + list(confusion_train.values()) + \
    #                list(target_test.values()) + list(confusion_test.values())
    # all_features = [x for i in all_features for x in i]
    # v.fit(all_features)


    # for target in target_words:
    #     print(target)
    #     models = []
    #     test_accs = []
    #     for i in range(2, 20):
    #         #clf = Perceptron(verbose=10, n_iter=1)
    #         clf = svm.LinearSVC()
    #         confusion_words = confusion_set[:i] #random.sample(confusion_set, i)
    #         X = [confusion_train[k] for k in confusion_words]
    #         X = [x for j in X for x in j]
    #         y = np.array([[-1] * len(X)])
    #         X += target_train[target]
    #         y = np.append(y, [[1] * len(target_train[target])])
    #         print(len(X))
    #         print(len(target_train[target]))
    #         X = v.fit_transform(X)

    #         X_test = [confusion_test[k] for k in confusion_words]
    #         X_test = [x for j in X_test for x in j]
    #         y_test = np.array([[-1] * len(X_test)])
    #         X_test += target_test[target]
    #         y_test = np.append(y_test, [1] * len(target_test[target]))
    #         print(len(X_test))
    #         print(len(target_test[target]))
    #         X_test = v.transform(X_test)

    #         # pca = PCA(n_components=2)
    #         # a = pca.fit_transform(X.todense())

    #         # print(a)
    #         # plt.scatter(a[:,0], a[:, 1])
    #         # plt.show()

    #         clf = clf.fit(X, y)
    #         models.append(clf)
    #         train_acc = clf.score(X, y)
    #         test_acc = clf.score(X_test, y_test) 
    #         # print(y_test)
    #         #print(clf.predict(X_test))
    #         # print(clf.coef_.tolist())
    #         # print(list(y_test))
    #         # print(list(clf.predict(X_test)))
    #         #print(list(X * clf.coef_.T + clf.intercept_))            
    #         #print(list(d * clf.coef_.T + clf.intercept_))
    #         #print(clf.coef_.tolist())
    #         print('Training Accuracy: %.03f' % train_acc)
    #         print('Test Accuracy: %.03f' % test_acc)
    #         test_accs.append(test_acc)
    #         print("success: %d" % i)
    #     full_models.append(models)
    #     accuracies.append(test_accs)


    models = []
    test_accs = [0] * 18
    baselines = [0] * 18
    for _ in range(5):
        for i in range(2, 20):
            clf = svm.LinearSVC()
            words = random.sample(target_words, i)

            X = []
            y = []

            for w in words:
                X += target_train[w]
                y += [w] * len(target_train[w])

            X = v.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            clf = clf.fit(X_train, y_train)

            models.append(clf)
            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)

            most_common = mode(y_train)

            baseline = np.sum(np.array([most_common] * len(y_train)) == np.array(y_train)) / len(y_train)
            baselines[i - 2] += baseline

            # print('Training Accuracy: %.4f' % train_acc)
            # print('Test Accuracy: %.4f' % test_acc)
            test_accs[i - 2] += test_acc

    test_accs = [j / 5 for j in test_accs]
    baselines = [j / 5 for j in baselines]
    print(test_accs)
    print(baselines)


    # Write the models to a pickle for future use
    with open('results/models.pkl', 'wb') as f:
        pickle.dump(full_models, f)

    with open('results/accuracies.pkl', 'wb') as f:
        pickle.dump(accuracies, f)

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