import codecs
import pickle
import numpy as np
from collections import defaultdict
import re


def formulate_data(file):

    # formulate input file into a list, each entry consist of 2 sublist, one for the tweets, one for its labels

    f = codecs.open(file, 'r', "UTF-8")

    data = []
    post = []
    labels = []

    for x in f.readlines():
        if x != '\n':
            post.append(x.split()[0])
            labels.append(x.split()[1])
        else:
            pair = [post, labels]
            data.append(pair)
            post = []
            labels = []

    return data


def viterbi(obs, weights, labels):

    matrix = np.zeros((len(labels), len(obs)))
    path_mat = []

    for i in range(matrix.shape[1]):
        path = []

        if i == 0:

            for j in range(matrix.shape[0]):

                matrix[j][i] = weights['emis'+labels[j]+obs[i].lower()]
                # matrix[j][i] += weights["tag=%s_noun_suffix" % labels[j]]

                path.append(j)

        else:

            for j in range(matrix.shape[0]):

                score = []
                for k in range(matrix.shape[0]):
                    score.append(weights['trans'+labels[k]+labels[j]]+matrix[k][i - 1])

                score += weights['emis'+labels[j]+obs[i].lower()]*np.ones(matrix.shape[0])
                # score += weights["tag=%s_noun_suffix" % labels[j]]*np.ones(matrix.shape[0])

                matrix[j][i] = max(score)
                path.append(np.argmax(score))

        path_mat.append(path)

    best_path_rev = [np.argmax(matrix[:, -1])]
    path_mat = np.transpose(path_mat)

    for i in range(len(obs) - 1):
        best_path_rev.append(path_mat[best_path_rev[i]][-i-1])

    best_path = best_path_rev[::-1]
    best_label = []

    for i in best_path:
        best_label.append(labels[i])

    return best_label


def local_emis(t, label, tweet):
    word = tweet[t].lower()
    feats = {}

    feats["emis%s%s" % (label, word)] = 1

    nounSuffixes = ['age', 'al', 'ance', 'ence', 'dom', 'ee', 'er', 'or', 'hood', 'ism', 'ist', 'ity', 'ty', 'ment',
                    'ness', 'ry', 'ship', 'sion', 'tion', 'xion']

    # for suf in nounSuffixes:
    #     if re.findall('.*' + suf + '$', word):
    #         feats["tag=%s_noun_suffix" % label] = 1

    return feats


def feature_counts(tweet, labels):

    features = defaultdict(float)

    for i in range(len(tweet)):

        for feature in local_emis(i, labels[i], tweet):
            features[feature] += local_emis(i, labels[i], tweet)[feature]

        if i > 0:
            features['trans'+labels[i-1]+labels[i]] += 1

    return features


def predict_labels(tweet, weight):

    label_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
               'B-neutral', 'I-neutral']

    return viterbi(tweet, weight, label_list)


def train_perceptron(data, learn_rate, num_iter):

    weights = defaultdict(float)

    S = defaultdict(float)  # for calculating average weights
    t = 0

    for i in range(num_iter):
        print('training iteration %d' % i)

        for pair in data:

            tweet = pair[0]
            labels = pair[1]

            gold_feat = feature_counts(tweet, labels)

            pred_labels = predict_labels(tweet, weights)
            pred_feat = feature_counts(tweet, pred_labels)

            g = defaultdict(float)

            for feature in gold_feat.keys():
                g[feature] = gold_feat[feature]-pred_feat[feature]

            for feature in g:
                weights[feature] = weights[feature] + learn_rate*g[feature]
                S[feature] = S[feature] + (t-1)*learn_rate*g[feature]

            t += 1

        avg_weight = defaultdict(float)
        for feature in weights:
            avg_weight[feature] = weights[feature]-S[feature]/t

    with open('weights.pickle', 'wb') as f:
        pickle.dump(avg_weight, f)

    return avg_weight


def out(learn_rate, iter):

    data = formulate_data('train')

    weights = train_perceptron(data, learn_rate, iter)

    labels = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
                   'B-neutral', 'I-neutral']

    out = codecs.open('perceptron.out', 'w', "UTF-8")
    f = codecs.open('dev.in', 'r', "UTF-8")
    data = []
    post = []

    for x in f.readlines():
        if x != '\n':
            post.append(x.replace('\n', ''))
        else:
            data.append(post)
            post = []

    for post in data:
        v = viterbi(post, weights, labels)
        for l in range(len(v)):
            out.write(post[l]+' '+v[l]+'\n')
        out.write('\n')
    f.close()
    out.close()

out(1, 50)
#
# l = pickle.load(open("weights.pickle", "rb"))
# print(l['transOB-negative'])
# print(l['transOB-positive'])
# print(l['transOB-neutral'])
# print(l['transOO'])
# print(l['emisB-negativewine'])