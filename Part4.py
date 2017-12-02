import Part2
import Part3
import numpy as np


def max_marginal(emmis, trans, obs, labels):

    alpha = np.zeros((len(labels), len(obs)))

    for j in range(len(obs)):
        if j == 0:
            for u in range(len(labels)):
                alpha[u][j] = (trans['START'+labels[u]])
        else:
            for u in range(len(labels)):
                for v in range(len(labels)):
                    if labels[v]+'-->'+obs[j] in emmis.keys():
                        alpha[u][j] += alpha[v][j-1]*emmis[labels[v]+'-->'+obs[j]]*trans[labels[v]+labels[u]]
                    else:
                        alpha[u][j] += alpha[v][j-1]*emmis[labels[v]+'-->'+'#UNK#']*trans[labels[v]+labels[u]]

    beta = np.zeros((len(labels), len(obs)))

    for j in range(len(obs)):
        b = -j-1
        if b == -1:
            for u in range(len(labels)):
                if labels[u] + '-->' + obs[b] in emmis.keys():
                    beta[u][b] = trans[labels[u]+'STOP']*emmis[labels[u] + '-->' + obs[b]]
                else:
                    beta[u][b] = trans[labels[u]+'STOP']*emmis[labels[u] + '-->' + '#UNK#']
        else:
            for u in range(len(labels)):
                for v in range(len(labels)):
                    if labels[u] + '-->' + obs[b] in emmis.keys():
                        beta[u][b] += beta[v][b+1] * emmis[labels[u] + '-->' + obs[b]] * trans[labels[u] + labels[v]]
                    else:
                        beta[u][b] += beta[v][b+1] * emmis[labels[u] + '-->' + '#UNK#'] * trans[labels[u] + labels[v]]

    path = []

    for j in range(len(obs)):
        score = []
        for u in range(len(labels)):
            score.append(alpha[u][j]*beta[u][j])
        y = np.argmax(score)
        path.append(labels[y])

    return path

trans = Part3.mle_transition('train_fixed')
emmis = Part2.mle_emission('train_fixed')

labels = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
          'B-neutral', 'I-neutral']

out = open('EN_Part4.out', 'w')
f = open('dev.in', 'r')
data = []
post = []

for x in f.readlines():
    if x != '\n':
        post.append(x.replace('\n', ''))
    else:
        data.append(post)
        post = []

for post in data:
    v = max_marginal(emmis, trans, post, labels)
    for l in range(len(v)):
        out.write(post[l]+' '+v[l]+'\n')
    out.write('\n')
f.close()
out.close()
