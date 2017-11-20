import Part2
import Part3
import numpy as np

def max_marginal(emmis, trans, obs, labels):

    alpha = []

    for j in range(len(obs)):
        alpha_j = []
        if j == 0:
            for u in range(len(labels)):
                alpha_j.append(trans['START'+labels[u]])
        else:
            for u in range(len(labels)):
                score = 0
                for v in range(len(labels)):
                    score += alpha[j-1][v]*emmis[labels[v]+'-->'+obs[j]]*trans[labels[v]+labels[u]]
                alpha_j.append(score)
        alpha.append(alpha_j)

    alpha = np.transpose(alpha)

    beta = []

    for j in range(len(obs)):
        beta_j = []
        b = -j-1
        if b == -1:
            for u in range(len(labels)):
                beta_j.append(trans[labels[u]+'STOP'])
        else:
            for u in range(len(labels)):
                score = 0
                for v in range(len(labels)):
                    score += beta[j-1][v] * emmis[labels[v] + '-->' + obs[b]] * trans[labels[v] + labels[u]]
                beta_j.append(score)
        beta.append(beta_j)

    beta = beta[::-1]

    beta = np.transpose(beta)

    path = []

    for j in range(len(obs)):
        score = []
        for u in range(len(labels)):
            score.append(alpha[u][j]*beta[u][j])
        y = np.argmax(score)
        path.append(labels[y])

    return path

trans = Part3.mle_transition('train')
emmis = Part2.mle_emission('train')
labels = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
inp = "the food good tuna good ."
obs = inp.split()
print(max_marginal(emmis, trans, obs, labels))