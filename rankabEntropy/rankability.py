# Rankability Module
#
# This module contains the functions necessary for measuring the rankability of a dataset.
# In particular, the dataset should be captured as a directed graph with weights between 
# zero and one.
# Given the corresponding adjacency matrix, a rankability measure is returned based on the 
# spectral-degree characterization of the graph Laplacian of a complete dominance graph.
#
# Author: Thomas R. Cameron
# Date: 11/1/2019
import numpy as np
import itertools
from math import factorial
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
def feature_selection_sim(matrix, measure='luca', p=1, drawfig = False, year = None):
    # Feature selection method using similarity measure and fuzzy entroropy
    # measures based on the article:

    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607
    import pandas as pd
    data = pd.DataFrame(matrix)
    m = data.shape[0]  # -samples
    t = data.shape[1]   # -features


    sim = data
    sim = normalize(sim, axis=1, norm='l1')
    if measure == 'luca':
        # moodifying zero and one values of the similarity values to work with
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 0] = delta
        sim[sim == 1] = 1 - delta

        H = (-sim * np.log(sim) - (1 - sim) * np.log(1 - sim)).sum(axis=1)  #better
        # H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)
    elif measure == 'park':
        H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)

    return (H/t).var(axis=0)/m * 1000
    # return (H).var(axis=0)



###############################################
###             Hausdorff                   ###
###############################################
#   Hausdorff distance between sets e and s.
###############################################
def Hausdorff(e,s):
    # spectral variation
    def _sv(e,s):
        return max([min([abs(e[i]-s[j]) for j in range(len(s))]) for i in range(len(e))])
    # Hausdorff distance
    return max(_sv(e,s),_sv(s,e))
###############################################
###             specR                       ###
###############################################
#   Computes Spectral-Degree Rankability Measure.
###############################################
def specR(a):
    # given graph Laplacian
    n = len(a)
    x = np.array([np.sum(a[i,:]) for i in range(n)])
    d = np.diag(x)
    l = d - a;
    # perfect dominance graph spectrum and out-degree
    s = np.array([n-k for k in range(1,n+1)])
    # eigenvalues of given graph Laplacian
    e = np.linalg.eigvals(l)
    # rankability measure
    return 1. - ((Hausdorff(e,s)+Hausdorff(x,s))/(2*(n-1)))
###############################################
###             edgeR                       ###
###############################################
#   Computes edge Rankability Measure using brute force approach.
###############################################
def edgeR(a):
    # size
    n = len(a)
    # complete dominance
    domMat = np.triu(1.0*np.ones((n,n)),1)
    # fitness list
    fitness = []
    # brute force (consider all permutations)
    for i in itertools.permutations(range(n)):
        b = a[i,:]
        b = b[:,i]
        # number of edge changes (k) for given permutation
        fitness.append(np.sum(np.abs(domMat - b)))
    # minimum number of edge chagnes
    k = np.amin(fitness)
    # number of permutations that gave this k
    p = np.sum(np.abs(fitness-k)<np.finfo(float).eps)
    # rankability measure
    return 1.0 - 2.0*k*p/(n*(n-1)*factorial(n))
###############################################
###             main                        ###
###############################################
#   main method tests SIMOD 1 examples
###############################################
def main():
    adj = [np.array([[0.,1,1,1,1,1],[0,0.,1,1,1,1],[0,0,0.,1,1,1],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,1,1,1],[0,0.,0,1,1,1],[1,0,0.,1,1,1],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,0,0,1],[0,0.,0,1,1,0],[0,0,0.,0,0,0],[1,1,0,0.,0,1],[1,1,0,0,0.,0],[1,1,1,0,1,0.]]),
            np.array([[0.,1,1,1,0,0],[0,0.,1,0,0,0],[0,0,0.,0,0,0],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,0,0,1],[0,0.,0,1,1,0],[0,0,0.,0,0,0],[1,0,0,0.,0,1],[1,1,0,0,0.,0],[0,1,1,0,1,0.]]),
            np.array([[0.,1,0,0,0,0],[0,0.,1,0,0,0],[0,0,0.,1,0,0],[0,0,0,0.,1,0],[0,0,0,0,0.,1],[1,0,0,0,0,0.]]),
            np.array([[0.,1,1,1,1,1],[1,0.,1,1,1,1],[1,1,0.,1,1,1],[1,1,1,0.,1,1],[1,1,1,1,0.,1],[1,1,1,1,1,0.]]),
            np.zeros((6,6))
            ]
    er = []
    sr = []
    for k in range(len(adj)):
        er.append(edgeR(adj[k]))
        sr.append(specR(adj[k]))
    corr,pval = spearmanr(er,sr)
    print('Anderson et al. Digraph Examples: ')
    print('edgeR = [%.4f' % er[0], end='')
    for k in range(len(er)):
        print(', %.4f' % er[k], end='')
    print(']')
    print('specR = [%.4f' % sr[0], end='')
    for k in range(len(sr)):
        print(', %.4f' % sr[k], end='')
    print(']')
    print('edgeR and specR corr = %.4f' % corr)
    print('edgeR and specR pval = %.4f' % pval)


def graphData():
    data = np.array([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    ])
    for label in range(5):
        a = data[label]
        er = edgeR(a)
        sr = specR(a)
        ou = feature_selection_sim(a)
        t= 14.243564650130098
        print("edge: ", np.round(er, 4), "specR: ", np.round(sr, 4), "entropy:", ou/t)

if __name__ == '__main__':
    # main()
    graphData()