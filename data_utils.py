# data_utils.py

import numpy as np
from Tom_New.similarity_methods import GIP_Calculate, GIP_Calculate1
from Tom_New.similarity_methods import *
# Load ABiofilm dataset
def load_aBiofilm_data():
    A = np.loadtxt("data/ABiofilm/drug_microbe_matrix.txt")
    # A = np.loadtxt("./data/ABiofilm/drug_dmicrobe_matrix.txt")
    mis = np.loadtxt('data/ABiofilm/microbe_similarity.txt')
    drs = np.loadtxt('data/ABiofilm/drug_similarity.txt')
    known = np.loadtxt("data/ABiofilm/known.txt")
    unknown = np.loadtxt("data/ABiofilm/unknown.txt")
    labels = np.loadtxt('data/ABiofilm/adj.txt')
    return A, mis, drs, known, unknown, labels


def RWR(SM, alpha=0.1):
    E = np.identity(len(SM))
    M = np.zeros((len(SM), len(SM)))
    s = []
    for i in range(len(M)):
        row_sum = np.sum(SM[i, :])
        if row_sum == 0:

            M[i, :] = 0
        else:
            for j in range(len(M)):
                M[i][j] = SM[i][j] / row_sum

    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return s



def normalize_matrix(W):
    W = np.asarray(W)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums

def Multi_View_RWR_with_Weights(W_views, alpha=0.1, beta=0.1, max_iter=10):
    n_views = len(W_views)
    n_nodes = len(W_views[0])
    M_views = [normalize_matrix(W) for W in W_views]
    restart_vectors = []
    for i in range(n_nodes):
        E = np.identity(n_nodes)
        e_i = E[i, :]
        p_views = [np.copy(e_i) for _ in range(n_views)]
        for k in range(max_iter):
            new_p_views = []
            for v in range(n_views):
                p_v = p_views[v]
                p_v_new = alpha * np.dot(M_views[v], p_v) + (1 - alpha) * e_i
                switch_term = np.sum([beta * p_views[v_alt] for v_alt in range(n_views) if v_alt != v], axis=0)
                p_v_new += switch_term / (n_views - 1)
                new_p_views.append(p_v_new)
            p_views = new_p_views
        final_restart_vector = np.mean(p_views, axis=0)
        restart_vectors.append(final_restart_vector)

    return restart_vectors


def normalize_matrix2(matrix):
    row_sums = matrix.sum(axis=0, keepdims=True)
    # 防止出现除以零的情况
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums
    return np.nan_to_num(normalized_matrix)

def Net_construct(Srr, Smm, A):
    N1 = np.hstack((Srr, A))
    N2 = np.hstack((A.T, Smm))
    Net = np.vstack((N1, N2))
    return Net


