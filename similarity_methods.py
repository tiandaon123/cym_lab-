import numpy as np
import math
import scipy.sparse as sp
import torch as th
from sklearn.metrics.pairwise import rbf_kernel




def functional_microbe():
    MF = np.loadtxt('data/ABiofilm/Microbe_functional.txt')
    return MF

def structural_drug():
    DS = np.loadtxt('data/ABiofilm/Drug_structural.txt')
    return DS

def HIP_Calculate(M):
    l=len(M)
    cl=np.size(M,axis=1)
    SM=np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k]!=M[j][k]:
                    dnum=dnum+1
            SM[i][j]=1-dnum/cl
    np.savetxt(r"data\ABiofilm\Drug_HIP_Similarity", SM)
    return SM

def HIP_Calculate1(M):
    M_T = M.T
    l = len(M_T)
    cl = np.size(M_T, axis=1)
    SM1 = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M_T[i][k] != M_T[j][k]:
                    dnum += 1
            SM1[i][j] = 1 - dnum / cl
    np.savetxt(r"data\ABiofilm\Microbe_HIP_Similarity", SM1)
    return SM1

def GIP_Calculate(M):
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    #计算gama
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)

    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    np.savetxt(r"data\ABiofilm\Microbe_GIP_Similarity", m)
    return m


def GIP_Calculate1(M):
    l=np.size(M,axis=0)
    sm=[]
    m=np.zeros((l,l))
    km=np.zeros((l,l))

    for i in range(l):
        tmp=(np.linalg.norm(M[i,:]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)

    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[i,:]-M[j,:]))**2))

    for i in range(l):
        for j in range(l):
            km[i,j]=1/(1+np.exp(-15*m[i,j]+math.log(9999)))
    np.savetxt(r"data\ABiofilm\Drug_GIP_Similarity.txt", km)
    return km

def Cosine_Sim(M):
    l=len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1=np.dot(M[i],M[j])
            v2=np.linalg.norm(M[i],ord=2)
            v3=np.linalg.norm(M[j],ord=2)
            if v2*v3==0:
                SM[i][j]=0
            else:
                SM[i][j]=v1/(v2*v3)

    return SM

def Drug_Chemical_Structure_Similarity(M):
    num_drugs = M.shape[0]
    drug_similarity = np.zeros((num_drugs, num_drugs))
    for i in range(num_drugs):
        for j in range(num_drugs):
            Di = M[i, :]
            Dj = M[j, :]
            numerator = np.dot(Di, Dj)
            denominator = np.sum(Di**2) + np.sum(Dj**2) - numerator
            if denominator != 0:
                drug_similarity[i, j] = numerator / denominator
            else:
                drug_similarity[i, j] = 0

    np.savetxt(r"data\ABiofilm\Drug_Chemical_Structure_Similarity.txt", drug_similarity)
    return drug_similarity


def Microbial_Genetic_Sequence_Similarity(M, gamma=0.1):
    microbe_features = M[:, :]
    microbe_similarity = rbf_kernel(microbe_features.T, gamma=gamma)
    return microbe_similarity




def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+sp.eye(adj.shape[0])
    return adj_normalized.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


