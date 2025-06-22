
import sys
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import binarize
import torch_geometric.utils as utils

# Importing custom modules and model definitions
from Tom_New.similarity_methods import GIP_Calculate, GIP_Calculate1
from Tom_New.model2 import MMGNN
from Tom_New.data_utils import load_aBiofilm_data , RWR
from torch_geometric.utils import dense_to_sparse


import tensorflow as tf



import torch_geometric
from torch_geometric.data import Data
from Tom_New.HSNN import GNNStack


# Setting up arguments for training
def load_args():
    parser = argparse.ArgumentParser(
        description='New model on ABiofilm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--epoch1', type=int, default=150, help='Number of epochs for training.')
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--wd2', type=float, default=0.005)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=512, help='hidden dimensions.')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.01, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.0005, help='lamda.')
    parser.add_argument('--use_center_moment', action='store_true', help='whether to use center moment for MMGNN')
    parser.add_argument('--moment', type=int, default=3, help='max moment used in multi-moment model(MMGNN)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size for GAT layers.')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension for GAT layers.')
    parser.add_argument('--epoch2', type=int, default=150, help='Number of epochs for training.')
    parser.add_argument('--num_layers', type=int, default=3, help='layers of GraphSage model ')
    parser.add_argument('--model_type', type=str, default='GraphSage', help='Type of GNN model to use')
    parser.add_argument('--weight_decay', type=int, default=0.02, help='decay for 同质图')
    parser.add_argument('--dropout2', type=float, default=0.1, help='dropout rate for 同质图')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    return args


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(drug_graph, microbe_graph, heterogeneous_graph, drug_similarity_labels, microbe_similarity_labels):
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    if isinstance(drug_graph, np.ndarray):
        drug_graph = torch.FloatTensor(drug_graph).to(device)
    if isinstance(microbe_graph, np.ndarray):
        microbe_graph = torch.FloatTensor(microbe_graph).to(device)
    if isinstance(heterogeneous_graph, np.ndarray):
        heterogeneous_graph = torch.FloatTensor(heterogeneous_graph).to(device)

    drug_features = torch.randn((drug_graph.shape[0], args.dim_hidden)).to(device)
    microbe_features = torch.randn((microbe_graph.shape[0], args.dim_hidden)).to(device)
    heterogeneous_features = torch.randn((heterogeneous_graph.shape[0], args.dim_hidden)).to(device)

    drug_model = GNNStack(input_dim=args.dim_hidden, hidden_dim=args.hidden_dim, output_dim=args.output_dim, args=args).to(device)
    microbe_model = GNNStack(input_dim=args.dim_hidden, hidden_dim=args.hidden_dim, output_dim=args.output_dim, args=args).to(device)

    edge_index_drug, _ = dense_to_sparse(drug_graph)
    edge_index_microbe, _ = dense_to_sparse(microbe_graph)
    loss_fn = nn.MSELoss()

    optimizer_drug = optim.AdamW(drug_model.parameters(), lr=0.0005, weight_decay=args.weight_decay)
    optimizer_microbe = optim.AdamW(microbe_model.parameters(), lr=0.0005, weight_decay=args.weight_decay)

    scheduler_drug = optim.lr_scheduler.ReduceLROnPlateau(optimizer_drug, mode='min', patience=10, factor=0.5,
                                                          verbose=True)
    scheduler_microbe = optim.lr_scheduler.ReduceLROnPlateau(optimizer_microbe, mode='min', patience=10, factor=0.5,
                                                             verbose=True)

    patience = args.patience
    best_loss_drug = float('inf')
    best_loss_microbe = float('inf')
    patience_counter_drug = 0
    patience_counter_microbe = 0

    dropout1 = nn.Dropout(p=args.dropout2)
    dropout = nn.Dropout(p=args.dropout)
    for epoch in range(args.epoch2):
        drug_model.train()
        microbe_model.train()
        optimizer_drug.zero_grad()
        drug_embedding = drug_model(drug_features, edge_index_drug)
        drug_embedding = dropout1(drug_embedding)
        target_drug = torch.FloatTensor(drug_similarity_labels).to(device)
        target_drug = target_drug / target_drug.max()
        drug_similarity = drug_embedding @ drug_embedding.T
        drug_similarity = drug_similarity / drug_similarity.max()
        loss_drug = loss_fn(drug_similarity, target_drug)
        loss_drug.backward()
        optimizer_drug.step()
        scheduler_drug.step(loss_drug)
        if loss_drug.item() < best_loss_drug:
            best_loss_drug = loss_drug.item()
            patience_counter_drug = 0
        else:
            patience_counter_drug += 1
            if patience_counter_drug >= patience:
                print(f"Early stopping drug model at epoch {epoch}")
                break


        optimizer_microbe.zero_grad()

        microbe_embedding = microbe_model(microbe_features, edge_index_microbe)
        microbe_embedding = dropout1(microbe_embedding)

        target_microbe = torch.FloatTensor(microbe_similarity_labels).to(device)
        target_microbe = target_microbe / target_microbe.max()
        microbe_similarity = microbe_embedding @ microbe_embedding.T
        microbe_similarity = microbe_similarity / microbe_similarity.max()
        loss_microbe = loss_fn(microbe_similarity, target_microbe)
        loss_microbe.backward()
        optimizer_microbe.step()
        scheduler_microbe.step(loss_microbe)
        if loss_microbe.item() < best_loss_microbe:
            best_loss_microbe = loss_microbe.item()
            patience_counter_microbe = 0
        else:
            patience_counter_microbe += 1
            if patience_counter_microbe >= patience:
                print(f"Early stopping microbe model at epoch {epoch}")
                break
    heterogeneous_model = MMGNN(
        nfeat=args.dim_hidden,
        nlayers=args.layer,
        nhidden=args.dim_hidden,
        nclass=1860,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        use_center_moment=args.use_center_moment,
        moment=args.moment
    ).to(device)

    optimizer_heterogeneous = optim.AdamW(heterogeneous_model.parameters(), lr=0.001, weight_decay=args.wd2)
    scheduler_heterogeneous = optim.lr_scheduler.CosineAnnealingLR(optimizer_heterogeneous, T_max=args.epoch1, eta_min=0.00001)

    for epoch in range(args.epoch1):
        heterogeneous_model.train()
        optimizer_heterogeneous.zero_grad()
        heterogeneous_output = heterogeneous_model(heterogeneous_features, heterogeneous_graph)
        heterogeneous_output = dropout(heterogeneous_output)
        loss_heterogeneous = torch.mean((heterogeneous_output - heterogeneous_features) ** 2)
        loss_heterogeneous.backward()
        optimizer_heterogeneous.step()
        scheduler_heterogeneous.step()

    drug_similarity = torch.mm(drug_embedding, drug_embedding.T)
    microbe_similarity = torch.mm(microbe_embedding, microbe_embedding.T)
    heterogeneous_similarity = torch.mm(heterogeneous_output, heterogeneous_output.T)

    microbe_padded = torch.zeros((1860, 1860), device=device)
    microbe_padded[-140:, -140:] = microbe_similarity

    drug_padded = torch.zeros((1860, 1860), device=device)
    drug_padded[:1720, :1720] = drug_similarity

    w1 = torch.nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)
    w2 = torch.nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)
    w3 = torch.nn.Parameter(torch.tensor(0.4, device=device), requires_grad=True)

    w1 = torch.nn.functional.relu(w1)
    w2 = torch.nn.functional.relu(w2)
    w3 = torch.nn.functional.relu(w3)

    updated_heterogeneous = w1 * drug_padded + w2 * microbe_padded + w3 * heterogeneous_similarity

    return updated_heterogeneous.detach().cpu().numpy()











