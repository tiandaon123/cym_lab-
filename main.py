import random
from sklearn.metrics import precision_recall_curve
from similarity_methods import *
from DSVM import *
from demo1 import *
from sklearn.metrics import accuracy_score, f1_score,matthews_corrcoef
from data_utils import load_aBiofilm_data, RWR, GIP_Calculate, GIP_Calculate1,Multi_View_RWR_with_Weights






A, mis, drs, known, unknown, labels = load_aBiofilm_data()


temp_label = np.zeros((1720, 140))
for temp in labels:
    temp_label[int(temp[0]) - 1, int(temp[1]) - 1] = int(temp[2])
labels = temp_label



# DS = np.loadtxt( './data/ABiofilm/Drug_structural.txt')
DS = np.loadtxt('data/ABiofilm/drug_structure2.txt')
DCS = Drug_Chemical_Structure_Similarity(A)
MF = np.loadtxt('data/ABiofilm/microbe_function2.txt')
# MF = np.loadtxt( './data/ABiofilm/Microbe_functional.txt')
MDS = Microbial_Genetic_Sequence_Similarity(A)


def kfold_5(num):
    auc_list = []
    aupr_list = []
    k = list(range(len(known)))
    unk = list(range(len(unknown)))
    random.shuffle(k)
    random.shuffle(unk)

    num_test = int(np.floor(labels.shape[0] / 5))
    num_train = labels.shape[0] - num_test
    all_index = list(range(labels.shape[0]))
    np.random.shuffle(all_index)
    for cv in range(1, 6):
        # Update interaction matrix
        interaction = np.array(list(A))
        if cv < 5:
            B1 = known[k[(cv - 1) * (len(known) // 5):(len(known) // 5) * cv], :]
            B2 = unknown[unk[(cv - 1) * (len(unknown) // 5):(len(unknown) // 5) * cv], :]
            for i in range(len(known) // 5):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1 = known[k[(cv - 1) * (len(known) // 5):], :]
            B2 = unknown[unk[(cv - 1) * (len(unknown) // 5):], :]
            for i in range(len(known) - (len(known) // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0

        W_views_drug = [DS, DCS]
        W_views_microbe = [MF, MDS]
        drug_homogeneous_graph = torch.FloatTensor(np.array(Multi_View_RWR_with_Weights(W_views_drug, alpha=0.1, beta=0.05, max_iter=20)))
        microbe_homogeneous_graph = torch.FloatTensor(np.array(Multi_View_RWR_with_Weights(W_views_microbe, alpha=0.1, beta=0.05, max_iter=20)))
        drug_microbe_heterogeneous_graph = construct_network(A, mis, drs)
        drug_similarity_labels = torch.FloatTensor(np.array(Multi_View_RWR_with_Weights(W_views_drug, alpha=0.1, beta=0.05, max_iter=20)))
        microbe_similarity_labels = torch.FloatTensor(np.array(Multi_View_RWR_with_Weights(W_views_microbe, alpha=0.1, beta=0.05, max_iter=20)))
        F = train(drug_homogeneous_graph, microbe_homogeneous_graph, drug_microbe_heterogeneous_graph,drug_similarity_labels, microbe_similarity_labels)

        train_index = all_index[:num_train]
        test_index = all_index[num_train:(num_train + num_test)]
        Y_train = labels[train_index]
        Y_test = labels[test_index]
        X_train = F[train_index]
        X_test = F[test_index]

        y_score = train_dual_svm(X_train, Y_train, X_test, Y_test)
        y_true = Y_test.flatten()
        y_pred = np.array(y_score).flatten()
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_binary = (y_pred > best_threshold).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        F1_score = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = auc(fpr, tpr)
        auc_list.append(auc_val)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)
        aupr_list.append(aupr)






for i in range(5):
    kfold_5(i)
    print("------------------------------")

