import numpy as np
import math
from sklearn.metrics import roc_curve,auc
import sklearn.metrics
def kappa(matrix):
    n = np.sum(matrix)
    n1= np.sum(n)
    sum_po = 0
    sum_pe = 0
    for i in range(matrix.shape[0]):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n1
    pe = sum_pe / (n1 * n1)
    return (po - pe) / (1 - pe)
def MCC(TP,FP,TN,FN):
    up = (TP*TN)-(FP*FN)
    do = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    nod = math.sqrt(do)
    if((up == 0) | (nod == 0)):
        mcc = 0
    else:
        mcc = up/nod
    return mcc
def compute_auc(t,s,le):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False,pos_label=le)
    return auc(fpr, tpr)
def getResStandard(y_true,y_pred,y_pred_pro,ma,le):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true1 = [x.astype(int) for x in y_true]
    y_pred2 = [x.astype(int) for x in y_pred]
    if(ma>le):
        index_ma = 1
        index_le = 0
    else:
        index_ma = 0
        index_le = 1
    y_pred_proba = np.array(y_pred_pro)[:,1]
    # for ipb in range(y_pred_pro.shape[0]):
    #     if(y_true[ipb]==ma):
    #         y_pred_proba.append(y_pred_pro[ipb,index_ma])
    #     else:
    #         y_pred_proba.append(y_pred_pro[ipb,index_le])
    y_pred_proba = np.array(y_pred_proba)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(y_true.shape[0]):
        if((y_true[i]==le)&(y_pred[i]==le)):
            TP += 1
        elif((y_true[i]==le)&(y_pred[i]==ma)):
            FN += 1
        elif((y_true[i]==ma)&(y_pred[i]==le)):
            FP += 1
        elif((y_true[i]==ma)&(y_pred[i]==ma)):
            TN += 1
    if (TP == 0):
        TPR = 0
    else:
        TPR = TP / (TP + FN)
    if (TN == 0):
        TNR = 0
    else:
        TNR = TN / (TN + FP)
    MA = [[TP, FP], [FN, TN]]
    MA = np.array(MA)
    GM = math.sqrt(TPR*TNR)
    if(TP == 0| FP == 0 ):
        pc = 0.0
    else:
        pc = TP /(TP+FP)
    if( (pc == 0.0) | (TPR == 0.0) ):
        F1 = 0.0
    else:
        F1 = (2*pc*TPR)/(pc+TPR)
    Kappa = kappa(MA)
    mc=MCC(TP,FP,TN,FN)
    # y_true = [i-1 for i in y_true]
    # y_pred = [i-1 for i in y_pred]
    auc1 = compute_auc(y_true1,y_pred2,ma)
    # auc2= sklearn.metrics.roc_auc_score(y_true,y_pred_proba)
    y_pred_proba = np.array(y_pred_pro)[:, index_le]
    preci,recall,_thresholds = sklearn.metrics.precision_recall_curve(y_true,y_pred_proba,pos_label=le)
    Aupr = auc(recall,preci)
    return GM,F1,Kappa,auc1,Aupr,mc