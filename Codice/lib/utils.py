from csv import writer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import auc as AUC

import pandas as pd
import numpy as np

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj, delimiter=',')
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def agregga_patch(df):
    df = df.groupby([0, len(df.columns) - 1], sort=False, as_index=False).mean()
    df = df[[i for i in range(len(df.columns))]]
    return df


def aggrega_crop(df):
    df1 = df.loc[:, "ID_type":"HIST_energy_around_maxRel"]
    df2 = df.loc[:, ["ID_type"] + list(df.columns[13:])]

    df1 = df1.groupby(["ID_type"], sort=False, as_index=False).mean()
    df2 = df2.groupby(["ID_type", "Adaptive"], sort=False, as_index=False).mean()
    df2 = df2[[df2.columns[0]] + list(df2.columns[2:]) + [df2.columns[1]]]

    df3 = pd.merge(df1, df2, how='left', on='ID_type')
    return df3

def aggregate(df):
    df = df.groupby([0, len(df.columns) - 1], sort=False, as_index=False).mean()
    df = df[[i for i in range(len(df.columns))]]

    return df

def interset_list(list1, list2, list3):
    list1 = set(list1)
    list2 = set(list2)

    if list3:
        list3 = set(list3)
        list_fin = list(list1 & list2 & list3)
    else:
        list_fin = list(list1 & list2)
        
    return list_fin


def classifier_performances(y_true, y_pred, y_scores, y_true_pat, y_pred_pat, y_scores_pat):
    cm = confusion_matrix(y_true, y_pred)
    #auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # , pos_label=1)
    auc = AUC(fpr, tpr)

    cm = confusion_matrix(y_true_pat, y_pred_pat)
    #auc_pat = roc_auc_score(y_true_pat, y_scores_pat)
    tn_pat, fp_pat, fn_pat, tp_pat = cm.ravel()

    fpr, tpr, thresholds = roc_curve(
        y_true_pat, y_scores_pat)  # , pos_label=1)
    auc_pat = AUC(fpr, tpr)

    return tn, fp, fn, tp, auc, tn_pat, fp_pat, fn_pat, tp_pat, auc_pat




def kronecker(df1, df2, df3=None):
    if df3 is kronecker.__defaults__[0]:
        columns = range((df1.shape[1]-2)*(df2.shape[1]-2)+2)
        df = pd.DataFrame(columns=columns)

        patients = df1.iloc[:, 0]
        labels = df1.iloc[:, -1]

        X1 = df1.iloc[:, 1:df1.shape[1]-1]
        X2 = df2.iloc[:, 1:df2.shape[1]-1]

        for i in range(X1.shape[0]):
            a = X1.iloc[i, :]
            b = X2.iloc[i, :]

            kron = np.kron(a, b)

            row = [patients.iloc[i]]
            row.extend(kron.tolist())
            row.append(labels.iloc[i])

            df_length = len(df)
            df.loc[df_length, :] = row
    else:
        dim1 = (df1.shape[1]-2)*(df2.shape[1]-2)
        dim2 = (df1.shape[1]-2)*(df3.shape[1]-2)
        dim3 = (df2.shape[1]-2)*(df3.shape[1]-2)
        dim4 = ((df1.shape[1]-2)*(df2.shape[1]-2))*(df3.shape[1]-2)
        columns = range(dim1+dim2+dim3+dim4+2)
        
        df = pd.DataFrame(columns=columns)

        patients = df1.iloc[:, 0]
        labels = df1.iloc[:, -1]

        X1 = df1.iloc[:, 1:df1.shape[1]-1]
        X2 = df2.iloc[:, 1:df2.shape[1]-1]
        X3 = df3.iloc[:, 1:df3.shape[1]-1]

        for i in range(X1.shape[0]):
            a = X1.iloc[i, :]
            b = X2.iloc[i, :]
            c = X3.iloc[i, :]

            kron1 = np.kron(a, b)
            kron2 = np.kron(a, c)
            kron3 = np.kron(b, c)
            kron4 = np.kron(kron1, c)
        
            row = [patients.iloc[i]]
            row.extend(kron1.tolist())
            row.extend(kron2.tolist())
            row.extend(kron3.tolist())
            row.extend(kron4.tolist())
            row.append(labels.iloc[i])

            df_length = len(df)
            df.loc[df_length, :] = row
        
    return df