import platform
import time

import numpy as np
import pandas as pd
from IPython import display
from imblearn.over_sampling import SMOTE
# from unbalanced_dataset.under_sampling import random_under_sampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc as AUC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from lib import utils


def classifier(db, dataset_name, pat_filter, exp, os_lab, under_sampling, over_sampling, standardize, pca):
    t0 = time.time()

    if pat_filter:
        db = db.loc[~db[0].isin(pat_filter), :]

    db.columns = range(db.shape[1])

    id_patients = db.iloc[:, 0]

    y_true = np.array
    y_true_pat = np.array
    y_pred = np.array
    y_pred_pat = np.array
    y_scores = np.array  # ([[0,0]])
    y_scores_pat = np.array  # ([[0,0]])

    num_patient = len(id_patients.unique())
    i = 0
    for idx in id_patients.unique():
        i = i + 1
        print(f"Patient {idx}, {i} of {num_patient}")
        display.clear_output(wait=True)

        clf = RandomForestClassifier(
            criterion='entropy', n_estimators=100, random_state=100, max_features='auto', n_jobs=6)

        ID_pat = db.loc[db[0] != idx, 0]
        X_train = db.loc[db[0] != idx, 1:db.shape[1] - 2]
        Y_train = db.loc[db[0] != idx, db.shape[1] - 1]
        Y_train_pat = db.loc[db[0] != idx, 0].tolist()
        X_test = db.loc[db[0] == idx, 1:db.shape[1] - 2]
        Y_test = db.loc[db[0] == idx, db.shape[1] - 1]
        if under_sampling:
            rand_und_samp = RandomUnderSampler(sampling_strategy=under_sampling, random_state=42)
            X_train, Y_train = rand_und_samp.fit_resample(X_train, Y_train)
        if over_sampling:
            smote = SMOTE(sampling_strategy=1, random_state=42)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
        if standardize:
            transformer = preprocessing.StandardScaler().fit(X_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)
        if pca:
            n_components = min(X_train.shape[0], X_train.shape[1])
            transformer = PCA(n_components=n_components)
            print(f"Numero di feature: {n_components}")
            transformer.fit(X_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        clf.fit(X_train, Y_train)

        y_true = np.append(y_true, Y_test)
        y_true_pat = np.append(y_true_pat, Y_test.tolist()[0])
        y_pred = np.append(y_pred, clf
                           .predict(X_test))
        y_pred_pat = np.append(y_pred_pat, np.argmax(
            np.bincount(clf.predict(X_test))))
        y_scores = np.append(y_scores, clf.predict_proba(X_test)[:, 1])
        y_scores_pat = np.append(y_scores_pat, np.mean(
            clf.predict_proba(X_test), 0)[1])

    y_true = y_true[1:].astype(int)
    y_true_pat = y_true_pat[1:].astype(int)
    y_pred = y_pred[1:].astype(int)
    y_pred_pat = y_pred_pat[1:].astype(int)
    y_scores = y_scores[1:]
    y_scores_pat = y_scores_pat[1:]

    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = cm.ravel()
    print(f"Per Patch tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
    print(f"AUC per Patch: {auc}")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # , pos_label=1)
    auc = AUC(fpr, tpr)
    print(f"AUC per Patch: {auc}")

    cm = confusion_matrix(y_true_pat, y_pred_pat)
    auc_pat = roc_auc_score(y_true_pat, y_scores_pat)
    tn_pat, fp_pat, fn_pat, tp_pat = cm.ravel()
    print(
        f"Per Patient tp: {tp_pat}, fp: {fp_pat}, tn: {tn_pat}, fn: {fn_pat}")
    print(f"AUC per Patient: {auc_pat}")

    fpr, tpr, thresholds = roc_curve(
        y_true_pat, y_scores_pat)  # , pos_label=1)
    auc = AUC(fpr, tpr)
    print(f"AUC per Patient: {auc}")

    comp_time = time.time() - t0

    row = [exp, dataset_name, tp, fp, tn, fn, auc,
           tp_pat, fp_pat, tn_pat, fn_pat, auc_pat, comp_time]

    return row


def main():
    if platform.system() == "Windows":
        pathomics_path = r"data/pathomics.csv"
        radiomics_path = r"data/radiomics.csv"
        semantic_path = r"data/semantic-features.csv"
        result_path = r"results/bootstrap_concatenation.csv"
    elif platform.system() == "Darwin":
        pathomics_path = "data/pathomics.csv"
        radiomics_path = "data/radiomics.csv"
        semantic_path = "data/semantic-features.csv"
        result_path = "results/bootstrap_concatenation.csv"

    df_path = pd.read_csv(pathomics_path, header=None)

    df_rad = pd.read_csv(radiomics_path)
    df_rad.dropna(inplace=True)
    df_rad = df_rad.iloc[:, [0] + [i for i in range(3, df_rad.shape[1])]]
    df_rad.columns = [i for i in range(len(df_rad.columns))]

    path_patients = df_path.iloc[:, 0].unique().tolist()
    rad_patients = df_rad.iloc[:, 0].unique().tolist()

    fuse_patients = utils.interset_list(path_patients, rad_patients, False)

    df_path = df_path.loc[df_path[0].isin(fuse_patients), :]
    df_rad = df_rad.loc[df_rad[0].isin(fuse_patients), :]

    path_dict = df_path[0].value_counts(sort=False).to_dict()
    rad_dict = df_rad[0].value_counts(sort=False).to_dict()
    final_dict = {key: rad_dict[key] if rad_dict[key] < path_dict[key] else path_dict[key] for key in rad_dict.keys()}

    rand_und_samp = RandomUnderSampler(sampling_strategy=final_dict, random_state=42)

    pat_filter = []

    num_exps = 10
    for i in range(num_exps):
        Y_train_pat = df_path.iloc[:, 0]
        df_path, Y_train_pat = rand_und_samp.fit_resample(df_path, Y_train_pat)

        Y_train_pat = df_rad.iloc[:, 0]
        df_rad, Y_train_pat = rand_und_samp.fit_resample(df_rad, Y_train_pat)

        df_path.sort_values([0], inplace=True)
        df_rad.sort_values([0], inplace=True)

        df_patrad = pd.concat([df_path.iloc[:, :df_path.shape[1]-1], df_rad.iloc[:, 1:]], axis=1)
        df_patrad.columns = range(len(df_patrad.columns))

        dataset = df_patrad
        dataset_name = "pathomics_radiomic"

        row = classifier(dataset, dataset_name=dataset_name, pat_filter=pat_filter, exp = i, os_lab=False, under_sampling=False,
                        over_sampling=False, standardize=False, pca=False)

        utils.append_list_as_row(result_path, row)


if __name__ == "__main__":
    main()
