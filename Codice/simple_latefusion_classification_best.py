import platform
import time

import numpy as np
import pandas as pd
from IPython import display
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pyswarm import pso
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold

from lib import utils, pso_fitness


def classifier_loop(dbs, dataset_name, result_path, pat_filter, os_lab, under_sampling, over_sampling, standardize,
                    pca):
    t0 = time.time()

    if pat_filter:
        for i in range(len(dbs)):
            dbs[i] = dbs[i].loc[~dbs[i][0].isin(pat_filter), :]

    id_patients = dbs[0].iloc[:, 0]  # .to_list()

    num_patient = len(id_patients.unique())

    y_preds = []
    y_trues = []
    y_scores = []
    best_mod_occ = np.zeros((1, len(dbs))).astype(int)

    i = 0
    for idx in id_patients.unique():
        i = i + 1
        print(f"Patient {idx}, {i} of {num_patient}")
        display.clear_output(wait=True)

        id_patients_train = id_patients.unique().tolist()
        id_patients_train.remove(idx)

        scores_mod = np.empty((2, len(dbs)))
        for j in range(len(dbs)):
            X_test = dbs[j].loc[dbs[j][0] == idx, 1:dbs[j].shape[1] - 2]
            Y_test = dbs[j].loc[dbs[j][0] == idx, dbs[j].shape[1] - 1]

            clf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=100,
                                            max_features='auto', n_jobs=6)

            X_train = dbs[j].loc[dbs[j][0] != idx, 1:dbs[j].shape[1] - 2]
            Y_train = dbs[j].loc[dbs[j][0] != idx, dbs[j].shape[1] - 1]

            if under_sampling:
                rand_und_samp = RandomUnderSampler(sampling_strategy=under_sampling, random_state=42)
                X_train, Y_train = rand_und_samp.fit_resample(X_train, Y_train)
            if over_sampling:
                smote = SMOTE(sampling_strategy=1, random_state=42)
                X_train, Y_train = smote.fit_resample(X_train, Y_train)
            if standardize:
                stand_transformer = preprocessing.StandardScaler().fit(X_train)
                X_train = stand_transformer.transform(X_train)
                X_test = stand_transformer.transform(X_test)
            if pca:
                n_components = min(X_train.shape[0], X_train.shape[1])
                pca_transformer = PCA(n_components=n_components).fit(X_train)
                X_train = pca_transformer.transform(X_train)
                X_test = pca_transformer.transform(X_test)

            clf.fit(X_train, Y_train)

            scores_mod[:, j] = np.mean(clf.predict_proba(X_test), 0)
        
        #y_pred = np.argmax(scores_mod[:, np.argmax(np.abs(scores_mod[0, :]-scores_mod[1, :]))])
        idx_best_mod = np.argmax(np.abs(scores_mod[0, :]-scores_mod[1, :]))
        best_mod_occ[0, idx_best_mod] += 1
        #y_pred = np.argmax(np.bincount(np.argmax(scores_mod, 0)))
        #y_pred = np.argmax(np.mean(scores_mod, 1))
        #y_score = scores_mod[:, np.argmax(np.abs(scores_mod[0, :]-scores_mod[1, :]))][1]
        #y_score = np.mean(scores_mod, 1)[1]
        

        #y_preds.append(y_pred)
        #y_scores.append(y_score)
        #y_trues.append(Y_test.iloc[0])


    #cm = confusion_matrix(y_trues, y_preds)
    #tn, fp, fn, tp = cm.ravel()
    #auc = roc_auc_score(y_trues, y_scores)

    # acc = (tn + tp) / (tn + fp + fn + tp)

    #comp_time = time.time() - t0

    #row = [dataset_name, tp, fp, tn, fn, auc, comp_time]

    row = [dataset_name]+best_mod_occ.tolist()
    utils.append_list_as_row(result_path, row)


def main():
    if platform.system() == "Windows":
        pathomics_path = r"data/pathomics.csv"
        radiomics_path = r"data/radiomics.csv"
        semantic_path = r"data/semantic-features.csv"
        #result_path = r"results/simple_late_fusion.csv"
        piechart_path = r"results/simple_late_fusion_bestmod.csv"
    elif platform.system() == "Darwin":
        pathomics_path = "data/pathomics.csv"
        radiomics_path = "data/radiomics.csv"
        semantic_path = "data/semantic-features.csv"
        #result_path = "results/simple_late_fusion.csv"
        piechart_path = "results/simple_late_fusion_bestmod.csv"

    df_path = pd.read_csv(pathomics_path, header=None)
    #df_path = utils.aggregate(df_path)

    df_rad = pd.read_csv(radiomics_path)
    df_rad.dropna(inplace=True)
    df_rad = df_rad.iloc[:, [0] + [i for i in range(3, df_rad.shape[1])]]
    df_rad.columns = range(len(df_rad.columns))
    #df_rad = utils.aggregate(df_rad)

    df_sem = pd.read_csv(semantic_path)
    df_sem.columns = range(len(df_sem.columns))
    #df_sem = utils.aggregate(df_sem)

    path_patients = df_path.iloc[:, 0].unique().tolist()
    rad_patients = df_rad.iloc[:, 0].unique().tolist()
    sem_patients = df_sem.iloc[:, 0].unique().tolist()

    fuse_patients = utils.interset_list(path_patients, rad_patients, sem_patients)

    df_path = df_path.loc[df_path[0].isin(fuse_patients), :]
    df_rad = df_rad.loc[df_rad[0].isin(fuse_patients), :]
    df_sem = df_sem.loc[df_sem[0].isin(fuse_patients), :]

    datasets = [[df_path], [df_rad], [df_sem], [df_path, df_rad, df_sem], [df_path, df_sem], 
    [df_rad, df_sem], [df_path, df_rad]]

    datasets_name = ["pathomics", "radiomic", "semantic", "pathomics_radiomic_semantic", "pathomics_semantic", "radiomic_semantic", "pathomics_radiomic"]

    pat_filter = []

    i = 0
    for dataset, dataset_name in zip(datasets, datasets_name):
        classifier_loop(dbs=dataset, dataset_name=dataset_name, result_path=piechart_path, pat_filter=pat_filter,
                        os_lab=False, under_sampling=False, over_sampling=False, standardize=False, pca=False)
        i += 1


if __name__ == "__main__":
    main()
