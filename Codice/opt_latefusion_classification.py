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
from sklearn.metrics import confusion_matrix
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

    y_pred = []
    y_true = []

    i = 0
    for idx in id_patients.unique():
        i = i + 1
        print(f"Patient {idx}, {i} of {num_patient}")
        display.clear_output(wait=True)

        id_patients_train = id_patients.unique().tolist()
        id_patients_train.remove(idx)
        kf = KFold(n_splits=10)
        num_exps = kf.get_n_splits(id_patients_train)
        pso_weights = np.empty((num_exps, len(dbs)))

        y = 0
        for train_index, val_index in kf.split(id_patients_train):
            scores_val_mods = []
            for j in range(len(dbs)):
                X_test = dbs[j].loc[dbs[j][0] == idx, 1:dbs[j].shape[1] - 2]
                Y_test = dbs[j].loc[dbs[j][0] == idx, dbs[j].shape[1] - 1]

                train_pats = [id_patients_train[z] for z in train_index]

                clf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=100,
                                             max_features='auto', n_jobs=6)

                X_train = dbs[j].loc[dbs[j][0].isin(train_pats), 1:dbs[j].shape[1] - 2]
                Y_train = dbs[j].loc[dbs[j][0].isin(train_pats), dbs[j].shape[1] - 1]

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

                scores_val = np.empty((2, len(val_index)))
                y_pat_val_true = []
                for w in range(len(val_index)):
                    val_pat = id_patients_train[val_index[w]]

                    X_val = dbs[j].loc[dbs[j][0] == val_pat, 1:dbs[j].shape[1] - 2]
                    Y_val = dbs[j].loc[dbs[j][0] == val_pat, dbs[j].shape[1] - 1]

                    if standardize:
                        X_val = stand_transformer.transform(X_val)
                    if pca:
                        X_val = pca_transformer.transform(X_val)

                    scores_val[:, w] = np.mean(clf.predict_proba(X_val), 0)
                    y_pat_val_true.append(Y_val.iloc[0])

                scores_val_mods.append(scores_val)

            # PSO: trova w1, w2, w3 ottimi per il dato fold di validazione

            y_pat_val_true = np.asarray(y_pat_val_true)
            fitness_fun = pso_fitness.obj_fun(scores_val_mods, y_pat_val_true, "opt")
            # Set-up boundaries
            lb = [0, 0, 0]
            ub = [1, 1, 1]
            # Initialize optimizer
            # Perform optimization
            best_pos, _ = pso(fitness_fun.solve, lb, ub, swarmsize=100, omega=0.5,
                              phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8, debug=False)
            pso_weights[y, :] = best_pos
            y += 1

        # Trova w1, w2, w3 ottimi per il dato fold Leave-One Patient-Out
        # Addestra su tutto dataset di training e test su paziente di test.
        # Appendi w1, w2, w3 ottimi.

        best_weights_fold = np.mean(pso_weights, 0)  # best weight in the k-experiment.
        scores_test_mods = []
        for j in range(len(dbs)):
            X_test = dbs[j].loc[dbs[j][0] == idx, 1:dbs[j].shape[1] - 2]
            Y_test = dbs[j].loc[dbs[j][0] == idx, dbs[j].shape[1] - 1]

            clf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=100, max_features='auto',
                                         n_jobs=6)

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

            scores_test_mods.append(np.mean(clf.predict_proba(X_test), 0).reshape((2, 1)))

        y_pat_test_true = [Y_test.iloc[0]]
        fitness_fun = pso_fitness.obj_fun(scores_test_mods, y_pat_test_true, "pred")
        y_pred.append(fitness_fun.solve(best_weights_fold)[0])
        y_true.append(y_pat_test_true[0])

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # acc = (tn + tp) / (tn + fp + fn + tp)

    comp_time = time.time() - t0

    row = [dataset_name[0], tp, fp, tn, fn, comp_time]

    utils.append_list_as_row(result_path, row)


def main():
    if platform.system() == "Windows":
        pathomics_path = r"data/pathomics.csv"
        radiomics_path = r"data/radiomics.csv"
        semantic_path = r"data/semantic-features.csv"
        result_path = r"results/late_fusion.csv"
    elif platform.system() == "Darwin":
        pathomics_path = "data/pathomics.csv"
        radiomics_path = "data/radiomics.csv"
        semantic_path = "data/semantic-features.csv"
        result_path = "results/late_fusion.csv"

    df_path = pd.read_csv(pathomics_path, header=None)

    df_rad = pd.read_csv(radiomics_path)
    df_rad.dropna(inplace=True)
    df_rad = df_rad.iloc[:, [0] + [i for i in range(3, df_rad.shape[1])]]
    df_rad.columns = range(len(df_rad.columns))

    df_sem = pd.read_csv(semantic_path)
    df_sem.columns = range(len(df_sem.columns))

    path_patients = df_path.iloc[:, 0].unique().tolist()
    rad_patients = df_rad.iloc[:, 0].unique().tolist()
    sem_patients = df_sem.iloc[:, 0].unique().tolist()

    fuse_patients = utils.interset_list(path_patients, rad_patients, sem_patients)

    df_path = df_path.loc[df_path[0].isin(fuse_patients), :]
    df_rad = df_rad.loc[df_rad[0].isin(fuse_patients), :]
    df_sem = df_sem.loc[df_sem[0].isin(fuse_patients), :]

    datasets = [df_path, df_rad, df_sem]

    datasets_name = ["pathomics_radiomic_semantic"]

    pat_filter = []

    classifier_loop(dbs=datasets, dataset_name=datasets_name, result_path=result_path, pat_filter=pat_filter,
                    os_lab=False, under_sampling=False, over_sampling=False, standardize=False, pca=False)


if __name__ == "__main__":
    main()
