import platform
import time

import numpy as np
import pandas as pd
from IPython import display
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from lib import utils


def classifier(db, dataset_name, result_path, pat_filter, os_lab, under_sampling, over_sampling, standardize, pca):
    df_path = db[0]
    df_rad = db[1]

    if pat_filter:
        df_path = df_path.loc[~df_path[0].isin(pat_filter), :]
        df_rad = df_rad.loc[~df_rad[0].isin(pat_filter), :]

    # db.columns = range(db.shape[1])

    id_patients = df_path.iloc[:, 0]  # .to_list()

    num_patient = len(id_patients.unique())
    n_bootstraps = 10
    for j in range(n_bootstraps):
        t0 = time.time()
        j = j + 1
        print(f"{dataset_name}: Bootstrap {j} of {n_bootstraps}")

        y_true = np.array
        y_true_pat = np.array
        y_pred = np.array
        y_pred_pat = np.array
        y_scores = np.array  # ([[0,0]])
        y_scores_pat = np.array  # ([[0,0]])

        i = 0
        for idx in id_patients.unique():
            i = i + 1
            print(f"Patient {idx}, {i} of {num_patient}")
            display.clear_output(wait=True)

            X_test1 = df_path.loc[df_path[0] == idx, :]
            X_test1 = utils.aggregate(X_test1)

            X_test2 = df_rad.loc[df_rad[0] == idx, :]
            X_test2 = utils.aggregate(X_test2)

            X_test3 = pd.merge(X_test1.iloc[:, 0:-1], X_test2, how='inner', on=0)
            X_test3.columns = range(len(X_test3.columns))

            X_test = X_test3.loc[:, 1:X_test3.shape[1] - 2]
            Y_test = X_test3.loc[:, X_test3.shape[1] - 1]

            df_path_bot = df_path.groupby(0).apply(
                lambda group_df: group_df.sample(frac=0.5, replace=True)).reset_index(drop=True)
            df_path_bot = df_path_bot.drop_duplicates().reset_index(drop=True)
            df_path_bot = utils.aggregate(df_path_bot)

            df_rad_bot = df_rad.groupby(0).apply(lambda group_df: group_df.sample(n=10, replace=True)).reset_index(
                drop=True)
            df_rad_bot = df_rad_bot.drop_duplicates().reset_index(drop=True)
            df_rad_bot = utils.aggregate(df_rad_bot)

            df_patrad = pd.merge(df_path_bot.iloc[:, 0:-1], df_rad_bot, how='inner', on=0)
            df_patrad.columns = range(len(df_patrad.columns))

            clf = RandomForestClassifier(
                criterion='entropy', n_estimators=100, random_state=100, max_features='auto', n_jobs=6)

            X_train = df_patrad.loc[df_patrad[0] != idx, 1:df_patrad.shape[1] - 2]
            Y_train = df_patrad.loc[df_patrad[0] != idx, df_patrad.shape[1] - 1]

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

        tn, fp, fn, tp, auc, tn_pat, fp_pat, fn_pat, tp_pat, auc_pat = utils.classifier_performances(y_true, y_pred,
                                                                                                     y_scores,
                                                                                                     y_true_pat,
                                                                                                     y_pred_pat,
                                                                                                     y_scores_pat)

        accuracy = (tn + tp) / (tn + fp + fn + tp)

        comp_time = time.time() - t0

        row = [dataset_name, j, tp, fp, tn, fn, auc,
               tp_pat, fp_pat, tn_pat, fn_pat, auc_pat, comp_time]

        utils.append_list_as_row(result_path, row)


def main():
    if platform.system() == "Windows":
        pathomics_path = r"data/pathomics.csv"
        radiomics_path = r"data/radiomics.csv"
        semantic_path = r"data/semantic-features.csv"
        result_path = r"results/simpleconcat_bootstrap_classification.csv"
    elif platform.system() == "Darwin":
        pathomics_path = "data/pathomics.csv"
        radiomics_path = "data/radiomics.csv"
        semantic_path = "data/semantic-features.csv"
        result_path = "results/simpleconcat_bootstrap_classification.csv"

    df_path = pd.read_csv(pathomics_path, header=None)

    df_rad = pd.read_csv(radiomics_path)
    df_rad.dropna(inplace=True)
    df_rad = df_rad.iloc[:, [0] + [i for i in range(3, df_rad.shape[1])]]
    df_rad.columns = [i for i in range(len(df_rad.columns))]

    df_sem = pd.read_csv(semantic_path)
    df_sem.columns = range(len(df_sem.columns))

    path_patients = df_path.iloc[:, 0].unique().tolist()
    rad_patients = df_rad.iloc[:, 0].unique().tolist()
    sem_patients = df_sem.iloc[:, 0].unique().tolist()

    fuse_patients = utils.interset_list(rad_patients, sem_patients, list3=None)

    df_path = df_path.loc[df_path[0].isin(fuse_patients), :]
    df_rad = df_rad.loc[df_rad[0].isin(fuse_patients), :]
    df_sem = df_sem.loc[df_sem[0].isin(fuse_patients), :]

    # datasets = [df_path, df_rad, df_sem]
    datasets = [df_rad, df_sem]
    # datasets_name = ["pathomics", "radiomics", "semantic"]
    datasets_name = "radiomics_semantic"

    pat_filter = []

    classifier(db=datasets, dataset_name=datasets_name, result_path=result_path, pat_filter=pat_filter, os_lab=False,
               under_sampling=False, over_sampling=False, standardize=False, pca=False)


if __name__ == "__main__":
    main()
