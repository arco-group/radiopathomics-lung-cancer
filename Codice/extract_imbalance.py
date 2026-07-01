import sys

import pandas as pd
from IPython import display
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from lib import utils

def extract_score(db, result_path, pat_filter, under_sampling, over_sampling, standardize, pca):
    if pat_filter:
        db = db.loc[~db[0].isin(pat_filter), :]

    id_patients = db.iloc[:, 0]
    num_patient = len(id_patients.unique())

    i = 0
    for idx in id_patients.unique():
        i = i + 1
        print(f"Patient {idx}, {i} of {num_patient}")
        display.clear_output(wait=True)

        clf = RandomForestClassifier(
            criterion='entropy', n_estimators=100, random_state=100, max_features='auto', n_jobs=6)

        X_train = db.loc[db[0] != idx, 1:db.shape[1] - 2]
        Y_train = db.loc[db[0] != idx, db.shape[1] - 1]
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

        y_true_pat = Y_test.tolist()[0]
        y_scores = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        for j in range(y_scores.shape[0]):
            row = [idx] + y_scores[j].tolist() + [y_pred[j]] + [y_true_pat]
            utils.append_list_as_row(result_path, row)


if __name__ == "__main__":
    if sys.platform == "darwin":
        pathomics_path = "data/pathomics.csv"
        result_path = "results/imbalance.csv"
    else:
        pathomics_path = r"data/pathomics.csv"
        result_path = r"results/imbalance.csv"

    pat_filter = []

    df = pd.read_csv(pathomics_path, header=None)

    extract_score(db=df, result_path=result_path, pat_filter=pat_filter, under_sampling=False, over_sampling=False,
                  standardize=False, pca=False)
