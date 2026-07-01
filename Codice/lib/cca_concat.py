from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd 

def pca_reduction(X):
    # Calculate the covariance matrix
    if X.shape[0] >= X.shape[1]:
        C = np.matmul(np.transpose(X), X) # pxp
    else:
        C = np.matmul(X, np.transpose(X)) # nxn
        
    # Perform eigenvalue decomposition
    eigVals, eigVecs = np.linalg.eig(C)
    eigVals = np.absolute(eigVals)
    
    # Ignore zero eigenvalues
    maxEigVal = np.max(eigVals)
    zeroEigIndx = eigVals/maxEigVal<1e-6

    eigVals = np.delete(eigVals, zeroEigIndx, 0)
    eigVecs = np.delete(eigVecs, zeroEigIndx, 1)
    
    # Sort in descending order
    index = np.argsort(-1*eigVals)
    eigVals = eigVals[index]
    eigVecs = eigVecs[:,index]
    
    # Obtain the projection matrix
    if X.shape[0] >= X.shape[1]:
        Wxpca = eigVecs
    else:
        Wxpca = np.matmul(np.matmul(np.transpose(X), eigVecs), np.diag(1/np.sqrt(eigVals)))
        
    # Update the train data
    X = np.matmul(X, Wxpca)

    return X, Wxpca

def ccaFuse(X, Y):
    """
    CCAFUSE applies feature level fusion using a method based on Canonical
    Correlation Analysis (CCA). 
    Feature fusion is the process of combining two feature vectors to obtain 
    a single feature vector, which is more discriminative than any of the 
    input feature vectors. 
    CCAFUSE gets the train and test data matrices from two modalities X & Y,
    and consolidates them into a single feature set Z.
    
    Inputs:
        trainX	:	nxp matrix containing the first set of training data
                    n:  number of training samples
                    p:  dimensionality of the first feature set
    
        trainY	:	nxq matrix containing the second set of training data
                    q:  dimensionality of the second feature set
    
        testX	:	mxp matrix containing the first set of test data
                    m:  number of test samples
    
        testY	:	mxq matrix containing the second set of test data
        
    Outputs:
        trainZ  :   matrix containing the fused training data    
    
    Sample use:
    trainX, trainY = ccaFuse(trainX, trainY);
    """

    n, p = X.shape
    q = Y.shape[1]
    if Y.shape[0] != n:
        raise Warning('X and Y must have the same number of columns (samples).')
    elif n == 1:
            raise Warning('X and Y must have more than one column (samples)')
            
    # Center the variables
    meanX = np.mean(X, 0).reshape((1, -1))
    meanY = np.mean(Y, 0).reshape((1, -1))
    X = X - meanX
    Y = Y - meanY

    # Dimensionality reduction using PCA for the first data X
    X, Wxpca = pca_reduction(X)
    # Dimensionality reduction using PCA for the first data Y
    Y, Wypca = pca_reduction(Y)

    # Fusion using Canonical Correlation Analysis (CCA)
    cca = CCA(n_components = min(np.linalg.matrix_rank(X), np.linalg.matrix_rank(Y)))
    X_trans, Y_trans = cca.fit_transform(X, Y)

    Wxcca = np.matmul(np.linalg.pinv(X), X_trans)
    Wycca = np.matmul(np.linalg.pinv(Y), Y_trans)

    return X_trans, Y_trans

def ccaFusing(df1, df2):
    df_fuse = pd.merge(df1.iloc[:, 0:-1], df2, how='inner', on='ID_type')

    id_label = df_fuse.iloc[:, 0]
    X = df_fuse.iloc[:, 1:df1.shape[1]-1].to_numpy()
    Y = df_fuse.iloc[:, df1.shape[1]-1:-1].to_numpy()
    label = df_fuse.iloc[:, -1].to_numpy()

    trainXcca, trainYcca = ccaFuse(X, Y)
    
    trainX = np.hstack((trainXcca, trainYcca)) # Fusing the two transformed feature matrices

    X = pd.DataFrame(trainX)
    label = pd.DataFrame(label)

    df_fuse = pd.concat([id_label, X, label], axis=1)
    df_fuse.columns = df_fuse.columns.to_list()[:-1]+[df_fuse.columns.shape[0]-2]
    
    return df_fuse