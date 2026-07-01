import numpy as np
import pandas as pd

def diagonal_scatterm(PhibX):
    artSbx = np.matmul(np.transpose(PhibX), PhibX)
    
    eigVals, eigVecs = np.linalg.eigh(artSbx)
    
    eigVals = np.abs(eigVals)
    
    # Ignore zero eigenvalues
    maxEigVal = np.max(eigVals)
    zeroEigIndx = eigVals/maxEigVal<1e-6

    eigVals = np.delete(eigVals, zeroEigIndx, 0)
    eigVecs = np.delete(eigVecs, zeroEigIndx, 1)
    
    # Sort in descending order
    index = np.argsort(-1*eigVals)
    eigVals = eigVals[index]
    eigVecs = eigVecs[:,index]
    
    # Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
    SbxEigVecs = np.matmul(PhibX, eigVecs)
    
    # Normalize to unit length to create orthonormal eigenvectors for Sbx:
    cx = len(eigVals) # Rank of Sbx
    for i in range(cx):
        SbxEigVecs[:,i] = SbxEigVecs[:,i]/np.linalg.norm(SbxEigVecs[:,i])
        
    # Unitize the between-class scatter matrix (Sbx) for X
    SbxEigVals = np.diag(eigVals) # SbxEigVals is a (cx x cx) diagonal matrix
    Wbx = np.matmul(SbxEigVecs, eigVals**(-1/2)).reshape((SbxEigVecs.shape[0], -1)) # Wbx is a (p x cx) matrix which unitizes Sbx
    
    return cx, Wbx

def dcaFuse(X, Y, label):
    """
    DCAFUSE calculates the Discriminant Correlation Analysis (DCA) for
    feature-level fusion in multimodal systems.
      Inputs:
            X       :   pxn matrix containing the first set of training feature vectors
                        p:  dimensionality of the first feature set
                        n:  number of training samples

            Y       :   qxn matrix containing the second set of training feature vectors
                        q:  dimensionality of the second feature set

            label   :   1xn row vector of length n containing the class labels

      Outputs:
            Ax  :   Transformation matrix for the first data set (rxp)
                    r:  maximum dimensionality in the new subspace
            Ay  :   Transformation matrix for the second data set (rxq)
            Xs  :   First set of transformed feature vectors (rxn)
            Xy  :   Second set of transformed feature vectors (rxn)

      Sample use:
          Calculate the transformation matrices Ax and Ay and project the
          training data into the DCA subspace
        >> Ax, Ay, trainXdca, trainYdca = dcaFuse(trainX, trainY, label);

          Project the test data into the DCA subspace
        >> testXdca = Ax * testX;
        >> testYdca = Ay * testY;

          Fuse the two transformed feature matrices with either concatenation or summation:
          Fusion by concatenation (Z1)
        >> trainZ1 = [trainXdca ; trainYdca];
        >> testZ1  = [testXdca ; testYdca];

          Fusion by summation (Z2)
        >> trainZ2 = [trainXdca + trainYdca];
        >> testZ2  = [testXdca + testYdca];
    """
    # Compute mean vectors for each class and for all training data
    p, n = X.shape
    q = Y.shape[0]
    if Y.shape[1] != n:
        raise Warning('X and Y must have the same number of columns (samples).')
    elif len(label)!= n:
        raise Warning('The length of the label must be equal to the number of samples.')
    elif n == 1:
        raise Warning('X and Y must have more than one column (samples)')

    classes = np.unique(label)
    c = classes.size
    nSample = np.bincount(label)

    meanX = np.mean(X, 1).reshape([p, -1])
    meanY = np.mean(Y, 1).reshape([q, -1])

    for i in range(c):
        if i == 0:
            classMeanX = np.mean(X[:, label == classes[i]], 1).reshape([p, -1])
            classMeanY = np.mean(Y[:, label == classes[i]], 1).reshape([q, -1])
        else:
            classMeanX = np.hstack((classMeanX, np.mean(X[:, label == classes[i]], 1).reshape([p, -1])))
            classMeanY = np.hstack((classMeanY, np.mean(Y[:, label == classes[i]], 1).reshape([q, -1])))


    for i in range(c):
      if i == 0:
          PhibX = np.sqrt(nSample[i]) * (classMeanX[:,i].reshape(p, -1) - meanX)
          PhibY = np.sqrt(nSample[i]) * (classMeanY[:,i].reshape(q, -1) - meanY)
      else:
        PhibX = np.hstack((PhibX, (np.sqrt(nSample[i]) * (classMeanX[:,i].reshape(p, -1) - meanX))))
        PhibY = np.hstack((PhibY, (np.sqrt(nSample[i]) * (classMeanY[:,i].reshape(q, -1) - meanY))))

    # Diagolalize the between-class scatter matrix (Sb) for X
    cx, Wbx = diagonal_scatterm(PhibX)
    # Diagolalize the between-class scatter matrix (Sb) for Y
    cy, Wby = diagonal_scatterm(PhibY)

    ## Project data in a space, where the between-class scatter matrices are 
    # identity and the classes are separated
    r = min(cx, cy) # Maximum length of the desired feature vector
    Wbx = Wbx[:,0:r]
    Wby = Wby[:,0:r]

    Xp = np.matmul(np.transpose(Wbx), X)  # Transform X (pxn) to Xprime (rxn)
    Yp = np.matmul(np.transpose(Wby), Y)  # Transform Y (qxn) to Yprime (rxn)

    ## Unitize the between-set covariance matrix (Sxy)
    #  Note that Syx == Sxy'
    Sxy = np.matmul(Xp, np.transpose(Yp)) # Between-set covariance matrix

    Wcx, S, Wcy = np.linalg.svd(Sxy) # Singular Value Decomposition (SVD)

    Wcx = np.matmul(Wcx, S**(-1/2)) # Transformation matrix for Xp
    Wcy = np.matmul(Wcy, S**(-1/2)) # Transformation matrix for Yp

    Xs = np.matmul(np.transpose(Wcx), Xp).reshape((r, n)) # Transform Xprime to XStar
    Ys = np.matmul(np.transpose(Wcy), Yp).reshape((r, n)) # Transform Yprime to YStar

    Ax = np.matmul(np.transpose(Wcx), np.transpose(Wbx)).reshape((r, p)) # Final transformation Matrix of size (rxp) for X
    Ay = np.matmul(np.transpose(Wcy), np.transpose(Wby)).reshape((r, q)) # Final transformation Matrix of size (rxq) for Y
    
    return Ax, Ay, Xs, Ys

def dcaFusing(df1, df2):
    df_fuse = pd.merge(df1.iloc[:, 0:-1], df2, how='inner', on='ID_type')

    id_label = df_fuse.iloc[:, 0]
    X = np.transpose(df_fuse.iloc[:, 1:df1.shape[1]-1].to_numpy())
    Y = np.transpose(df_fuse.iloc[:, df1.shape[1]-1:-1].to_numpy())
    label = df_fuse.iloc[:, -1].to_numpy()

    Ax, Ay, trainXdca, trainYdca = dcaFuse(X, Y, label)
    trainX = np.vstack((trainXdca, trainYdca)) # Fusing the two transformed feature matrices
    trainX = np.transpose(trainX)
    X = pd.DataFrame(trainX)
    label = pd.DataFrame(label)

    df_fuse = pd.concat([id_label, X, label], axis=1)
    df_fuse.columns = df_fuse.columns.to_list()[:-1]+[df_fuse.columns.shape[0]-2]
    
    return df_fuse