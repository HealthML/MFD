import numpy as np
def regress_out_X(Y, X=None, Xdagger=None):
    """
    
    Y: the features, shape = (n_observations, n_features)
    
    X: the variables to regress out, shape = (n_observations, n_variables)
       e.g., if we only remove a single variable, e.g., GC-content, the shape is (n_observations, 1)
       
    Xdagger: ignore this for now...

    To get the variables with the influence of GC-content removed:
        
        Y_out, _ = regress_out_X(Y, X)
    
    """
    if X is None:
        RxY = Y - Y.mean(0)
        return RxY, None
    else:
        if Xdagger is None:
            Xdagger = np.linalg.pinv(X)
        RxY = Y - X.dot(Xdagger.dot(Y))
        return RxY, Xdagger