import torch
import pandas as pd
import numpy as np

def aggregate_embeddings(Xembeddings, enhancer_numbers, y, tissues, agg_method):
    y = np.array(y)
    unique_elements, element_indices = np.unique(enhancer_numbers, return_inverse=True)
    
    num_features = Xembeddings.shape[1]
    num_unique_elements = len(unique_elements)
    
    Xembeddings_agg = np.zeros((num_unique_elements, num_features))
    y_agg = []
    tissues_y_agg = []

    for i, _ in enumerate(unique_elements):
        idxs = np.where(element_indices == i)[0]
        subset_X = Xembeddings[idxs, :]
        
        if agg_method == 'max':
            subset_X = subset_X[np.abs(subset_X).argmax(axis=0), np.arange(subset_X.shape[1])]
        elif agg_method == 'mean':
            subset_X = torch.mean(subset_X, axis=0)
                
        Xembeddings_agg[i, :] = subset_X
        y_agg.append(y[idxs][0])

        tissues_y_agg.append([tissues[i] for i in idxs][0])

    Xembeddings_agg = torch.tensor(Xembeddings_agg)
    return Xembeddings_agg, y_agg, tissues_y_agg, unique_elements.tolist()


