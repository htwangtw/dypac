import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

def get_transform_matrices(dypac, data, use_lasso=False):
    inv_transform_mat = dypac.components_
    embedded_data = dypac.transform_sparse(data)
    masked_data = dypac.masker_.transform(data)
    if use_lasso:
        reg = Lasso().fit(masked_data, embedded_data)
    else:
        reg = LinearRegression().fit(masked_data, embedded_data)
    transform_mat = np.transpose(reg.coef_) # TODO save transform_mat as sparse matrix
    return transform_mat, inv_transform_mat

class Embedding:
    def __init__(self, dypac, data):
        self.size = dypac.components_.shape[0]
        self.transform_mat, self.inv_transform_mat = get_transform_matrices(dypac, data)

    def transform(self, data):
        return np.matmul(data, self.transform_mat)

    def inv_transform(self, data):
        return data * self.inv_transform_mat

class TrainingEmbbedings:
    def __init__(self, dypac_list, data):
        if not dypac_list:
            raise ValueError('Input dypac_list must not be empty.')
        self.masker_ = dypac_list[0].masker_
        self.embedding_list = []
        for dypac in dypac_list:
            self.embedding_list.append(Embedding(dypac, data))
