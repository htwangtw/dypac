import numpy as np
from sklearn.preprocessing import StandardScaler


def miss_constant(X, precision=1e-10):
    """Check if a constant vector is missing in a vector basis."""
    return np.min(np.sum(np.absolute(X - 1), axis=1)) > precision


class Embedding:
    def __init__(self, X, add_constant=True):
        """
        Transformation to and from an embedding.

        Parameters
        ----------
        X: ndarray
            The vector basis defining the embedding (each row is a vector).

        add_constant: boolean
            Add a constant vector to the vector basis, if none is present.

        Attributes
        ----------
        size: int
            the number of vectors defining the embedding, not including the
            intercept.

        transform_mat: ndarray
            matrix projection from original to embedding space.

        inverse_transform_mat: ndarray
            matrix projection from embedding to original space.

        """
        self.size = X.shape[0]
        # Once we have the embedded representation beta, the inverse transform
        # is a simple linear mixture:
        # Y_hat = beta * X
        # We store X as the inverse transform matrix
        if add_constant and miss_constant(X):
            self.inverse_transform_mat = np.concatenate([np.ones([1, X.shape[1]]), X])
        else:
            self.inverse_transform_mat = X
        # The embedded representation beta is also derived by a simple linear
        # mixture Y * P, where P is the pseudo-inverse of X
        # We store P as our transform matrix
        self.transform_mat = np.linalg.pinv(self.inverse_transform_mat)

    def transform(self, data):
        """Project data in embedding space."""
        # Given Y, we get
        # beta = Y * P
        return np.matmul(data, self.transform_mat)

    def inverse_transform(self, embedded_data):
        """Project embedded data back to original space."""
        # Given beta, we get:
        # Y_hat = beta * X
        return np.matmul(embedded_data, self.inverse_transform_mat)

    def compress(self, data):
        """Embedding compression of data in original space."""
        # Given Y, by combining transform and inverse_transform, we get:
        # Y_hat = Y * P * X
        return self.inverse_transform(self.transform(data))

    def score(self, data):
        """Average residual squares after compress in embedding space."""
        # The R2 score is only interpretable for standardized data
        data = StandardScaler().fit_transform(data)
        return 1 - np.var(data - self.compress(data), axis=0)
