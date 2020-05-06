import numpy as np

def projector(X):
    """Ordinary least-square projection."""
    # when solving Y = beta * X + E, for beta minimizing sum of E squares,
    # beta takes the form of Y * P, with P the projection matrix into the space
    # spanned by X. The following formula computes P.
    return np.dot(X.transpose(), np.pinv(np.dot(X, X.transpose())))


def miss_constant(X, precision=1e-10):
    """Check if a constant vector is missing in a vector basis.
    """
    return np.min(np.sum(np.absolute(X-1), axis=1)) > precision


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
        self.size = dypac.components_.shape[0]
        # Once we have the embedded representation beta, the inverse transform
        # is a simple linear mixture:
        # Y_hat = beta * X
        # We store X as the inverse transform matrix
        if add_constant and miss_constant(X):
            self.inv_transform_mat = np.concat(np.ones([1, X.shape[1]]), X)
        else:
            self.inv_transform_mat = X
        # The embedded representation beta is also derived by a simple linear
        # mixture Y * P, where P is defined in `projector`
        # We store P as our transform matrix
        self.transform_mat = projector(self.inv_transform_mat)

    def transform(self, data):
        """Project data in embedding space."""
        # Given Y, we get
        # beta = Y * P
        return np.matmul(data, self.transform_mat)

    def inv_transform(self, embedded_data):
        """Project embedded data back to original space."""
        # Given beta, we get:
        # Y_hat = beta * X
        return np.matmul(embedded_data, self.inv_transform_mat)

    def compression(self, data):
        """embedding compression of data in original space."""
        # Given Y, by combining transform and inverse_transform, we get:
        # Y_hat = Y * P * X
        return self.inv_transform(self.transform(data))

    def score(self, data):
        """Average residual squares after compression in embedding space."""
        # Given Y, compute || Y - Y_hat ||^2
        return np.sum(np.square(data - self.transform(data)))
