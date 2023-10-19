import numpy as np
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from typing import overload, Tuple

# possible classes for data whitening
class Whitenings(Enum):
    NONE = 0
    ZCA = 1
    PCA = 2
    ZCA_COR = 3

class WhitenTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, whiten : Whitenings = Whitenings.ZCA_COR):
        """Initializes Transformer with the specified whitening

        Args:
            whiten (Whitenings, optional): whitening method chosen (possible options: `Whitenings`). Defaults to Whitenings.ZCA_COR.
        """
        self.whiten = whiten
        self.W = None
        self.X_mean = None

    def fit(self, X : np.array, y : np.array = None):
        """Generate Whitening / Sphering Matrix and X mean
        
        Method is descibed in https://arxiv.org/pdf/1512.00809.pdf

        Args:
            X (np.array): matrix to be whitened
            y (np.array, optional): ignored. Defaults to None

        Raises:
            Exception: Non existent Whitening will raise an error

        Returns:
            WhitenTransformer: the transformer filled with W and X_mean
        """
        # numerical stability
        eps = 1e-8
        
        # normalize data to have mean 0
        X_mean = X.mean(axis=0)
        X_ = X - X_mean
        
        if self.whiten == Whitenings.NONE:
            self.X_mean = X_mean
            self.W = np.identity(X.shape[1])
            return self

        sigma = np.cov(X_.T)
        
        if self.whiten in [Whitenings.PCA, Whitenings.ZCA]:

            # compute $$U \Lambda U^T = \Sigma$$
            U, lam, _ = np.linalg.svd(sigma)
            lam_sqrt_inv = np.diag(1.0 / np.sqrt(lam + eps))
            
            # Compute $$\Sigma^{-1/2}$$
            sigma_sqrt_inv = np.matmul(np.matmul(U, lam_sqrt_inv),U.T)

            if self.whiten == Whitenings.ZCA:
                W = sigma_sqrt_inv
            elif self.whiten == Whitenings.PCA:
                W = np.matmul(lam_sqrt_inv, U.T)
        elif self.whiten == Whitenings.ZCA_COR:
            vars = np.var(X_,axis=0) # same as np.diagonal(X)
            V_sqrt_inv = np.diag(1.0 / (np.sqrt(vars) + eps))
            P = np.matmul(np.matmul(V_sqrt_inv, sigma), V_sqrt_inv)
            G, theta, _ = np.linalg.svd(P)
            theta_sqrt_inv = np.diag(1.0 / np.sqrt(theta + eps))
            P_sqrt_inv = np.matmul(np.matmul(G, theta_sqrt_inv),G.T)

            W = np.matmul(P_sqrt_inv, V_sqrt_inv)
        else:
            raise Exception('Whitening method not found.')

        self.X_mean = X_mean
        self.W = W
        return self
    
    def transform(self, X : np.array, y : np.array = None) -> np.array:
        """Whiten matrix X given the whitening method
        
        Method is descibed in https://arxiv.org/pdf/1512.00809.pdf

        Args:
            X (np.array): matrix to be whitened
            y (np.array, optional): ignored. Defaults to None.

        Returns:
            np.array: The whitened matrix
        """
        if self.X_mean is None:
            raise Exception('No mean is found. Try to fit before transform.')
        if self.W is None:
            raise Exception('No whitening matrix is found. Try to fit before transform.')
        return np.matmul(X - self.X_mean, self.W.T)
