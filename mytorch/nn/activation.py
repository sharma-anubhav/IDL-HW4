import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        
        dim = self.dim if self.dim >= 0 else len(Z.shape) + self.dim
        
        self.original_shape = Z.shape
        
        Z_max = np.max(Z, axis=dim, keepdims=True)
        Z_shifted = Z - Z_max
        
        exp_Z = np.exp(Z_shifted)
        
        sum_exp = np.sum(exp_Z, axis=dim, keepdims=True)
        
        self.A = exp_Z / sum_exp
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        shape = self.A.shape
        dim = self.dim if self.dim >= 0 else len(shape) + self.dim
        
        sum_term = np.sum(self.A * dLdA, axis=dim, keepdims=True)  # Sum along softmax dimension
        dLdZ = self.A * (dLdA - sum_term)  # Element-wise multiplication

        return dLdZ
 

    