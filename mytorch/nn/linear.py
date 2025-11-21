import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        
        # Reshape A to 2D: (*, in_features) -> (batch_size, in_features)
        # where batch_size is the product of all dimensions except the last
        original_shape = A.shape
        A_2d = A.reshape(-1, A.shape[-1])  # (batch_size, in_features)
        
        # Compute Z = A @ W^T + b
        # A_2d: (batch_size, in_features)
        # W: (out_features, in_features)
        # Z: (batch_size, out_features)
        Z_2d = A_2d @ self.W.T + self.b
        
        # Reshape back to original shape but with out_features as last dimension
        # (*, in_features) -> (*, out_features)
        Z = Z_2d.reshape(*original_shape[:-1], self.W.shape[0])
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Reshape inputs to 2D for easier computation
        original_shape = self.A.shape
        A_2d = self.A.reshape(-1, self.A.shape[-1])  # (batch_size, in_features)
        dLdZ_2d = dLdZ.reshape(-1, dLdZ.shape[-1])  # (batch_size, out_features)
        
        # Compute gradients
        # dL/dW = dL/dZ @ A
        # dLdZ_2d: (batch_size, out_features)
        # A_2d: (batch_size, in_features)
        # dLdW: (out_features, in_features)
        self.dLdW = dLdZ_2d.T @ A_2d  # (out_features, batch_size) @ (batch_size, in_features) = (out_features, in_features)
        
        # dL/db = sum over batch dimension of dL/dZ
        # dLdZ_2d: (batch_size, out_features)
        # dLdb: (out_features,)
        self.dLdb = np.sum(dLdZ_2d, axis=0)  # Sum over batch dimension
        
        # dL/dA = dL/dZ @ W
        # dLdZ_2d: (batch_size, out_features)
        # W: (out_features, in_features)
        # dLdA_2d: (batch_size, in_features)
        dLdA_2d = dLdZ_2d @ self.W  # (batch_size, out_features) @ (out_features, in_features) = (batch_size, in_features)
        
        # Reshape back to original shape
        self.dLdA = dLdA_2d.reshape(original_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
