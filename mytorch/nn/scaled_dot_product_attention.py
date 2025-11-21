import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        # Softmax should be applied along the S (source sequence) dimension, which is the last dimension
        # Shape is (N, ..., H, L, S), so dim=-1
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        E = Q.shape[-1]
        d_k = np.sqrt(E)
        
        K_transposed = np.swapaxes(K, -2, -1)
        scaled_dot_product = (Q @ K_transposed) / d_k
        
        if mask is not None:
            mask_float = mask.astype(np.float32) * (-self.eps)
            scaled_dot_product = scaled_dot_product + mask_float

        self.attention_scores = self.softmax.forward(scaled_dot_product)

        output = self.attention_scores @ V

        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass
        E = self.Q.shape[-1]
        d_k = np.sqrt(E)

        attention_scores_T = np.swapaxes(self.attention_scores, -2, -1)
        d_V = attention_scores_T @ d_output
        
        V_T = np.swapaxes(self.V, -2, -1)
        d_attention_scores = d_output @ V_T
        
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        if self.mask is not None:
            pass
        
        d_scaled_dot_product = d_scaled_dot_product / d_k
        
        d_Q = d_scaled_dot_product @ self.K
        d_scaled_dot_product_T = np.swapaxes(d_scaled_dot_product, -2, -1)
        d_K = d_scaled_dot_product_T @ self.Q
        
        return d_Q, d_K, d_V

