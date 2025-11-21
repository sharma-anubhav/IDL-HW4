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
        
        # Store Q, K, V for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        # Get embedding dimension E
        E = Q.shape[-1]
        d_k = np.sqrt(E)  # Scaling factor
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        # Need to transpose K: (N, ..., H, S, E) -> (N, ..., H, E, S)
        K_transposed = np.swapaxes(K, -2, -1)  # Swap last two dimensions
        scaled_dot_product = (Q @ K_transposed) / d_k
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            # Convert boolean mask to float and apply: True/1 -> -eps, False/0 -> 0
            mask_float = mask.astype(np.float32) * (-self.eps)
            scaled_dot_product = scaled_dot_product + mask_float

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores @ V

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Get embedding dimension E
        E = self.Q.shape[-1]
        d_k = np.sqrt(E)  # Scaling factor

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # d_output: (N, ..., H, L, Ev)
        # attention_scores: (N, ..., H, L, S)
        # d_V = attention_scores^T @ d_output
        # (N, ..., H, L, S) -> (N, ..., H, S, L) @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev)
        attention_scores_T = np.swapaxes(self.attention_scores, -2, -1)  # (N, ..., H, S, L)
        d_V = attention_scores_T @ d_output  # (N, ..., H, S, L) @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev)
        
        # Calculate gradients for attention scores
        # d_attention_scores = d_output @ V^T
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        V_T = np.swapaxes(self.V, -2, -1)  # (N, ..., H, Ev, S)
        d_attention_scores = d_output @ V_T  # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        
        # Backpropagate through softmax
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Apply mask gradient if mask was used
        if self.mask is not None:
            # Mask doesn't affect gradient computation, but we already applied it in forward
            pass
        
        # Scale gradients by sqrt(d_k) (undo the scaling from forward)
        d_scaled_dot_product = d_scaled_dot_product / d_k
        
        # Calculate gradients for Q and K
        # d_Q = d_scaled_dot_product @ K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K  # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)
        
        # d_K = d_scaled_dot_product^T @ Q
        # (N, ..., H, L, S) -> (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_scaled_dot_product_T = np.swapaxes(d_scaled_dot_product, -2, -1)  # (N, ..., H, S, L)
        d_K = d_scaled_dot_product_T @ self.Q  # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

