import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement greedy search
        batch_size = x.size(0)
        sequences = x.clone()  
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        
        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break
            
            next_logits = self.score_fn(sequences)
            
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            if repeat_penalty != 1.0:
                next_logits = self._apply_repeat_penalty(next_logits, sequences, repeat_penalty)

            log_probs = torch.log_softmax(next_logits, dim=-1)
            
            next_tokens = torch.argmax(log_probs, dim=-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)
            
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
        
        return sequences, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement beam search
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        
        sequences = x.unsqueeze(1).expand(batch_size, beam_width, -1).clone()
        scores = torch.zeros(batch_size, beam_width, device=x.device)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=x.device)
        
        for step in range(self.max_length - initial_seq_len):
            if finished.all():
                break
            
            current_seq_len = sequences.size(2)

            batch_log_probs_list = []
            vocab_size = None
            
            for batch_idx in range(batch_size):
                batch_sequences = sequences[batch_idx]
                beam_log_probs_list = []
                
                if hasattr(self.score_fn, 'trees') and hasattr(self.score_fn, 'tokenizer'):
                    original_score_fn = self.score_fn
                    class BatchScoreFn:
                        def __init__(self, tree, tokenizer, original_fn):
                            self.tree = tree
                            self.tokenizer = tokenizer
                            self.original_fn = original_fn
                        def __call__(self, x):
                            batch_size = x.size(0)
                            seq_len = x.size(1)
                            scores = torch.full((batch_size, seq_len, self.tokenizer.vocab_size), 1.0)
                            for i in range(batch_size):
                                scores[i, -1, :] = self.original_fn.update_scores(self.tree, x[i])
                            return scores[:, -1, :].log_softmax(dim=-1)
                    
                    temp_score_fn = BatchScoreFn(self.score_fn.trees[batch_idx], self.score_fn.tokenizer, self.score_fn)
                else:
                    temp_score_fn = self.score_fn
                
                for beam_idx in range(beam_width):
                    beam_sequence = batch_sequences[beam_idx:beam_idx+1]
                    
                    beam_logits = temp_score_fn(beam_sequence)
                    if vocab_size is None:
                        vocab_size = beam_logits.size(-1)
                    
                    beam_logits = beam_logits.squeeze(0)

                    if temperature != 1.0:
                        beam_logits = beam_logits / temperature
                    
                    if repeat_penalty != 1.0:
                        beam_logits = beam_logits.unsqueeze(0)
                        beam_seq_for_penalty = beam_sequence
                        beam_logits = self._apply_repeat_penalty(beam_logits, beam_seq_for_penalty, repeat_penalty)
                        beam_logits = beam_logits.squeeze(0)
                    
                    beam_log_probs = torch.log_softmax(beam_logits, dim=-1)
                    beam_log_probs_list.append(beam_log_probs)
                
                batch_log_probs = torch.stack(beam_log_probs_list)
                batch_log_probs_list.append(batch_log_probs)
            
            log_probs = torch.stack(batch_log_probs_list)
            
            new_sequences_list = []
            new_scores_list = []
            new_finished_list = []
            
            for batch_idx in range(batch_size):
                batch_log_probs = log_probs[batch_idx]
                batch_scores = scores[batch_idx]
                batch_finished = finished[batch_idx]
                batch_sequences = sequences[batch_idx]
                
                candidates_scores = []
                candidates_tokens = []
                candidates_sequences = []
                candidates_finished = []
                
                for beam_idx in range(beam_width):
                    if batch_finished[beam_idx]:
                        candidates_scores.append(batch_scores[beam_idx])
                        candidates_tokens.append(None)
                        candidates_sequences.append(batch_sequences[beam_idx])
                        candidates_finished.append(True)
                    else:
                        top_k = min(beam_width, vocab_size)
                        top_log_probs_beam, top_indices_beam = torch.topk(
                            batch_log_probs[beam_idx], top_k, dim=-1
                        )
                        
                        for k in range(top_k):
                            token_score = top_log_probs_beam[k]
                            token = top_indices_beam[k]
                            cumulative_score = batch_scores[beam_idx] + token_score
                            
                            candidates_scores.append(cumulative_score)
                            candidates_tokens.append(token)
                            candidates_sequences.append(batch_sequences[beam_idx])
                            candidates_finished.append(False)
                
                candidates_scores_tensor = torch.tensor(candidates_scores, device=x.device)
                
                sorted_indices = torch.argsort(
                    -candidates_scores_tensor, 
                    stable=True, 
                    descending=False
                )
                
                selected_indices = []
                seen_sequences = set()
                
                for idx in sorted_indices:
                    if len(selected_indices) >= beam_width:
                        break
                    
                    prev_seq = candidates_sequences[idx]
                    is_finished = candidates_finished[idx]
                    token = candidates_tokens[idx] if not is_finished else None
                    
                    if token is not None:
                        seq_list = prev_seq.cpu().tolist()
                        seq_key = (tuple(seq_list), token.item())
                    else:
                        seq_list = prev_seq.cpu().tolist()
                        seq_key = tuple(seq_list)
                    
                    if seq_key not in seen_sequences:
                        seen_sequences.add(seq_key)
                        selected_indices.append(idx.item())
                
                if len(selected_indices) < beam_width:
                    remaining_needed = beam_width - len(selected_indices)
                    for idx in sorted_indices:
                        if remaining_needed <= 0:
                            break
                        if idx.item() not in selected_indices:
                            selected_indices.append(idx.item())
                            remaining_needed -= 1
                
                new_batch_sequences = []
                new_batch_scores = []
                new_batch_finished = []
                
                for idx in selected_indices:
                    prev_seq = candidates_sequences[idx]
                    is_finished = candidates_finished[idx]
                    
                    if not is_finished:
                        token = candidates_tokens[idx]
                        new_seq = torch.cat([prev_seq, token.unsqueeze(0)], dim=0)
                        new_finished = (token == self.tokenizer.eos_id)
                    else:
                        new_seq = prev_seq
                        new_finished = True
                    
                    new_batch_sequences.append(new_seq)
                    new_batch_scores.append(candidates_scores_tensor[idx])
                    new_batch_finished.append(new_finished)
                
                max_len = max(seq.size(0) for seq in new_batch_sequences)
                padded_sequences = []
                for seq in new_batch_sequences:
                    if seq.size(0) < max_len:
                        padding = torch.full((max_len - seq.size(0),), 
                                            self.tokenizer.pad_id, dtype=seq.dtype, device=seq.device)
                        seq = torch.cat([seq, padding], dim=0)
                    padded_sequences.append(seq)
                
                new_sequences_list.append(torch.stack(padded_sequences))
                new_scores_list.append(torch.stack(new_batch_scores))
                new_finished_list.append(torch.tensor(new_batch_finished, device=x.device))
            
            sequences = torch.stack(new_sequences_list)
            scores = torch.stack(new_scores_list)
            finished = torch.stack(new_finished_list)
        
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        sequences = sequences[batch_indices, sorted_indices]
        scores = scores[batch_indices, sorted_indices]
        
        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]