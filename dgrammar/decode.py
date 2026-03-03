"""Main decode loop for Dgrammar.

Implements the four-step decode loop:
1. Initialize: x = [MASK] * L
2. For each step: forward pass -> unmask top-K -> grammar check -> selective remask
3. Final check with optional fallback to per-token mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from dgrammar._core import GrammarChecker, MASK_TOKEN


@dataclass
class DecodeConfig:
    """Configuration for the Dgrammar decode loop."""
    # Length of the output sequence (number of tokens to generate)
    output_length: int = 64
    # Total denoising steps (iterations over the sequence)
    total_steps: int = 32
    # Confidence threshold for unmasking (adaptive K)
    confidence_threshold: float = 0.5
    # Max tokens to unmask per step (cap for K)
    max_unmask_per_step: int = 8
    # Max remask attempts per step before moving on
    max_remask_attempts: int = 3
    # Whether to use greedy violation detection
    greedy_violations: bool = False
    # Enable per-token fallback for final correction
    enable_fallback: bool = True
    # Fallback max steps (per-token mode)
    fallback_max_steps: int = 32


@dataclass
class DecodeStats:
    """Statistics collected during decoding."""
    total_steps: int = 0
    total_violations: int = 0
    total_remasks: int = 0
    degraded_to_fallback: bool = False
    tokens_per_step: list[int] = field(default_factory=list)


# Type for subword-to-grammar token mapping functions.
# Takes a list of decoded subword strings, returns a list of grammar-level tokens.
TokenMapper = Callable[[list[str]], list[str]]


def identity_mapper(tokens: list[str]) -> list[str]:
    """Default mapper: no transformation (tokens are grammar-level already)."""
    return tokens


def decode(
    model: Any,
    tokenizer: Any,
    prompt: str,
    grammar_checker: GrammarChecker,
    config: DecodeConfig | None = None,
    token_mapper: TokenMapper = identity_mapper,
    device: str = "cuda",
) -> tuple[str, DecodeStats]:
    """Run the Dgrammar decode loop.

    Args:
        model: A diffusion LLM (Dream/LLaDA) with a forward method that
            returns logits for masked positions.
        tokenizer: The model's tokenizer with encode/decode and mask_token_id.
        prompt: The input prompt.
        grammar_checker: A GrammarChecker instance with the target grammar.
        config: Decode configuration. Uses defaults if None.
        token_mapper: Function to map subword tokens to grammar-level tokens.
        device: Torch device.

    Returns:
        (decoded_text, stats) tuple.
    """
    if config is None:
        config = DecodeConfig()

    stats = DecodeStats()
    mask_token_id = tokenizer.mask_token_id

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    # Initialize: prompt + output_length mask tokens
    gen_len = config.output_length
    seq_len = prompt_len + gen_len
    x = torch.full(
        (1, seq_len),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x[0, :prompt_len] = prompt_ids[0]

    # Track which positions are still masked (only generation part)
    is_masked = torch.ones(gen_len, dtype=torch.bool, device=device)

    for step in range(config.total_steps):
        if not is_masked.any():
            break

        stats.total_steps += 1

        # Step 2a: Forward pass
        with torch.no_grad():
            logits = model(x).logits[0]  # (seq_len, vocab_size)

        # Step 2b: Unmask top-K confident masked positions
        gen_logits = logits[prompt_len:]  # only generation part
        probs = F.softmax(gen_logits, dim=-1)
        max_probs, predicted_ids = probs.max(dim=-1)

        # Only consider masked positions
        masked_indices = is_masked.nonzero(as_tuple=True)[0]
        if masked_indices.numel() == 0:
            break

        masked_confidences = max_probs[masked_indices]

        # Adaptive K: select positions above threshold, capped
        above_threshold = masked_confidences >= config.confidence_threshold
        if above_threshold.any():
            candidate_idx = masked_indices[above_threshold]
            candidate_conf = masked_confidences[above_threshold]
        else:
            # If nothing above threshold, take all masked and pick most confident
            candidate_idx = masked_indices
            candidate_conf = masked_confidences

        # Sort by confidence descending, take top K
        k = min(config.max_unmask_per_step, candidate_idx.numel())
        _, top_k_order = candidate_conf.topk(k)
        unmask_positions = candidate_idx[top_k_order]

        # Unmask these positions
        for gen_pos in unmask_positions:
            pos = gen_pos.item()
            x[0, prompt_len + pos] = predicted_ids[pos]
            is_masked[pos] = False

        stats.tokens_per_step.append(k)

        # Step 2c-d: Grammar check and selective remask
        raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
        grammar_tokens = token_mapper(raw_tokens)
        unmasked_pos_list = unmask_positions.tolist()

        # Map unmasked positions through the token mapper if needed
        grammar_unmasked = _map_positions(raw_tokens, grammar_tokens, unmasked_pos_list)

        for attempt in range(config.max_remask_attempts):
            if config.greedy_violations:
                violations = grammar_checker.find_violations_greedy(
                    grammar_tokens, grammar_unmasked
                )
            else:
                violations = grammar_checker.find_violations(
                    grammar_tokens, grammar_unmasked
                )

            if not violations:
                break

            stats.total_violations += len(violations)
            stats.total_remasks += 1

            # Map grammar-level violations back to sequence positions
            seq_violations = _unmap_positions(raw_tokens, grammar_tokens, violations)

            # Remask violating positions
            for v_pos in seq_violations:
                if v_pos < gen_len:
                    x[0, prompt_len + v_pos] = mask_token_id
                    is_masked[v_pos] = True

            # Re-predict the remasked positions
            with torch.no_grad():
                logits = model(x).logits[0]

            gen_logits = logits[prompt_len:]
            probs = F.softmax(gen_logits, dim=-1)
            _, predicted_ids = probs.max(dim=-1)

            for v_pos in seq_violations:
                if v_pos < gen_len:
                    x[0, prompt_len + v_pos] = predicted_ids[v_pos]
                    is_masked[v_pos] = False

            raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
            grammar_tokens = token_mapper(raw_tokens)
            grammar_unmasked = _map_positions(raw_tokens, grammar_tokens, seq_violations)

    # Step 3: Final check and optional fallback
    raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
    grammar_tokens = token_mapper(raw_tokens)
    if config.enable_fallback and not grammar_checker.check(grammar_tokens):
        stats.degraded_to_fallback = True
        x, is_masked = _fallback_per_token(
            model, tokenizer, x, is_masked, prompt_len, gen_len,
            grammar_checker, token_mapper, config.fallback_max_steps, device,
        )

    # Decode final output
    gen_ids = x[0, prompt_len:]
    # Trim trailing masks
    non_mask = (gen_ids != mask_token_id).nonzero(as_tuple=True)[0]
    if non_mask.numel() > 0:
        last_non_mask = non_mask[-1].item() + 1
        gen_ids = gen_ids[:last_non_mask]

    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return output_text, stats


def _ids_to_token_strings(
    tokenizer: Any,
    ids: torch.Tensor,
    is_masked: torch.Tensor,
) -> list[str]:
    """Convert token IDs to strings, preserving MASK tokens."""
    tokens = []
    for i, tid in enumerate(ids):
        if is_masked[i]:
            tokens.append(MASK_TOKEN)
        else:
            tokens.append(tokenizer.decode([tid.item()]).strip())
    return tokens


def _map_positions(
    raw_tokens: list[str],
    grammar_tokens: list[str],
    positions: list[int],
) -> list[int]:
    """Map raw sequence positions to grammar-token positions.

    When token_mapper is identity, this is a no-op.
    """
    if len(raw_tokens) == len(grammar_tokens):
        return positions
    # For non-trivial mappers, positions map 1:1 since we track by index.
    # Subword merging would require a more complex mapping.
    return positions


def _unmap_positions(
    raw_tokens: list[str],
    grammar_tokens: list[str],
    grammar_positions: list[int],
) -> list[int]:
    """Map grammar-token positions back to raw sequence positions."""
    if len(raw_tokens) == len(grammar_tokens):
        return grammar_positions
    return grammar_positions


def _fallback_per_token(
    model: Any,
    tokenizer: Any,
    x: torch.Tensor,
    is_masked: torch.Tensor,
    prompt_len: int,
    gen_len: int,
    grammar_checker: GrammarChecker,
    token_mapper: TokenMapper,
    max_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token fallback mode (LAVE-like sequential decoding)."""
    mask_token_id = tokenizer.mask_token_id

    for _ in range(max_steps):
        if not is_masked.any():
            break

        # Check current state
        raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
        grammar_tokens = token_mapper(raw_tokens)
        if grammar_checker.check(grammar_tokens):
            break

        # Find the first masked position and unmask it
        masked_idx = is_masked.nonzero(as_tuple=True)[0]
        if masked_idx.numel() == 0:
            break

        pos = masked_idx[0].item()

        with torch.no_grad():
            logits = model(x).logits[0]

        probs = F.softmax(logits[prompt_len + pos], dim=-1)
        predicted_id = probs.argmax()

        x[0, prompt_len + pos] = predicted_id
        is_masked[pos] = False

    return x, is_masked
