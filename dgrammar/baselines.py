"""Baseline runners for comparison experiments.

Implements:
- No-CD: Unconstrained dLLM decoding
- Mundler (FS-CD): Per-token rejection sampling
- LAVE: Per-token + lookahead verify
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from dgrammar._core import GrammarChecker
from dgrammar.decode import DecodeStats, TokenMapper, identity_mapper, _ids_to_token_strings


@dataclass
class BaselineConfig:
    """Configuration for baseline decoders."""
    output_length: int = 64
    total_steps: int = 64
    # For FS-CD: max rejection attempts per position
    max_rejections: int = 10
    # For LAVE: lookahead window
    lookahead_k: int = 3


def decode_no_cd(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: BaselineConfig | None = None,
    device: str = "cuda",
) -> tuple[str, DecodeStats]:
    """Unconstrained dLLM decoding (No-CD baseline).

    Standard denoising without any grammar constraints.
    """
    if config is None:
        config = BaselineConfig()

    stats = DecodeStats()
    mask_token_id = tokenizer.mask_token_id

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    gen_len = config.output_length
    seq_len = prompt_len + gen_len
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt_ids[0]
    is_masked = torch.ones(gen_len, dtype=torch.bool, device=device)

    for step in range(config.total_steps):
        if not is_masked.any():
            break

        stats.total_steps += 1

        with torch.no_grad():
            logits = model(x).logits[0]

        gen_logits = logits[prompt_len:]
        probs = F.softmax(gen_logits, dim=-1)
        max_probs, predicted_ids = probs.max(dim=-1)

        # Unmask the single most confident position
        masked_idx = is_masked.nonzero(as_tuple=True)[0]
        if masked_idx.numel() == 0:
            break

        confidences = max_probs[masked_idx]
        best = confidences.argmax()
        pos = masked_idx[best].item()

        x[0, prompt_len + pos] = predicted_ids[pos]
        is_masked[pos] = False
        stats.tokens_per_step.append(1)

    gen_ids = x[0, prompt_len:]
    non_mask = (gen_ids != mask_token_id).nonzero(as_tuple=True)[0]
    if non_mask.numel() > 0:
        gen_ids = gen_ids[:non_mask[-1].item() + 1]

    return tokenizer.decode(gen_ids, skip_special_tokens=True), stats


def decode_fscd(
    model: Any,
    tokenizer: Any,
    prompt: str,
    grammar_checker: GrammarChecker,
    config: BaselineConfig | None = None,
    token_mapper: TokenMapper = identity_mapper,
    device: str = "cuda",
) -> tuple[str, DecodeStats]:
    """Per-token rejection sampling (Mundler FS-CD baseline).

    Unmasks one token at a time. If it violates the grammar,
    reject and resample from the next-best prediction.
    """
    if config is None:
        config = BaselineConfig()

    stats = DecodeStats()
    mask_token_id = tokenizer.mask_token_id

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    gen_len = config.output_length
    seq_len = prompt_len + gen_len
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt_ids[0]
    is_masked = torch.ones(gen_len, dtype=torch.bool, device=device)

    for step in range(config.total_steps):
        if not is_masked.any():
            break

        stats.total_steps += 1

        with torch.no_grad():
            logits = model(x).logits[0]

        gen_logits = logits[prompt_len:]
        probs = F.softmax(gen_logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        masked_idx = is_masked.nonzero(as_tuple=True)[0]
        if masked_idx.numel() == 0:
            break

        confidences = max_probs[masked_idx]
        best = confidences.argmax()
        pos = masked_idx[best].item()

        # Try top predictions with rejection
        sorted_probs, sorted_ids = probs[pos].sort(descending=True)
        accepted = False

        for attempt in range(min(config.max_rejections, sorted_ids.numel())):
            candidate_id = sorted_ids[attempt]
            x[0, prompt_len + pos] = candidate_id
            is_masked[pos] = False

            raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
            grammar_tokens = token_mapper(raw_tokens)
            if grammar_checker.check(grammar_tokens):
                accepted = True
                break
            else:
                stats.total_violations += 1

        if not accepted:
            x[0, prompt_len + pos] = sorted_ids[0]
            is_masked[pos] = False

        stats.tokens_per_step.append(1)

    gen_ids = x[0, prompt_len:]
    non_mask = (gen_ids != mask_token_id).nonzero(as_tuple=True)[0]
    if non_mask.numel() > 0:
        gen_ids = gen_ids[:non_mask[-1].item() + 1]

    return tokenizer.decode(gen_ids, skip_special_tokens=True), stats


def decode_lave(
    model: Any,
    tokenizer: Any,
    prompt: str,
    grammar_checker: GrammarChecker,
    config: BaselineConfig | None = None,
    token_mapper: TokenMapper = identity_mapper,
    device: str = "cuda",
) -> tuple[str, DecodeStats]:
    """Per-token + lookahead verify (LAVE baseline).

    Like FS-CD but with a lookahead window to verify consistency.
    """
    if config is None:
        config = BaselineConfig()

    stats = DecodeStats()
    mask_token_id = tokenizer.mask_token_id

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    gen_len = config.output_length
    seq_len = prompt_len + gen_len
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt_ids[0]
    is_masked = torch.ones(gen_len, dtype=torch.bool, device=device)

    for step in range(config.total_steps):
        if not is_masked.any():
            break

        stats.total_steps += 1

        with torch.no_grad():
            logits = model(x).logits[0]

        gen_logits = logits[prompt_len:]
        probs = F.softmax(gen_logits, dim=-1)
        max_probs, predicted_ids = probs.max(dim=-1)

        masked_idx = is_masked.nonzero(as_tuple=True)[0]
        if masked_idx.numel() == 0:
            break

        # Sort masked positions by confidence
        confidences = max_probs[masked_idx]
        sorted_order = confidences.argsort(descending=True)
        sorted_positions = masked_idx[sorted_order]

        # Unmask the most confident position
        pos = sorted_positions[0].item()
        x[0, prompt_len + pos] = predicted_ids[pos]
        is_masked[pos] = False

        # Lookahead: tentatively unmask next k positions
        lookahead_positions = []
        lookahead_originals = []
        for la_idx in range(1, min(config.lookahead_k + 1, sorted_positions.numel())):
            la_pos = sorted_positions[la_idx].item()
            lookahead_positions.append(la_pos)
            lookahead_originals.append(x[0, prompt_len + la_pos].item())
            x[0, prompt_len + la_pos] = predicted_ids[la_pos]
            is_masked[la_pos] = False

        # Verify with grammar
        raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
        grammar_tokens = token_mapper(raw_tokens)
        if not grammar_checker.check(grammar_tokens):
            stats.total_violations += 1
            # Revert lookahead positions
            for la_pos, orig_id in zip(lookahead_positions, lookahead_originals):
                x[0, prompt_len + la_pos] = orig_id
                is_masked[la_pos] = True
            # Check if main position alone is valid
            raw_tokens = _ids_to_token_strings(tokenizer, x[0, prompt_len:], is_masked)
            grammar_tokens = token_mapper(raw_tokens)
            if not grammar_checker.check(grammar_tokens):
                x[0, prompt_len + pos] = mask_token_id
                is_masked[pos] = True

        stats.tokens_per_step.append(1)

    gen_ids = x[0, prompt_len:]
    non_mask = (gen_ids != mask_token_id).nonzero(as_tuple=True)[0]
    if non_mask.numel() > 0:
        gen_ids = gen_ids[:non_mask[-1].item() + 1]

    return tokenizer.decode(gen_ids, skip_special_tokens=True), stats
