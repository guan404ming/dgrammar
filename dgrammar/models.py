"""Model wrappers for diffusion LLMs.

Provides a unified interface for Dream, LLaDA, and other dLLMs
so the decode loop can call model(x).logits without worrying
about model-specific quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel


@dataclass
class ModelOutput:
    logits: torch.Tensor


class DLLMWrapper:
    """Wraps a diffusion LLM to provide a consistent forward pass interface.

    Handles model-specific quirks:
    - LLaDA: needs use_cache=False, mask_token_id from config (not tokenizer)
    - Dream: uses different generation API
    """

    def __init__(self, model: Any, mask_token_id: int):
        self.model = model
        self.mask_token_id = mask_token_id

    def __call__(self, x: torch.Tensor) -> ModelOutput:
        with torch.no_grad():
            output = self.model(x, use_cache=False)
        return ModelOutput(logits=output.logits)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self


class WrappedTokenizer:
    """Tokenizer wrapper that ensures mask_token_id is set."""

    def __init__(self, tokenizer: Any, mask_token_id: int):
        self._tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def encode(self, text: str, **kwargs) -> Any:
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, ids: Any, **kwargs) -> str:
        return self._tokenizer.decode(ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def patch_transformers_compat():
    """Patch transformers for compatibility with custom model code.

    Applies patches only when needed (detects transformers version).
    """
    import inspect

    # Patch 1: missing all_tied_weights_keys attribute (transformers >= 5.x)
    if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
        _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
        def _safe_adjust(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
                object.__setattr__(self, "all_tied_weights_keys", {})
            return _orig_adjust(self, *args, **kwargs)
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

    # Patch 2: tie_weights signature changed in newer transformers (>= 5.x)
    if hasattr(PreTrainedModel, "_finalize_model_loading"):
        _orig_finalize = PreTrainedModel._finalize_model_loading
        def _safe_finalize(model, load_config, loading_info):
            sig = inspect.signature(model.tie_weights)
            if "missing_keys" not in sig.parameters:
                _orig_tw = model.tie_weights
                def _patched_tw(**kwargs):
                    return _orig_tw()
                model.tie_weights = _patched_tw
            return _orig_finalize(model, load_config, loading_info)
        PreTrainedModel._finalize_model_loading = staticmethod(_safe_finalize)


def load_llada(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load LLaDA-8B-Instruct with compatibility patches.

    Returns:
        (model_wrapper, tokenizer_wrapper) ready for the decode loop.
    """
    from transformers import AutoModel, AutoTokenizer

    patch_transformers_compat()

    LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
    LLADA_MASK_ID = 126336

    tokenizer = AutoTokenizer.from_pretrained(LLADA_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_MODEL_ID, trust_remote_code=True,
        torch_dtype=dtype, device_map="auto",
    ).eval()

    return (
        DLLMWrapper(model, mask_token_id=LLADA_MASK_ID),
        WrappedTokenizer(tokenizer, mask_token_id=LLADA_MASK_ID),
    )
