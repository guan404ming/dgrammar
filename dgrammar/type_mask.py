"""Type-aware token masking for JSON schema generation.

Pre-computes type-specific token masks from the vocabulary, then applies
them at positions where the expected JSON type can be inferred from the
left context. Blocks 96-99% of vocabulary at value-start positions.
"""

import json
import re

import torch


class TypeAwareMasker:
    """Applies schema-driven type masks at value-start positions."""

    def __init__(self, schema_str, tokenizer, device="cpu"):
        self.schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
        self.tokenizer = tokenizer
        self.device = device

        # Build property type map (including nested)
        self.prop_types = {}
        self._extract_prop_types(self.schema, self.prop_types)

        # Pre-compute type masks
        vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)
        all_decoded = tokenizer.batch_decode(list(range(vocab_size)))

        # Masks: True = blocked
        self.number_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.string_start_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.bool_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.null_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.array_start_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.object_start_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        for i, text in enumerate(all_decoded):
            s = text.lstrip()
            if not s:
                # Whitespace-only tokens are allowed everywhere
                self.number_mask[i] = False
                self.string_start_mask[i] = False
                self.bool_mask[i] = False
                self.null_mask[i] = False
                self.array_start_mask[i] = False
                self.object_start_mask[i] = False
                continue
            # Number: starts with digit or minus
            if s[0] in "0123456789-":
                self.number_mask[i] = False
            # String start: starts with quote
            if s[0] == '"':
                self.string_start_mask[i] = False
            # Boolean
            sl = s.lower()
            if sl.startswith("true") or sl.startswith("false"):
                self.bool_mask[i] = False
            # Null
            if sl.startswith("null"):
                self.null_mask[i] = False
            # Array start
            if s[0] == "[":
                self.array_start_mask[i] = False
            # Object start
            if s[0] == "{":
                self.object_start_mask[i] = False

        # Pad masks to model vocab size
        self._masks = {
            "number": self.number_mask,
            "integer": self.number_mask,
            "string": self.string_start_mask,
            "boolean": self.bool_mask,
            "null": self.null_mask,
            "array": self.array_start_mask,
            "object": self.object_start_mask,
        }

    def _extract_prop_types(self, schema, prop_types, prefix=""):
        """Recursively extract property name -> type mapping."""
        props = schema.get("properties", {})
        for name, prop_schema in props.items():
            t = prop_schema.get("type", "string")
            prop_types[name] = t
            # For arrays, also store item type
            if t == "array" and "items" in prop_schema:
                items = prop_schema["items"]
                prop_types[f"{name}.__items__"] = items.get("type", "string")
                if items.get("type") == "object":
                    self._extract_prop_types(items, prop_types)
            # For nested objects
            if t == "object":
                self._extract_prop_types(prop_schema, prop_types)

    def pad_mask(self, mask, target_size):
        """Pad mask to target vocab size."""
        if mask.shape[0] < target_size:
            pad = torch.ones(target_size - mask.shape[0], dtype=torch.bool, device=self.device)
            return torch.cat([mask, pad])
        return mask[:target_size]

    def get_type_mask(self, type_name, vocab_size):
        """Get the block mask for a given type. True = blocked."""
        mask = self._masks.get(type_name)
        if mask is None:
            return None
        return self.pad_mask(mask, vocab_size)

    def analyze_position(self, generated_words, pos, prompt_len):
        """Determine expected JSON type at a given position.

        Only applies when the immediate left context (last few tokens) is
        fully placed (no gaps). Diffusion models fill tokens by confidence,
        not left-to-right, so gappy context leads to wrong type predictions.

        Returns type string or None if ambiguous.
        """
        # Require the last N tokens before pos to all be placed (no gaps).
        # This ensures we have reliable local context for type inference.
        lookback = 5
        start = max(prompt_len, pos - lookback)
        for i in range(start, pos):
            if generated_words[i] is None:
                return None  # gap in recent context, skip
            if not isinstance(generated_words[i], str):
                return None  # EOS or other sentinel, skip

        # Build left context string from placed tokens only
        left_parts = []
        for i in range(prompt_len, pos):
            w = generated_words[i]
            if w is not None and isinstance(w, str):
                left_parts.append(w)
            # Skip gaps and EOS sentinels

        if not left_parts:
            return None  # not enough context yet

        left_text = "".join(left_parts)
        stripped = left_text.rstrip()

        if not stripped:
            return None

        # After colon (with optional space): expect value
        if re.search(r':\s*$', stripped):
            prop_name = self._extract_last_property(stripped)
            if prop_name and prop_name in self.prop_types:
                return self.prop_types[prop_name]
            return None  # unknown property, don't mask

        # After { or after comma: expect property name (string)
        if stripped.endswith("{") or re.search(r',\s*$', stripped):
            return "string"  # property name starts with "

        # After [ : expect array item
        if stripped.endswith("["):
            prop_name = self._extract_array_context(stripped)
            if prop_name:
                items_key = f"{prop_name}.__items__"
                if items_key in self.prop_types:
                    return self.prop_types[items_key]
            return None

        return None  # ambiguous, don't apply type mask

    def _extract_last_property(self, text):
        """Extract the last property name before a colon from JSON text."""
        # Match "propName": at the end
        m = re.search(r'"(\w+)"\s*:\s*$', text)
        if m:
            return m.group(1)
        return None

    def _extract_array_context(self, text):
        """Try to find which property's array we're in."""
        # Look for "propName": [ pattern
        m = re.search(r'"(\w+)"\s*:\s*\[$', text.rstrip())
        if m:
            return m.group(1)
        return None

    def apply_masks(self, logits_with_noise, generated_words, prompt_len,
                    mask_id, block_start, block_end):
        """Apply type masks to logits at value-start mask positions.

        Scans mask positions in [block_start, block_end) and applies
        type-specific masks where the expected type can be determined.

        Returns number of positions masked.
        """
        vocab_size = logits_with_noise.shape[-1]
        n_masked = 0

        for pos in range(block_start, min(block_end, len(generated_words))):
            if generated_words[pos] is not None:
                continue  # not a mask position

            expected_type = self.analyze_position(generated_words, pos, prompt_len)
            if expected_type is None:
                continue

            type_mask = self.get_type_mask(expected_type, vocab_size)
            if type_mask is None:
                continue

            logits_with_noise[0, pos, type_mask] = float("-inf")
            n_masked += 1

        return n_masked
