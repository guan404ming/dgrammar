"""Token-level DFA pre-masking using rustformlang's lex DFAs.

Precomputes per-state token masks for each terminal's prefix DFA.
At runtime, determines the lex state from the placed prefix and
applies the appropriate mask at masked positions. Uses the same
grammar engine as post-checking (rustformlang), so no conflicts.
"""

import re
import time

import numpy as np
import torch

from constrained_diffusion.constrain_utils import (
    lex,
    CompiledLexMap,
)
from rustformlang.cfg import CFG, is_intersection_empty_threaded
from rustformlang.fa.bytes_dfa import regex_to_dfa
from rustformlang.fa.epsilon_nfa import ENFA, minimize_enfa_threaded


class ParsedDFA:
    """DFA transition table parsed from rustformlang's to_text() output."""

    def __init__(self, text):
        self.transitions = {}  # (state_idx, byte) -> state_idx
        self.start_state = None
        self.accept_states = set()
        self.states = []
        self.num_states_count = 0
        self._parse(text)

    def _parse(self, text):
        m = re.search(r"Start State:\s*(\d+)", text)
        if m:
            self.start_state = int(m.group(1))

        accept_block = re.search(r"Accept States: \{([^}]*)\}", text)
        if accept_block:
            for m in re.finditer(r'State \{ name: "q(\d+)" \}', accept_block.group(1)):
                self.accept_states.add(int(m.group(1)))

        states_block = re.search(r"States: \[([^\]]*)\]", text)
        if states_block:
            for m in re.finditer(r'State \{ name: "q(\d+)" \}', states_block.group(1)):
                self.states.append(int(m.group(1)))
        self.num_states_count = len(self.states)

        for m in re.finditer(r"q(\d+) -- (\d+) --> q(\d+)", text):
            src = int(m.group(1))
            byte_val = int(m.group(2))
            dst = int(m.group(3))
            self.transitions[(src, byte_val)] = dst

    def walk_bytes(self, byte_sequence):
        """Walk the DFA on a byte sequence from the start state. Returns final state or None."""
        state = self.start_state
        for b in byte_sequence:
            state = self.transitions.get((state, b))
            if state is None:
                return None
        return state

    def build_numpy_table(self):
        """Build a numpy transition table for vectorized DFA walking."""
        n = self.num_states_count
        # Map state indices to contiguous 0..n-1
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        # -1 means dead state
        table = np.full((n, 256), -1, dtype=np.int32)
        for (src, byte_val), dst in self.transitions.items():
            si = state_to_idx.get(src)
            di = state_to_idx.get(dst)
            if si is not None and di is not None:
                table[si, byte_val] = di
        return table, state_to_idx


class TokenDFATable:
    """Precomputed token masks based on lex-level DFA states.

    For each terminal's prefix DFA and each state, stores a boolean mask
    indicating which vocabulary tokens can advance the DFA from that state.
    At runtime, determines the current lex state from the placed prefix
    and applies the mask at the first masked position.
    """

    def __init__(
        self,
        tokenizer,
        lex_map_dict: dict,
        compiled_lex_map: CompiledLexMap,
        device,
        constraint_lang: CFG = None,
        prelex=None,
        strip_chars=None,
        trace=False,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.compiled_lex_map = compiled_lex_map
        self.constraint_lang = constraint_lang
        self.prelex = prelex
        self.strip_chars = strip_chars
        # Cache for CFG-aware next terminal queries
        self._next_terminal_cache = {}

        vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)
        self.vocab_size = vocab_size

        t0 = time.monotonic()

        # Decode all tokens to byte sequences
        all_tokens_decoded = tokenizer.batch_decode(torch.arange(0, vocab_size))
        self.all_token_bytes = []
        for s in all_tokens_decoded:
            self.all_token_bytes.append(s.encode("utf-8"))

        # Build padded byte array for vectorized DFA walking
        max_len = max(len(b) for b in self.all_token_bytes) if self.all_token_bytes else 1
        self.max_token_len = max_len
        token_bytes_arr = np.zeros((vocab_size, max_len), dtype=np.uint8)
        token_lens = np.zeros(vocab_size, dtype=np.int32)
        for i, b in enumerate(self.all_token_bytes):
            token_bytes_arr[i, : len(b)] = list(b)
            token_lens[i] = len(b)
        self.token_bytes_arr = token_bytes_arr
        self.token_lens = token_lens

        # Parse each terminal's prefix DFA and original DFA, build numpy tables
        self.terminal_names = []
        self.terminal_prefix_dfas = {}  # name -> BytesDFA (prefix language)
        self.terminal_parsed = {}  # name -> ParsedDFA (prefix)
        self.terminal_tables = {}  # name -> (numpy_table, state_to_idx)
        self.terminal_orig_parsed = {}  # name -> ParsedDFA (original, for accept detection)
        self.terminal_orig_tables = {}  # name -> (numpy_table, state_to_idx)

        for name, regex_or_dfa in lex_map_dict.items():
            dfa = regex_to_dfa(regex_or_dfa) if isinstance(regex_or_dfa, str) else regex_or_dfa
            # Prefix language DFA (for intra-terminal continuation)
            pref_dfa = dfa.true_prefix_language()
            self.terminal_prefix_dfas[name] = pref_dfa
            parsed = ParsedDFA(pref_dfa.to_text())
            self.terminal_parsed[name] = parsed
            table, state_map = parsed.build_numpy_table()
            self.terminal_tables[name] = (table, state_map)
            # Original DFA (for accept/completion detection)
            orig_parsed = ParsedDFA(dfa.to_text())
            self.terminal_orig_parsed[name] = orig_parsed
            orig_table, orig_state_map = orig_parsed.build_numpy_table()
            self.terminal_orig_tables[name] = (orig_table, orig_state_map)
            self.terminal_names.append(name)

        # Precompute per-state token masks
        self.state_masks = {}  # (terminal_name, orig_state) -> bool tensor (vocab_size,)
        self._precompute_masks(trace)

        # Logit-dimension mask (padded if model vocab > tokenizer vocab)
        self._logit_vocab_size = None

        # Build EOS whitelist mask (these tokens should never be blocked)
        self.eos_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        eos_token = tokenizer.special_tokens_map.get("eos_token", "")
        eos_names = {
            eos_token, "<|eot_id|>", "<|endoftext|>",
            "<\uff5cend\u2581of\u2581sentence\uff5c>",
        }
        for tok_id, tok_obj in tokenizer.added_tokens_decoder.items():
            if tok_obj.content in eos_names and tok_id < vocab_size:
                self.eos_mask[tok_id] = True

        dt = time.monotonic() - t0
        if trace:
            total_states = sum(p.num_states_count for p in self.terminal_parsed.values())
            print(f"TokenDFATable: {len(self.terminal_names)} terminals, "
                  f"{total_states} total states, precomputed in {dt:.2f}s")

    def _precompute_masks(self, trace):
        """Vectorized mask computation: for each (terminal, state), which tokens survive the DFA walk."""
        for name in self.terminal_names:
            table, state_map = self.terminal_tables[name]
            parsed = self.terminal_parsed[name]
            n_states = parsed.num_states_count

            if n_states == 0:
                continue

            # Original DFA for completion detection
            orig_table, orig_state_map = self.terminal_orig_tables[name]
            orig_parsed = self.terminal_orig_parsed[name]

            # For each state, compute intra-terminal mask (prefix DFA walk)
            for orig_state in parsed.states:
                idx = state_map[orig_state]
                mask = self._compute_mask_for_state(
                    idx, table, self.token_bytes_arr, self.token_lens
                )
                self.state_masks[(name, orig_state)] = torch.tensor(
                    mask, dtype=torch.bool, device=self.device
                )

            # Compute completion masks: tokens that complete this terminal
            # from each original DFA state. A token completes the terminal if
            # at some byte position during the walk, the original DFA is in
            # an accept state.
            # Map prefix DFA states to original DFA states using byte walking.
            # For each prefix state, find the corresponding original state by
            # walking both DFAs on the same example path.
            # Simpler approach: for each prefix DFA state, precompute which
            # tokens, when walked on the ORIGINAL DFA, reach an accept state.
            self._compute_completion_masks(name, parsed, state_map,
                                           orig_parsed, orig_table, orig_state_map)

            if trace:
                start_idx = state_map[parsed.start_state]
                start_mask = self._compute_mask_for_state(
                    start_idx, table, self.token_bytes_arr, self.token_lens
                )
                n_valid = start_mask.sum()
                print(
                    f"  {name}: {n_states} states, "
                    f"{n_valid}/{self.vocab_size} tokens valid at start "
                    f"({n_valid / self.vocab_size * 100:.1f}%)"
                )

    def _compute_completion_masks(self, name, pref_parsed, _pref_state_map,
                                  orig_parsed, orig_table, orig_state_map):
        """Compute completion masks by mapping prefix DFA states to original DFA states.

        A token is a "completion" token at a prefix state if it can walk the
        original DFA to an accept state at ANY byte position during consumption.
        """
        # Build a mapping from prefix DFA states to original DFA states.
        # Both DFAs have the same byte-level transitions (same language).
        # Walk them in parallel on every possible path to build the correspondence.
        pref_to_orig = {}
        pref_to_orig[pref_parsed.start_state] = orig_parsed.start_state
        # BFS through the prefix DFA, tracking corresponding original state
        visited = {pref_parsed.start_state}
        queue = [(pref_parsed.start_state, orig_parsed.start_state)]
        while queue:
            ps, os_ = queue.pop(0)
            pref_to_orig[ps] = os_
            for byte_val in range(256):
                p_next = pref_parsed.transitions.get((ps, byte_val))
                o_next = orig_parsed.transitions.get((os_, byte_val))
                if p_next is not None and o_next is not None and p_next not in visited:
                    visited.add(p_next)
                    queue.append((p_next, o_next))

        # For each prefix DFA state that maps to an original DFA state,
        # compute the completion mask using the original DFA.
        orig_accept_set = orig_parsed.accept_states
        for pref_state, orig_state in pref_to_orig.items():
            if orig_state not in orig_state_map:
                continue
            orig_idx = orig_state_map[orig_state]
            # Vectorized: walk original DFA and check if any intermediate
            # position reaches an accept state.
            mask = self._compute_completion_mask_for_state(
                orig_idx, orig_table, orig_state_map, orig_accept_set,
                self.token_bytes_arr, self.token_lens
            )
            self.state_masks[(name, pref_state, "completion")] = torch.tensor(
                mask, dtype=torch.bool, device=self.device
            )

    def _compute_completion_mask_for_state(self, start_idx, trans_table,
                                           state_to_idx, accept_states,
                                           token_bytes_arr, token_lens):
        """Vectorized: check which tokens reach an accept state at any byte during walk."""
        vocab_size = len(token_lens)
        current_states = np.full(vocab_size, start_idx, dtype=np.int32)
        alive = np.ones(vocab_size, dtype=bool)
        completed = np.zeros(vocab_size, dtype=bool)

        # Build accept index set (using mapped indices)
        accept_indices = set()
        for as_ in accept_states:
            if as_ in state_to_idx:
                accept_indices.add(state_to_idx[as_])

        for byte_pos in range(self.max_token_len):
            active = alive & (byte_pos < token_lens)
            if not active.any():
                break
            active_indices = np.where(active)[0]
            bytes_at_pos = token_bytes_arr[active_indices, byte_pos]
            next_states = trans_table[current_states[active_indices], bytes_at_pos]
            dead = next_states == -1
            alive[active_indices[dead]] = False
            current_states[active_indices[~dead]] = next_states[~dead]
            # Check if any reached an accept state
            for ai in accept_indices:
                completed[active_indices[~dead]] |= (next_states[~dead] == ai)

        return completed

    def _compute_mask_for_state(self, start_idx, trans_table, token_bytes_arr, token_lens):
        """Vectorized DFA walk from a single start state over all vocab tokens."""
        vocab_size = len(token_lens)
        current_states = np.full(vocab_size, start_idx, dtype=np.int32)
        alive = np.ones(vocab_size, dtype=bool)

        for byte_pos in range(self.max_token_len):
            active = alive & (byte_pos < token_lens)
            if not active.any():
                break
            active_indices = np.where(active)[0]
            bytes_at_pos = token_bytes_arr[active_indices, byte_pos]
            next_states = trans_table[current_states[active_indices], bytes_at_pos]
            dead = next_states == -1
            alive[active_indices[dead]] = False
            current_states[active_indices[~dead]] = next_states[~dead]

        return alive

    def _build_boundary_mask(self):
        """Build a mask for terminal boundaries: union of all terminal start masks."""
        combined = torch.zeros(self.vocab_size, dtype=torch.bool)
        for name in self.terminal_names:
            parsed = self.terminal_parsed[name]
            key = (name, parsed.start_state)
            if key in self.state_masks:
                combined |= self.state_masks[key].cpu()
        self.boundary_mask = combined.to(self.device)

    def _ensure_boundary_mask(self):
        if not hasattr(self, "boundary_mask"):
            self._build_boundary_mask()

    def get_lex_state(self, prefix_text):
        """Determine the lex state after consuming prefix_text.

        Returns (mid_states, terminal_prefixes) where:
        - mid_states: list of (terminal_name, dfa_state) pairs, or None if at boundary
        - terminal_prefixes: set of terminal prefix tuples (for CFG lookup).
          Multiple prefixes arise from ambiguous lexings (e.g. "name" matching
          both lexProperty and lexValue).

        Uses the parsed DFA walk (not accept_prefix_string) because
        rustformlang's internal state indices don't match to_text() names.
        """
        if not prefix_text:
            return None, {()}

        lexings = lex(prefix_text, self.compiled_lex_map, is_first=True,
                      strip_chars=self.strip_chars)
        if not lexings:
            return None, {()}

        # Collect all possible mid-terminal states and boundary prefixes
        mid_states = []
        at_boundary = False
        boundary_prefixes = set()

        for terminals, _, last_p in lexings:
            if last_p is not None and terminals:
                current_terminal = terminals[-1]
                parsed = self.terminal_parsed.get(current_terminal)
                if parsed is not None:
                    state = parsed.walk_bytes(last_p.encode("utf-8"))
                    if state is not None:
                        mid_states.append((current_terminal, state))
            else:
                at_boundary = True
                boundary_prefixes.add(terminals if terminals else ())

        if not mid_states:
            return None, boundary_prefixes or {()}

        if at_boundary:
            return None, boundary_prefixes

        # For mid-terminal states, collect all possible prefixes
        mid_prefixes = set()
        for terminals, _, last_p in lexings:
            if last_p is not None and terminals:
                mid_prefixes.add(terminals[:-1])
        return mid_states, mid_prefixes or {()}

    def _get_valid_next_terminals(self, terminal_prefix):
        """Compute which terminals can follow the given prefix using the CFG.

        Results are cached for efficiency.
        """
        if self.constraint_lang is None:
            return None  # No CFG, can't filter

        key = tuple(terminal_prefix)
        if key in self._next_terminal_cache:
            return self._next_terminal_cache[key]

        all_terminals = self.constraint_lang.get_terminals()
        valid = set()
        for t in all_terminals:
            enfa = ENFA()
            enfa.set_start_state("q0")
            for i, term in enumerate(terminal_prefix):
                enfa.add_transition(f"q{i}", term, f"q{i + 1}")
            n = len(terminal_prefix)
            enfa.add_transition(f"q{n}", t, f"q{n + 1}")
            for t2 in all_terminals:
                enfa.add_transition(f"q{n + 1}", t2, f"q{n + 1}")
            enfa.add_accept_state(f"q{n + 1}")
            dfa = minimize_enfa_threaded(enfa)
            if not is_intersection_empty_threaded(self.constraint_lang, dfa, timeout=10):
                valid.add(t)

        self._next_terminal_cache[key] = valid
        return valid

    def _get_cfg_boundary_mask(self, terminal_prefix):
        """Get mask restricted to terminals valid after the given prefix."""
        valid_terminals = self._get_valid_next_terminals(terminal_prefix)
        if valid_terminals is None:
            self._ensure_boundary_mask()
            return self.boundary_mask  # No CFG, use unrestricted boundary mask

        combined = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for name in valid_terminals:
            parsed = self.terminal_parsed.get(name)
            if parsed is not None:
                key = (name, parsed.start_state)
                if key in self.state_masks:
                    combined |= self.state_masks[key]
        return combined

    def get_mask_for_state(self, lex_state, terminal_prefixes=None):
        """Get the combined token mask for a lex state.

        Args:
            lex_state: list of (terminal, state) pairs, or None for boundary.
            terminal_prefixes: set of terminal prefix tuples for CFG lookup.
                Multiple prefixes arise from ambiguous lexings.

        Returns:
            Boolean tensor of shape (vocab_size,) where True = allowed.
        """
        if terminal_prefixes is None:
            terminal_prefixes = {()}

        if lex_state is None:
            # At boundary: union of CFG-aware masks over all possible prefixes
            combined = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            for prefix in terminal_prefixes:
                combined |= self._get_cfg_boundary_mask(prefix)
            return combined

        # Mid-terminal: union of masks for all possible (terminal, state) pairs.
        combined = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for terminal, state in lex_state:
            key = (terminal, state)
            if key in self.state_masks:
                combined |= self.state_masks[key]
            comp_key = (terminal, state, "completion")
            if comp_key in self.state_masks:
                combined |= self.state_masks[comp_key]

        return combined

    def apply_prefix_masks(self, logits, token_ids, block_start, block_end, mask_id,
                           gen_start=None, soft_bias=None):
        """Apply lex-level masks at positions with known lex state.

        Walks the placed prefix to determine the lex state, then masks
        the first masked position after the prefix. Also masks isolated
        gap positions (with placed left context).

        Args:
            gen_start: Start position of generation area. If provided,
                the full text from gen_start is used for lex state
                analysis (needed for blocks > 0 to get correct context).
            soft_bias: If set (float), add this value to valid token logits
                instead of setting invalid tokens to -inf. This preserves
                the model's joint distribution while gently steering toward
                grammar-valid tokens.

        Returns: number of positions masked.
        """
        logit_dim = logits.shape[-1]
        n_masked = 0

        # Find contiguous placed prefix from block_start
        prefix_end = block_start
        while prefix_end < block_end and token_ids[prefix_end].item() != mask_id:
            prefix_end += 1

        if prefix_end >= block_end:
            return 0  # All placed, nothing to mask

        # Get the text of the placed prefix. Use full context from gen_start
        # so lex state analysis includes all preceding blocks.
        text_start = gen_start if gen_start is not None else block_start
        if prefix_end > text_start:
            prefix_text = self.tokenizer.decode(token_ids[text_start:prefix_end])
        else:
            prefix_text = ""

        # Determine lex state at the end of the prefix
        lex_state, terminal_prefix = self.get_lex_state(prefix_text)

        # Get the mask (CFG-aware at boundaries)
        allow_mask = self.get_mask_for_state(lex_state, terminal_prefix)
        # Always allow EOS tokens
        allow_mask = allow_mask | self.eos_mask[:self.vocab_size]

        # Pad mask to logit dimension if needed
        if allow_mask.shape[0] < logit_dim:
            # Extra logit positions (padding tokens) are blocked
            pad = torch.zeros(logit_dim - allow_mask.shape[0],
                              dtype=torch.bool, device=self.device)
            allow_mask = torch.cat([allow_mask, pad])

        # Apply at the first masked position
        pos = prefix_end
        if pos < block_end and token_ids[pos].item() == mask_id:
            self._apply_bias_at_pos(logits, 0, pos, allow_mask, soft_bias)
            n_masked += 1

        # Also apply at other masked positions with placed left context.
        self._ensure_boundary_mask()
        for p in range(prefix_end + 1, block_end):
            if token_ids[p].item() != mask_id:
                continue
            if p > 0 and token_ids[p - 1].item() != mask_id:
                local_start = p - 1
                while local_start > block_start and token_ids[local_start - 1].item() != mask_id:
                    local_start -= 1
                local_prefix = self.tokenizer.decode(token_ids[local_start:p])
                local_state, _ = self.get_lex_state(local_prefix)
                local_mask = self.get_mask_for_state(local_state) | self.eos_mask[:self.vocab_size]
                if local_mask.shape[0] < logit_dim:
                    pad = torch.zeros(logit_dim - local_mask.shape[0],
                                      dtype=torch.bool, device=self.device)
                    local_mask = torch.cat([local_mask, pad])
                self._apply_bias_at_pos(logits, 0, p, local_mask, soft_bias)
                n_masked += 1

        return n_masked

    @staticmethod
    def _apply_bias_at_pos(logits, batch, pos, allow_mask, soft_bias):
        """Apply constraint at a position using the specified mode.

        Modes:
        - None: hard mask, set invalid to -inf
        - float: soft bias, add value to valid token logits
        - "kl_project": KL projection, renormalize logits to valid tokens.
          Equivalent to setting invalid to -inf for token selection, but
          preserves original logits elsewhere for confidence computation.
          Mathematically: argmin_q KL(p || q) s.t. support(q) ⊆ C
          Solution: q(x) = p(x) / Z for x in C, 0 otherwise.
          In logit space: subtract log(Z) from valid logits, -inf for invalid.
          But since we only care about relative ranking within valid set,
          just set invalid to -inf (same effect on softmax within C).
        """
        if soft_bias == "kl_project":
            logits[batch, pos, ~allow_mask] = -float("inf")
        elif soft_bias is not None:
            logits[batch, pos, allow_mask] += soft_bias
        else:
            logits[batch, pos, ~allow_mask] = -float("inf")
