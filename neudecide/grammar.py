"""Grammar-constrained decoding for tool calls, with llguidance.

The model writes a JSON list of calls. `tool_call_grammar` describes exactly
that shape, specialised to the tools offered:

    [ ]  |  [ CALL (, CALL)* ]      CALL := {"name": NAME, "arguments": {ARGS}}

NAME must be one of the tool names, and ARGS may only use that tool's
parameter names as keys (in any order, at most as many keys as it has
parameters), each with any well-formed JSON value. Whitespace between JSON
tokens is free, so compact and `json.dumps`-spaced output are both accepted.

Decoding policy lives in a `Selector`: anything with `select(logits) -> id`.
`GreedySelector` is a plain argmax; `ConstrainedSelector` masks the logits to
the tokens the grammar allows next and takes the argmax of what's left.
"""

import json
import re

import numpy as np
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
from llguidance.numpy import apply_token_bitmask_inplace

try:
    from ._rust import masked_argmax as _rust_masked_argmax
except ImportError:
    _rust_masked_argmax = None

WS = "▁"  # SentencePiece's whitespace marker


def tool_parameters(tools):
    """{tool name: [argument keys]} for a list (or JSON string) of tool
    definitions, each with a `name` and `parameters` either as JSON Schema
    (`{"type": "object", "properties": {...}}`) or as a flat `{key: {...}}` mapping."""
    if isinstance(tools, str):
        tools = json.loads(tools)
    params = {}
    for tool in tools or []:
        if not isinstance(tool, dict) or "name" not in tool:
            continue
        p = tool.get("parameters") or {}
        # a flat mapping may itself have a parameter called "type", so check the schema shape
        is_schema = p.get("type") == "object" and isinstance(p.get("properties"), dict)
        props = p["properties"] if is_schema else p
        params[tool["name"]] = list(props) if isinstance(props, dict) else []
    return params


def tool_call_grammar(tools):
    """A Lark grammar, in llguidance's dialect, for the answer to `tools`."""
    calls = []
    for name, keys in tool_parameters(tools).items():
        # patternProperties, not properties: llguidance emits properties in
        # schema order, and the model needn't. maxProperties bounds repeated
        # keys, which a JSON grammar can't rule out.
        arguments = {
            "type": "object",
            "patternProperties": {f"^{re.escape(k)}$": {} for k in keys},
            "additionalProperties": False,
            "maxProperties": len(keys),
        }
        calls.append(
            {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": arguments},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    schema = {
        "type": "array",
        "items": {"anyOf": calls} if calls else False,
        # \uXXXX escapes in values, as json.dumps writes non-ASCII text by default
        "x-guidance": {"json_allow_general_unicode_escapes": True},
    }
    # leading whitespace: SentencePiece's dummy prefix puts a space before the "["
    return "start: WS? calls\n" r"WS: /[ \t\n\r]+/" "\n" f"calls: %json {json.dumps(schema)}"


class _Vocab:
    """The attributes llguidance.TokenizerWrapper reads off a tokenizer."""

    def __init__(self, tokens, special_token_ids, eos_token_id, encode):
        self.tokens = tokens
        self.special_token_ids = special_token_ids
        self.eos_token_id = eos_token_id
        self.bos_token_id = None
        self._encode = encode

    def __call__(self, text):
        return self._encode(text)


def ll_tokenizer(tokenizer, eos_id, special_ids=()):
    """The SentencePiece vocabulary as an llguidance tokenizer: each piece as
    the bytes it adds to the output (▁ -> space, <0xNN> -> that byte).
    Control / unknown / unused pieces and `special_ids` are special tokens,
    which the grammar never produces."""
    sp = tokenizer.sp
    tokens, special = [], set(special_ids)
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        if sp.is_byte(i):
            tokens.append(bytes([int(piece[1:-1], 16)]))
        else:
            tokens.append(piece.replace(WS, " ").encode("utf-8"))
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            special.add(i)
    vocab = _Vocab(tokens, sorted(special), eos_id, tokenizer.sp_fragment.encode)
    return LLTokenizer(TokenizerWrapper(vocab))


def tool_call_matcher(ll_tok, tools):
    """A fresh llguidance matcher for the answer to `tools`."""
    matcher = LLMatcher(ll_tok, tool_call_grammar(tools), log_level=0)
    if matcher.is_error():
        raise ValueError(f"couldn't build the tool-call grammar: {matcher.get_error()}")
    return matcher


# --------------------------------------------------------------------------
# Token selection
# --------------------------------------------------------------------------


class GreedySelector:
    """Plain argmax."""

    def select(self, logits):
        return int(np.argmax(logits))


class ConstrainedSelector:
    """Greedy decoding restricted to `matcher`'s grammar, for one sequence.

    The first step is forced to `tool_call_id` (every answer starts with it)
    and `eos_id` is forced as soon as the grammar is complete."""

    def __init__(self, matcher, tool_call_id, eos_id):
        self.matcher = matcher
        self.tool_call_id = tool_call_id
        self.eos_id = eos_id
        self.started = False

    def select(self, logits):
        if not self.started:
            self.started = True
            return self.tool_call_id
        if self.matcher.is_accepting():
            return self.eos_id
        mask = np.frombuffer(self.matcher.compute_bitmask(), dtype=np.int32)
        if _rust_masked_argmax is not None:
            token = _rust_masked_argmax(np.asarray(logits, dtype=np.float32), mask)
        else:
            logits = np.array(logits, dtype=np.float32).reshape(1, -1)
            apply_token_bitmask_inplace(logits, mask)
            token = int(np.argmax(logits))
        if not self.matcher.consume_token(token):
            return self.eos_id  # nothing fits; shouldn't happen
        return token
