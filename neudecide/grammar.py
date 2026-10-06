"""Grammar-constrained decoding for tool calls.

The model writes a JSON list of calls. `ToolCallGrammar` is a character-level
acceptor for exactly that shape, specialised to the tools offered:

    [ ]  |  [ CALL (, CALL)* ]      CALL := {"name": NAME, "arguments": {ARGS}}
    ARGS := (KEY: VALUE (, KEY: VALUE)*)?

NAME must be one of the tool names, KEY one of that tool's parameters (each
at most once), and VALUE any well-formed JSON value. Whitespace between JSON
tokens is free, so compact and `json.dumps`-spaced output are both accepted.

Decoding policy lives in a `Selector`: anything with `select(logits) -> id`.
`GreedySelector` is a plain argmax; `ConstrainedSelector` walks candidates in
descending logit order and takes the first one whose text the grammar accepts,
which for greedy decoding is the same as masking then taking the argmax, but
usually only looks at the top candidate.
"""

import json
import re

import numpy as np

_NUMBER_RE = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")
_NUMBER_CHARS = frozenset("-+.eE0123456789")
_WS = frozenset(" \t\n\r")
_WORDS = ("true", "false", "null")


class Trie:
    __slots__ = ("children", "terminal")

    def __init__(self, words=()):
        self.children = {}
        self.terminal = False
        for word in words:
            node = self
            for ch in word:
                node = node.children.setdefault(ch, Trie())
            node.terminal = True


def _json_str(s):
    """A string as it appears between the quotes of a json.dumps literal."""
    return json.dumps(s)[1:-1]


class ToolSchema:
    """Valid tool names and, per tool, valid argument keys.

    `tools` is a list (or JSON string) of tool definitions, each with a `name`
    and `parameters` either as JSON Schema (`{"type": "object", "properties":
    {...}}`) or as a flat `{key: {...}}` mapping."""

    def __init__(self, tools):
        if isinstance(tools, str):
            tools = json.loads(tools)
        self.params = {}
        for tool in tools or []:
            if not isinstance(tool, dict) or "name" not in tool:
                continue
            params = tool.get("parameters") or {}
            # a flat mapping may itself have a parameter called "type", so check the schema shape
            is_schema = params.get("type") == "object" and isinstance(
                params.get("properties"), dict
            )
            props = params["properties"] if is_schema else params
            keys = list(props) if isinstance(props, dict) else []
            self.params[_json_str(tool["name"])] = [_json_str(k) for k in keys]
        self.names = Trie(self.params)
        self._keys = {}

    def keys(self, name, used=()):
        """Trie of `name`'s argument keys not yet in `used`."""
        cache_key = (name, frozenset(used))
        if cache_key not in self._keys:
            self._keys[cache_key] = Trie(k for k in self.params.get(name, []) if k not in used)
        return self._keys[cache_key]


# Top-level phases.
(
    START,
    LIST_OPEN,
    LITERAL,
    NAME,
    ARGS_OPEN,
    ARGS_NEXT_KEY,
    KEY,
    VALUE,
    AFTER_VALUE,
    AFTER_CALL,
    END,
) = range(11)
# Sub-states while reading an argument value.
V_START, V_ARR_FIRST, V_STRING, V_NUMBER, V_WORD, V_OBJ_FIRST, V_OBJ_KEY, V_OBJ_COLON, V_AFTER = (
    range(9)
)


class ToolCallGrammar:
    """Accepts text one character at a time. `feed` returns False as soon as
    the text can no longer be the prefix of a valid output, leaving the state
    undefined -- so `clone()` before trying a candidate."""

    __slots__ = (
        "schema",
        "phase",
        "literals",
        "then",
        "node",
        "buf",
        "fn",
        "used",
        "vstate",
        "vstack",
        "vesc",
        "vtext",
    )

    def __init__(self, schema):
        self.schema = schema if isinstance(schema, ToolSchema) else ToolSchema(schema)
        self.phase = START
        self.literals, self.then = [], None
        self.node, self.buf, self.fn, self.used = None, "", None, ()
        self.vstate, self.vstack, self.vesc, self.vtext = V_START, (), False, ""

    def clone(self):
        c = ToolCallGrammar.__new__(ToolCallGrammar)
        for slot in ToolCallGrammar.__slots__:
            setattr(c, slot, getattr(self, slot))
        c.literals = list(self.literals)
        return c

    @property
    def done(self):
        return self.phase == END

    def feed_text(self, text):
        return all(self.feed(ch) for ch in text)

    def accepts(self, text):
        """Whether `text` (from the start) is a complete, valid output."""
        g = ToolCallGrammar(self.schema)
        return g.feed_text(text) and g.done

    # -- structure ----------------------------------------------------------

    def _expect(self, literals, then):
        """Match `literals` in order, whitespace allowed before each one."""
        self.phase, self.literals, self.then, self.buf = LITERAL, list(literals), then, ""

    def _enter(self, phase):
        if phase == NAME:
            self.phase, self.node, self.buf = NAME, self.schema.names, ""
        elif phase == VALUE:
            self.phase, self.vstate, self.vstack = VALUE, V_START, ()
        else:
            self.phase = phase

    def _open_key(self):
        """Opens an argument key, only if an unused one is left, so the grammar
        never admits a prefix it can't complete."""
        trie = self.schema.keys(self.fn, self.used)
        if not trie.children:
            return False
        self.phase, self.node, self.buf = KEY, trie, ""
        return True

    def feed(self, ch):
        p = self.phase
        if p == LITERAL:
            return self._feed_literal(ch)
        if p in (NAME, KEY):
            return self._feed_trie(ch)
        if p == VALUE:
            return self._feed_value(ch)
        if ch in _WS:
            return True
        if p == START:
            if ch == "[":
                self.phase = LIST_OPEN
                return True
        elif p == LIST_OPEN:
            if ch == "]":
                self.phase = END
                return True
            if ch == "{":
                self._expect(['"name"', ":", '"'], NAME)
                return True
        elif p == ARGS_OPEN:
            if ch == "}":
                self._expect(["}"], AFTER_CALL)
                return True
            if ch == '"':
                return self._open_key()
        elif p == ARGS_NEXT_KEY:
            if ch == '"':
                return self._open_key()
        elif p == AFTER_VALUE:
            if ch == "," and self.schema.keys(self.fn, self.used).children:
                self.phase = ARGS_NEXT_KEY
                return True
            if ch == "}":
                self._expect(["}"], AFTER_CALL)
                return True
        elif p == AFTER_CALL:
            if ch == ",":
                self._expect(["{", '"name"', ":", '"'], NAME)
                return True
            if ch == "]":
                self.phase = END
                return True
        return False

    def _feed_literal(self, ch):
        head = self.literals[0]
        if ch in _WS and not self.buf:
            return True
        if ch != head[len(self.buf)]:
            return False
        self.buf += ch
        if self.buf == head:
            self.buf = ""
            self.literals.pop(0)
            if not self.literals:
                self._enter(self.then)
        return True

    def _feed_trie(self, ch):
        if ch == '"':
            if not self.node.terminal:
                return False
            if self.phase == NAME:
                self.fn, self.used = self.buf, ()
                self._expect([",", '"arguments"', ":", "{"], ARGS_OPEN)
            else:
                self.used = self.used + (self.buf,)
                self._expect([":"], VALUE)
            return True
        node = self.node.children.get(ch)
        if node is None:
            return False
        self.node = node
        self.buf += ch
        return True

    # -- argument values: any well-formed JSON --------------------------------
    # vstack holds the containers ("{" / "[") open inside the value.

    def _value_closed(self):
        if self.vstack:
            self.vstate = V_AFTER
        else:
            self.phase = AFTER_VALUE

    def _feed_value(self, ch):
        s = self.vstate
        if s in (V_STRING, V_OBJ_KEY) and self.vtext == '"':  # inside a string
            if self.vesc:
                self.vesc = False
            elif ch == "\\":
                self.vesc = True
            elif ch == '"':
                self.vtext = ""
                if s == V_STRING:
                    self._value_closed()
                else:
                    self.vstate = V_OBJ_COLON
            elif ch in "\n\r":
                return False
            return True
        if s == V_NUMBER:
            if ch in _NUMBER_CHARS:
                self.vtext += ch
                return True
            if not _NUMBER_RE.fullmatch(self.vtext):
                return False
            self.vtext = ""
            self._value_closed()
            return self.feed(ch)  # the terminating character belongs to the enclosing grammar
        if s == V_WORD:
            self.vtext += ch
            if not any(w.startswith(self.vtext) for w in _WORDS):
                return False
            if self.vtext in _WORDS:
                self.vtext = ""
                self._value_closed()
            return True
        if ch in _WS:
            return True
        if s in (V_START, V_ARR_FIRST):
            if ch == "]" and s == V_ARR_FIRST:
                self.vstack = self.vstack[:-1]
                self._value_closed()
            elif ch == '"':
                self.vstate, self.vtext, self.vesc = V_STRING, '"', False
            elif ch in "-0123456789":
                self.vstate, self.vtext = V_NUMBER, ch
            elif ch in "tfn":
                self.vstate, self.vtext = V_WORD, ch
            elif ch == "{":
                self.vstack, self.vstate = self.vstack + ("{",), V_OBJ_FIRST
            elif ch == "[":
                self.vstack, self.vstate = self.vstack + ("[",), V_ARR_FIRST
            else:
                return False
            return True
        if s in (V_OBJ_FIRST, V_OBJ_KEY):
            if ch == "}" and s == V_OBJ_FIRST:
                self.vstack = self.vstack[:-1]
                self._value_closed()
            elif ch == '"':
                self.vstate, self.vtext, self.vesc = V_OBJ_KEY, '"', False
            else:
                return False
            return True
        if s == V_OBJ_COLON:
            if ch != ":":
                return False
            self.vstate = V_START
            return True
        if s == V_AFTER:
            top = self.vstack[-1]
            if ch == ",":
                self.vstate = V_OBJ_KEY if top == "{" else V_START
            elif (ch == "}" and top == "{") or (ch == "]" and top == "["):
                self.vstack = self.vstack[:-1]
                self._value_closed()
            else:
                return False
            return True
        return False


# --------------------------------------------------------------------------
# Token selection
# --------------------------------------------------------------------------


class GreedySelector:
    """Plain argmax."""

    def select(self, logits):
        return int(np.argmax(logits))


class ConstrainedSelector:
    """Greedy decoding restricted to `grammar`, for one sequence.

    The first step is forced to `tool_call_id` (every answer starts with it)
    and `eos_id` is forced exactly when the grammar is complete. Tokens with
    no text (control / unknown pieces) are never chosen."""

    top_k = 64

    def __init__(self, grammar, token_strings, tool_call_id, eos_id):
        self.grammar = grammar
        self.token_strings = token_strings
        self.tool_call_id = tool_call_id
        self.eos_id = eos_id
        self.started = False

    def _try(self, token_id):
        text = self.token_strings[token_id]
        if not text:
            return None
        g = self.grammar.clone()
        return g if g.feed_text(text) else None

    def select(self, logits):
        if not self.started:
            self.started = True
            return self.tool_call_id
        if self.grammar.done:
            return self.eos_id
        logits = np.asarray(logits).reshape(-1)
        k = min(self.top_k, logits.size)
        top = np.argpartition(-logits, k - 1)[:k]
        top = top[np.argsort(-logits[top], kind="stable")]
        for candidates in (top, None):
            if candidates is None:  # rare: none of the top k fit, try the rest in order
                order = np.argsort(-logits, kind="stable")
                candidates = order[~np.isin(order, top)]
            for token_id in candidates.tolist():
                g = self._try(token_id)
                if g is not None:
                    self.grammar = g
                    return token_id
        return self.eos_id  # nothing fits; shouldn't happen
