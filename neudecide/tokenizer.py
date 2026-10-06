"""SentencePiece BPE encode / decode in pure Python.

Reads the tokenizer.json shipped with the model (pieces with scores and types,
plus the normalizer flags encoding depends on) so the package doesn't need the
sentencepiece wheel. Covers the configuration NeuDecide's tokenizer uses:
identity normalizer, remove_extra_whitespaces, add_dummy_prefix, byte_fallback.
"""

import heapq
import json
import re
from pathlib import Path

WS = "▁"  # ▁
_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class Tokenizer:
    def __init__(self, spec):
        if spec.get("model_type") != "bpe":
            raise ValueError(f"unsupported tokenizer model_type {spec.get('model_type')!r}")
        self.spec = spec
        self.pieces = [p[0] for p in spec["pieces"]]
        self.scores = [p[1] for p in spec["pieces"]]
        self.types = [p[2] for p in spec["pieces"]]
        self.unk_id = spec["unk_id"]
        # Pieces a merge may produce (NORMAL + USER_DEFINED); control / unknown /
        # byte pieces never come out of merging text.
        self.piece_to_id = {}
        self.byte_to_id = {}
        user_defined = []
        for i, (piece, typ) in enumerate(zip(self.pieces, self.types)):
            if typ in ("NORMAL", "USER_DEFINED"):
                self.piece_to_id[piece] = i
            if typ == "USER_DEFINED":
                user_defined.append(piece)
            if typ == "BYTE":
                self.byte_to_id[int(piece[3:5], 16)] = i
        self.user_defined = sorted(user_defined, key=len, reverse=True)  # longest match first

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self):
        return len(self.pieces)

    def normalize(self, text):
        """normalizer.cc with an identity charsmap: only U+0020 is whitespace."""
        s = text
        if self.spec.get("remove_extra_whitespaces", True):
            s = re.sub(r" {2,}", " ", s.strip(" "))
        if not s:
            return ""
        if self.spec.get("add_dummy_prefix", True):
            s = " " + s
        return s.replace(" ", WS) if self.spec.get("escape_whitespaces", True) else s

    def _split(self, normalized):
        """Initial symbols: user-defined pieces matched whole, else one character each."""
        out, i = [], 0
        while i < len(normalized):
            for u in self.user_defined:
                if normalized.startswith(u, i):
                    out.append(u)
                    i += len(u)
                    break
            else:
                out.append(normalized[i])
                i += 1
        return out

    def encode(self, text):
        piece = self._split(self.normalize(text))
        n = len(piece)
        if not n:
            return []
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n)) + [-1]
        heap = []

        def push(left, right):
            if left < 0 or right < 0:
                return
            merged = piece[left] + piece[right]
            pid = self.piece_to_id.get(merged)
            if pid is not None:
                # highest score first, ties to the leftmost pair (bpe_model.cc)
                heapq.heappush(heap, (-self.scores[pid], left, right, merged))

        for i in range(1, n):
            push(i - 1, i)
        while heap:
            _, left, right, merged = heapq.heappop(heap)
            # stale: one side was merged away or has grown since this pair was queued
            if not piece[left] or not piece[right] or piece[left] + piece[right] != merged:
                continue
            piece[left], piece[right] = merged, ""
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            push(prev[left], left)
            push(left, nxt[left])

        ids, i = [], 0
        while i >= 0:
            pid = self.piece_to_id.get(piece[i])
            if pid is not None:
                ids.append(pid)
            elif self.spec.get("byte_fallback", False):
                ids.extend(self.byte_to_id[b] for b in piece[i].encode("utf-8"))
            else:
                ids.append(self.unk_id)
            i = nxt[i]
        return ids

    def decode(self, ids):
        """Control pieces vanish, byte pieces are re-assembled as UTF-8, ▁ becomes
        a space and the dummy-prefix space is dropped."""
        out, buf, first = [], bytearray(), True
        for i in ids:
            typ = self.types[i]
            if typ == "CONTROL":
                continue
            if typ == "BYTE":
                buf.append(int(self.pieces[i][3:5], 16))
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            s = " ⁇ " if typ == "UNKNOWN" else self.pieces[i]
            s = s.replace(WS, " ")
            if first and self.spec.get("add_dummy_prefix", True) and s.startswith(" "):
                s = s[1:]
            first = False
            out.append(s)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token_strings(self):
        """The text each id contributes to the output, for grammar checks:
        ▁ -> space, byte pieces -> their byte as a character, control /
        unknown pieces -> "" (never chosen by constrained decoding)."""
        out = []
        for piece, typ in zip(self.pieces, self.types):
            if typ in ("CONTROL", "UNKNOWN", "UNUSED"):
                out.append("")
            elif typ == "BYTE":
                m = _BYTE_RE.match(piece)
                out.append(chr(int(m.group(1), 16)) if m else "")
            else:
                out.append(piece.replace(WS, " "))
        return out
