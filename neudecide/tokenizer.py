"""SentencePiece tokenizer, built from the tokenizer.json shipped with the model.

tokenizer.json holds a SentencePiece BPE model as JSON: its pieces (text, score,
type) plus the normalizer flags encoding depends on. It's turned back into a
ModelProto here so the sentencepiece library does the encoding and decoding.
"""

import json
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2


def model_proto(spec):
    """tokenizer.json contents -> a SentencePiece ModelProto (identity normalizer)."""
    if spec.get("model_type") != "bpe":
        raise ValueError(f"unsupported tokenizer model_type {spec.get('model_type')!r}")
    proto = sp_pb2.ModelProto()
    proto.trainer_spec.model_type = sp_pb2.TrainerSpec.BPE
    proto.trainer_spec.unk_id = spec["unk_id"]
    proto.trainer_spec.byte_fallback = spec.get("byte_fallback", False)
    proto.normalizer_spec.name = "identity"
    for flag in ("add_dummy_prefix", "remove_extra_whitespaces", "escape_whitespaces"):
        setattr(proto.normalizer_spec, flag, spec.get(flag, True))
    piece_type = sp_pb2.ModelProto.SentencePiece.Type
    for piece, score, typ in spec["pieces"]:
        proto.pieces.add(piece=piece, score=score, type=piece_type.Value(typ))
    return proto


class Tokenizer:
    def __init__(self, spec):
        proto = model_proto(spec)
        self.sp = spm.SentencePieceProcessor(model_proto=proto.SerializeToString())
        # The same model without the dummy prefix or whitespace cleanup, for
        # tokenizing text fragments byte-for-byte (constrained decoding needs that).
        proto.normalizer_spec.add_dummy_prefix = False
        proto.normalizer_spec.remove_extra_whitespaces = False
        self.sp_fragment = spm.SentencePieceProcessor(model_proto=proto.SerializeToString())

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self):
        return self.sp.vocab_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode([int(i) for i in ids])
