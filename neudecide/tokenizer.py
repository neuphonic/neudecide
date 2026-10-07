"""SentencePiece tokenizer, from the model's tokenizer.model or tokenizer.json.

tokenizer.model is SentencePiece's own file (a serialized ModelProto).
tokenizer.json holds the same BPE model as JSON: its pieces (text, score, type)
plus the normalizer flags encoding depends on; it's turned back into a
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
    def __init__(self, model):
        """model: a ModelProto, or tokenizer.json's contents."""
        proto = model if isinstance(model, sp_pb2.ModelProto) else model_proto(model)
        serialized = proto.SerializeToString()
        self.sp = spm.SentencePieceProcessor(model_proto=serialized)
        # The same model without the dummy prefix or whitespace cleanup, for
        # tokenizing text fragments byte-for-byte (constrained decoding needs that).
        fragment = sp_pb2.ModelProto.FromString(serialized)
        fragment.normalizer_spec.add_dummy_prefix = False
        fragment.normalizer_spec.remove_extra_whitespaces = False
        self.sp_fragment = spm.SentencePieceProcessor(model_proto=fragment.SerializeToString())

    @classmethod
    def from_file(cls, path):
        """A tokenizer.model, or a tokenizer.json."""
        path = Path(path)
        if path.suffix == ".json":
            return cls(json.loads(path.read_text(encoding="utf-8")))
        return cls(sp_pb2.ModelProto.FromString(path.read_bytes()))

    def __len__(self):
        return self.sp.vocab_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode([int(i) for i in ids])
