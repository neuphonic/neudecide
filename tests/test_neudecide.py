import io
import json
import os
import wave

import numpy as np
import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from neudecide.audio import load_audio, prepare_audio, resample
from neudecide.grammar import ConstrainedSelector, ll_tokenizer, tool_call_matcher
from neudecide.model import complete_calls
from neudecide.tokenizer import Tokenizer

TOOLS = [
    {
        "name": "getWeather",
        "description": "Gets the weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
    {
        "name": "getTime",
        "description": "Tells the current time.",
        "parameters": {"type": "object", "properties": {}},
    },
]
EOS, TOOL_CALL = 1, 4


@pytest.fixture(scope="module")
def sp_model():
    """A tiny SentencePiece BPE model laid out like NeuDecide's: <pad> </s> <s>
    <unk> <tool_call> <tools>, then byte pieces; identity normalizer."""
    corpus = [
        json.dumps(TOOLS),
        json.dumps([{"name": "getWeather", "arguments": {"city": "Paris", "days": 3}}]),
        "what is the weather in paris",
        "what time is it",
    ]
    out = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus * 20),
        model_writer=out,
        model_type="bpe",
        vocab_size=400,
        hard_vocab_limit=False,
        byte_fallback=True,
        pad_id=0,
        eos_id=EOS,
        bos_id=2,
        unk_id=3,
        user_defined_symbols=["<tool_call>", "<tools>"],
        normalization_rule_name="identity",
        minloglevel=2,
    )
    return out.getvalue()


@pytest.fixture(scope="module")
def tokenizer(sp_model):
    """The tiny model loaded from tokenizer.json, the format the model ships."""
    proto = sp_pb2.ModelProto.FromString(sp_model)
    norm = proto.normalizer_spec
    spec = {
        "model_type": "bpe",
        "unk_id": 3,
        "byte_fallback": proto.trainer_spec.byte_fallback,
        "add_dummy_prefix": norm.add_dummy_prefix,
        "remove_extra_whitespaces": norm.remove_extra_whitespaces,
        "escape_whitespaces": norm.escape_whitespaces,
        "pieces": [
            [p.piece, p.score, sp_pb2.ModelProto.SentencePiece.Type.Name(p.type)]
            for p in proto.pieces
        ],
    }
    return Tokenizer(json.loads(json.dumps(spec)))


@pytest.fixture(scope="module")
def ll_tok(tokenizer):
    return ll_tokenizer(tokenizer, EOS, special_ids=[0, EOS, 2, TOOL_CALL, 5])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "weather in paris",
        "  extra   spaces ",
        "caf\u00e9 \U0001f335",
        "<tools>[1]",
        "x\u2581y",
    ],
)
def test_tokenizer_matches_sentencepiece(sp_model, tokenizer, text):
    reference = spm.SentencePieceProcessor(model_proto=sp_model)
    ids = reference.encode(text)
    assert tokenizer.encode(text) == ids
    assert tokenizer.decode(ids) == reference.decode(ids)


def test_tokenizer_loads_tokenizer_model(sp_model, tokenizer, tmp_path):
    (tmp_path / "tokenizer.model").write_bytes(sp_model)
    native = Tokenizer.from_file(tmp_path / "tokenizer.model")
    text = json.dumps(TOOLS)
    assert native.encode(text) == tokenizer.encode(text)
    assert native.sp_fragment.encode(" [") == tokenizer.sp_fragment.encode(" [")


def accepts(tokenizer, ll_tok, text, tools=TOOLS):
    matcher = tool_call_matcher(ll_tok, tools)
    return matcher.consume_tokens(tokenizer.encode(text)) and matcher.is_accepting()


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        json.dumps([{"name": "getTime", "arguments": {}}]),
        json.dumps([{"name": "getWeather", "arguments": {"city": "Paris", "days": 3}}]),
        json.dumps([{"name": "getWeather", "arguments": {"days": 3, "city": "Paris"}}]),
        json.dumps(
            [
                {"name": "getWeather", "arguments": {"city": 'a "b"'}},
                {"name": "getTime", "arguments": {}},
            ]
        ),
        '[{"name":"getWeather","arguments":{"city":["x",{"y":null}],"days":-1.5e3}}]',
        json.dumps([{"name": "getWeather", "arguments": {"city": "São Paulo 🌵"}}]),  # \u escapes
        '\n [ {"name" : "getTime" ,\n "arguments": { } } ]',
    ],
)
def test_grammar_accepts_valid_calls(tokenizer, ll_tok, text):
    assert accepts(tokenizer, ll_tok, text)


@pytest.mark.parametrize(
    "text",
    [
        '[{"name": "getNews", "arguments": {}}]',  # unknown tool
        '[{"name": "getWeather", "arguments": {"units": 1}}]',  # unknown key
        # repeated keys can't be ruled out by a JSON grammar, only bounded
        '[{"name": "getWeather", "arguments": {"city": "a", "city": "b", "city": "c"}}]',
        '[{"name": "getTime", "arguments": {"tool_id": 4}}]',  # tool takes no arguments
        '[{"name": "getWeather", "arguments": {"city": Paris}}]',  # malformed value
        '[{"name": "getWeather", "arguments": {"city": "Paris"}}',  # incomplete
    ],
)
def test_grammar_rejects_invalid_calls(tokenizer, ll_tok, text):
    assert not accepts(tokenizer, ll_tok, text)


def test_grammar_tool_definitions(tokenizer, ll_tok):
    flat = [
        {"name": "setTimer", "parameters": {"type": {}, "minutes": {}, "time-zone": {}}},
        {"description": "x"},
    ]
    text = '[{"name": "setTimer", "arguments": {"time-zone": "UTC", "type": 1}}]'
    assert accepts(tokenizer, ll_tok, text, flat)
    assert accepts(tokenizer, ll_tok, "[]", [])
    assert not accepts(tokenizer, ll_tok, '[{"name": "x", "arguments": {}}]', [])


def test_constrained_selector_follows_grammar(tokenizer, ll_tok):
    sel = ConstrainedSelector(tool_call_matcher(ll_tok, TOOLS), TOOL_CALL, EOS)
    n = len(tokenizer)
    assert sel.select(np.zeros(n)) == TOOL_CALL  # forced <tool_call>
    x = tokenizer.sp.piece_to_id("<0x78>")
    for token in tokenizer.encode("[]"):
        logits = np.zeros(n)
        logits[[EOS, x]] = 9.0  # the model's favourites, which the grammar doesn't allow
        logits[token] = 5.0
        assert sel.select(logits) == token
    assert sel.select(np.zeros(n)) == EOS  # complete -> EOS


@pytest.mark.parametrize(
    "text, names",
    [
        ('[{"name": "a", "arguments": {}}, {"name": "b", "argu', ["a"]),
        ('[{"name": "a", "argu', []),
        (
            ' [ {"name" : "a", "arguments": {"x": [1, 2]}} ,\n {"name": "b", "arguments": {}}, {',
            ["a", "b"],
        ),
        ('[{"name": "a", "arguments": {}}]', ["a"]),  # not cut off at all
        ("[", []),
        ("", []),
    ],
)
def test_complete_calls(text, names):
    assert [c["name"] for c in complete_calls(text)] == names


def test_resample_length():
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    assert resample(x, 44100, 16000).shape == (16000,)


def test_load_audio_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pcm = (rng.uniform(-0.5, 0.5, (800, 2)) * 32767).astype("<i2")
    path = tmp_path / "x.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(pcm.tobytes())
    audio, sr = load_audio(path)
    assert sr == 8000
    np.testing.assert_allclose(audio, pcm.astype(np.float32).mean(axis=1) / 32768, atol=1e-6)
    assert prepare_audio(path, None, 16000).shape == (1600,)


def test_prepare_audio_scales_integer_arrays():
    expected = [0.0, 0.5, -1.0]
    pcm16 = np.array([0, 16384, -32768], dtype=np.int16)
    np.testing.assert_allclose(prepare_audio(pcm16, None, 16000), expected)
    np.testing.assert_allclose(prepare_audio(pcm16.astype(np.int32) << 16, None, 16000), expected)
    np.testing.assert_allclose(
        prepare_audio(np.array([128, 192, 0], np.uint8), None, 16000), expected
    )
    floats = np.array(expected, dtype=np.float32)
    np.testing.assert_allclose(prepare_audio(floats, None, 16000), expected)  # unchanged


@pytest.mark.slow
def test_generate_from_hub():
    from neudecide import NeuDecide

    try:
        model = NeuDecide.from_pretrained(token=os.environ.get("HF_TOKEN"))
    except Exception as e:  # no network / no access
        pytest.skip(f"couldn't download the model: {e}")
    ids = model.tokenizer.encode(json.dumps(TOOLS))
    assert model.tokenizer.decode(ids) == json.dumps(TOOLS)
    calls = model.generate(np.zeros(16000, dtype=np.float32), TOOLS)
    assert isinstance(calls, list)
    assert all(c["name"] in {"getWeather", "getTime"} for c in calls)
