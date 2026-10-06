import json
import os
import wave

import numpy as np
import pytest

from neudecide.audio import load_audio, prepare_audio, resample
from neudecide.grammar import ConstrainedSelector, ToolCallGrammar

TOOLS = [
    {
        "name": "cleanRoom",
        "description": "Vacuum-cleans the specified room.",
        "parameters": {
            "type": "object",
            "properties": {"room": {"type": "string"}, "power": {"type": "integer"}},
            "required": ["room"],
        },
    },
    {
        "name": "returnToBase",
        "description": "Docks the robot.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def accepts(text, tools=TOOLS):
    return ToolCallGrammar(tools).accepts(text)


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        json.dumps([{"name": "returnToBase", "arguments": {}}]),
        json.dumps([{"name": "cleanRoom", "arguments": {"room": "kitchen", "power": 3}}]),
        json.dumps(
            [
                {"name": "cleanRoom", "arguments": {"room": 'a "b"'}},
                {"name": "returnToBase", "arguments": {}},
            ]
        ),
        '[{"name":"cleanRoom","arguments":{"room":["x",{"y":null}],"power":-1.5e3}}]',
    ],
)
def test_grammar_accepts_valid_calls(text):
    assert accepts(text)


@pytest.mark.parametrize(
    "text",
    [
        '[{"name": "mopRoom", "arguments": {}}]',  # unknown tool
        '[{"name": "cleanRoom", "arguments": {"floor": 1}}]',  # unknown key
        '[{"name": "cleanRoom", "arguments": {"room": "a", "room": "b"}}]',  # repeated key
        '[{"name": "returnToBase", "arguments": {"tool_id": 4}}]',  # tool takes no arguments
        '[{"name": "cleanRoom", "arguments": {"room": kitchen}}]',  # malformed value
        '[{"name": "cleanRoom", "arguments": {"room": "kitchen"}}',  # incomplete
    ],
)
def test_grammar_rejects_invalid_calls(text):
    assert not accepts(text)


def test_constrained_selector_follows_grammar():
    strings = ["", "", "[", "]", "x", "[]"]  # ids 0/1 have no text, like control tokens
    sel = ConstrainedSelector(ToolCallGrammar(TOOLS), strings, tool_call_id=0, eos_id=1)
    assert sel.select(np.zeros(6)) == 0  # forced <tool_call>
    assert sel.select(np.array([9, 9, 0, 0, 5, 1.0])) == 5  # best token the grammar allows
    assert sel.select(np.zeros(6)) == 1  # complete -> EOS


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
    assert all(c["name"] in {"cleanRoom", "returnToBase"} for c in calls)
