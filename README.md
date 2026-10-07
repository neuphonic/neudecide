# NeuDecide

**A tiny voice-action model. Audio and a list of tools in, a tool call out. No transcript in between.**

[Hugging Face](https://huggingface.co/neuphonic/neudecide) · [PyPI](https://pypi.org/project/neudecide/) · [arxiv (coming soon}]()

NeuDecide converts speech directly into function calls. It's 43 MB, runs on a single CPU thread via ONNX Runtime, and needs no GPU or internet connection once downloaded. Tools are passed in as JSON schemas at inference time, so you change what the model can do by changing the JSON, not by retraining.

```
"clean the kitchen"  +  [cleanRoom, goToBase, getBattery]   →   [{"name": "cleanRoom", "arguments": {"room": "kitchen"}}]
```

## Installation

```bash
pip install neudecide
```

Requires Python 3.10 – 3.13. The model weights are downloaded from the Hugging Face Hub on first use and cached after that.

## Quickstart

This example turns a recorded voice command into a call for a robot vacuum. Save a WAV recording of yourself saying something like "clean the kitchen" as `command.wav`, then run:

```python
from neudecide import NeuDecide

TOOLS = [
    {
        "name": "cleanRoom",
        "description": "Vacuum-cleans one room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "enum": [
                        "living room",
                        "bathroom",
                        "kitchen",
                        "bedroom",
                    ],
                },
            },
            "required": ["room"],
        },
    },
    {
        "name": "goToBase",
        "description": "Sends the vacuum back to its charging base.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "getBattery",
        "description": "Reports the vacuum's battery level.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def main():
    model = NeuDecide.from_pretrained()
    calls = model.generate("command.wav", TOOLS)
    print(calls)


if __name__ == "__main__":
    main()
```

Output:

```
[{'name': 'cleanRoom', 'arguments': {'room': 'kitchen'}}]
```

`generate` returns a list of `{"name", "arguments"}` dicts, or `[]` when no tool applies.

This script is also in the repo as `examples/basic_example.py`, so from a clone you can run `python examples/basic_example.py`.

## Usage

### Audio input

`audio` can be a path to a PCM WAV file or a NumPy array. Arrays may be mono `(samples,)` or multi-channel `(samples, channels)`; channels are averaged to mono. Audio is resampled to 16 kHz automatically, so pass `sample_rate=` if your array is at a different rate:

```python
calls = model.generate("command.wav", TOOLS)                     # WAV file
calls = model.generate(waveform, TOOLS, sample_rate=44100)       # NumPy array
```

Only WAV is read natively. For other formats (MP3, FLAC, OGG), load the audio yourself and pass the array. Audio longer than 30 seconds is truncated.

### Tool definitions

Each tool has a `name`, a `description`, and `parameters` as a JSON Schema object. You can pass the list directly or as a JSON string. The model only knows what you tell it about each tool, so clear names and descriptions do most of the work. When an argument has a fixed set of values, like the rooms above, list them with `enum`.

## How it works

NeuDecide is a pre-trained streaming English speech encoder joined to a compact tool-calling encoder–decoder, initialised from [Needle](https://github.com/cactus-compute/needle) by Cactus Compute, and trained end to end as one system. Inference runs three ONNX graphs:

| Graph | Runs | Does | Parameters | Size |
|---|---|---|---|---|
| `audio_encoder` | Once per utterance | Waveform → 12.5 Hz latent frames | 28.7M | 22.5 MB |
| `tool_encoder` | Once per utterance and tool list | Audio frames + tool tokens → cross-attention keys and values | 11.8M | 10.1 MB |
| `decoder_step` | Once per output token | Next token, reusing a cached self-attention state | 15.0M | 10.0 MB |

**TBC** the `tool_encoder` KV-cache state can be retained between tool calls, if the tools remain the same.

| | |
|---|---|
| Model width | 512 |
| Decoder layers | 8 |
| Attention | Grouped-query, 4 key/value heads of 64 dimensions |
| Vocabulary | 8,192 SentencePiece BPE tokens |
| Sample rate | 16 kHz |
| Limits | 30 s of audio, 1,536 tool tokens, 128 output tokens |

Tools are serialised exactly as Python's `json.dumps` writes them by default, matching how the model was trained.

## Results

<p align="center">
  <img src="assets/param_bubbles.png" alt="Exact match against tool accuracy on FSC, SNIPS SLU and SLURP for NeuDecide, Voxtral Mini 3B and ASR + decision-model cascades, with bubble area showing model size" width="720">
</p>

Exact match against tool accuracy on FSC, SNIPS SLU and SLURP, choosing from 10 tools. Each bubble is one system, and its area is the model's size on disk; a cascade counts its ASR and decision model together. ([PDF](assets/param_bubbles.pdf))

We compared NeuDecide with five speech-recognition cascades (Parakeet 110M and 660M feeding Needle or FunctionGemma, and Whistle feeding Needle 3) and with Voxtral Mini 3B, an end-to-end speech model. Each system chooses from 10 tools and must pick the right tool and fill its arguments. "Best cascade" is the highest score any of the five cascades reached on that metric.

| Dataset | Metric | NeuDecide | Best cascade | Voxtral Mini 3B |
|---|---|---|---|---|
| SLURP | Tool accuracy | **77.3%** | 39.9% | 54.4% |
| SLURP | Exact match | **24.7%** | 3.9% | 7.6% |
| Fluent Speech Commands | Tool accuracy | **97.0%** | 90.1% | 96.4% |
| Fluent Speech Commands | Exact match | 82.8% | 65.4% | **95.9%** |
| SNIPS SLU | Tool accuracy | 70.1% | 78.0% | **87.9%** |
| SNIPS SLU | Exact match | 20.9% | 25.5% | **36.8%** |

On SLURP, the most varied of the three datasets, NeuDecide beats every other system on every metric. With 10 tools, it gets six times as many commands exactly right as the best cascade and three times as many as Voxtral Mini 3B. On Fluent Speech Commands, it picks the right tool more often than any other system, and it beats every cascade on exact match. Voxtral Mini 3B fills arguments more accurately there, at roughly 50 times NeuDecide's parameter count. SNIPS, with its free-form names, is the weak spot (see [Limitations](#limitations)).

## Performance

Measured with the `q4` export on a single CPU thread:

| Device | Time to call | Load time |
|---|---|---|
| MacBook Pro M3 | 46 ms | 159 ms |
| Samsung S24+ | 82 ms | 267 ms |
| Raspberry Pi 5 | 206 ms | 499 ms |

Peak RAM ranged from 146 MB to 174 MB across these devices; it depends on the runtime and workload.

## Limitations

- **It can fire on noise.** Given 30 s of low-level noise, it still produced a call. There is no calibrated confidence score yet, so put a gate in front: voice-activity detection, or an explicit "no action" tool.
- **Keep tool lists short.** We recommend at most 10 tools. Longer lists also cost more compute than longer audio, and lists over 1,536 tokens are truncated with a warning.
- **English only.** Inputs are capped at 30 s of audio and 128 output tokens.
- **Free-form arguments are harder.** On SNIPS, where arguments are often open-ended names such as artists and playlists, transcript-based cascades still fill arguments more accurately. Use `enum` wherever an argument has a fixed set of values.

Test it on your own tools and your users' voices before deploying. NeuDecide works best next to a larger model: let it handle the fast, frequent, bounded decisions, and route anything it can't place to a bigger model or a person.

## Development

```bash
pip install -e . -r requirements-dev.txt
pre-commit install
pytest -m "not slow"   # "slow" tests download the model from the Hub
```

## Acknowledgements

The tool-calling encoder–decoder and tokenizer are built on [Needle](https://github.com/cactus-compute/needle) by [Cactus Compute](https://cactuscompute.com/).
The frozen ASR model is dervied from [Nvidia FastConformer 32M](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms).

## License

Apache 2.0. See [LICENSE](LICENSE).