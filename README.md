# NeuDecide

**A tiny voice-action model. Audio and a list of tools in, a tool call out. No transcript in between.**

[Hugging Face](https://huggingface.co/neuphonic/neudecide) · [PyPI](https://pypi.org/project/neudecide/) · [arxiv (coming soon)]()

NeuDecide converts speech directly into function calls. It's 43 MB, can run on a single thread via ONNX Runtime, and needs no GPU or internet connection once downloaded. Tools are passed in as JSON schemas at inference time, so you change what the model can do by changing the JSON, not by retraining.

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
    calls = model.generate("./examples/command.wav", TOOLS)
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

NeuDecide is a pre-trained streaming English speech encoder joined to a compact tool-calling encoder–decoder, initialised from [Needle](https://github.com/cactus-compute/needle), and trained end to end as one system. Inference runs three ONNX graphs:

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
  <img src="assets/tool_acc_params.png" alt="Tool accuracy vs Model Paramaters on the SLURP dataset." width="720">
</p>

We compared NeuDecide with five speech-recognition cascades (Parakeet 110M and 660M feeding Needle or FunctionGemma, and Whistle feeding Needle 3) and with Voxtral Mini 3B, an end-to-end speech model, on [SLURP](https://aclanthology.org/2020.emnlp-main.588/) (Bastianelli et al., 2020), a dataset of spoken commands to a home assistant across 18 domains. Each system chooses from 10 tools. Some requests need only the right tool; the rest also need its arguments filled, and exact match counts a call as right only if the tool and every argument match. The results cover both kinds of request together, weighted by how many of each SLURP has, as in the plot. The table shows two of the cascades alongside Voxtral Mini 3B; a cascade's size counts its ASR and decision model together.

| | | NeuDecide | Whistle → Needle 3 | Parakeet 660M → FunctionGemma | Voxtral Mini 3B |
|---|---|---|---|---|---|
| **Size** | Parameters | 55.5M | 176M | 870M | 3B |
| **Size** | File size | 43 MB | 52 MB | 3.0 GB | 18.7 GB |
| **Size** | File size vs NeuDecide | 1× | 1.2× | 71× | 435× |
| SLURP | Tool accuracy | **75.5%** | 25.6% | 25.2% | 48.7% |
| SLURP | Exact match | **37.5%** | 7.1% | 1.9% | 6.9% |

On SLURP, NeuDecide beats every other system on both metrics. With 10 tools, it gets more than five times as many commands exactly right as the best cascade or Voxtral Mini 3B.

## Performance

Measured with the `q4` export:

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

- https://github.com/cactus-compute/needle
- https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms

## License

Apache 2.0. See [LICENSE](LICENSE).
