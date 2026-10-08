import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio import prepare_audio
from .grammar import ConstrainedSelector, GreedySelector, ll_tokenizer, tool_call_matcher
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_REPO = "neuphonic/neudecide"
DEFAULT_VARIANT = "q4"  # int4 MatMulNBits graphs, the ones the native runtime ships
TOKENIZER_FILES = ("tokenizer.model", "tokenizer.json")  # first one present wins


class NeuDecide:
    """Speech + a list of tools -> tool calls, with onnxruntime.

    Three ONNX graphs run per request:
      audio_encoder   once per utterance:            waveform -> audio frames
      tool_encoder    once per (utterance, tools):   -> cross-attention keys / values
      decoder_step    once per output token, with a growing self-attention KV cache

        model = NeuDecide.from_pretrained()
        calls = model.generate("command.wav", tools)
        # [{"name": "set_timer", "arguments": {"minutes": 5}}]
    """

    def __init__(self, model_dir, variant=DEFAULT_VARIANT, threads=None, providers=None):
        """model_dir: a directory with config.json, tokenizer.model (or tokenizer.json)
        and the ONNX graphs.
        variant: which set of graphs from config.json's "variants".
        threads: onnxruntime intra-op threads (None = onnxruntime's default)."""
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        tokenizer = next((model_dir / f for f in TOKENIZER_FILES if (model_dir / f).exists()), None)
        if tokenizer is None:
            raise FileNotFoundError(f"{model_dir} has no {' or '.join(TOKENIZER_FILES)}")
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.special = self.config["special_tokens"]
        self.sample_rate = self.config["sample_rate"]
        self.variant = variant
        graphs = self.graph_files(self.config, variant)

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = list(providers or ["CPUExecutionProvider"])
        self.audio_encoder, self.tool_encoder, self.decoder_step = (
            ort.InferenceSession(
                str(model_dir / graphs[k]), sess_options=options, providers=providers
            )
            for k in ("audio_encoder", "tool_encoder", "decoder_step")
        )
        self._ll_tokenizer = None

    @staticmethod
    def graph_files(config, variant):
        variants = config.get("variants") or {"fp32": {"graphs": config["graphs"]}}
        if variant not in variants:
            raise ValueError(f"model has no {variant!r} variant (has: {sorted(variants)})")
        return variants[variant]["graphs"]

    @classmethod
    def from_pretrained(
        cls, repo_id=DEFAULT_REPO, variant=DEFAULT_VARIANT, revision=None, token=None, **kwargs
    ):
        """Downloads (or reuses the cached) model from the Hugging Face Hub."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        def fetch(name):
            return hf_hub_download(repo_id, name, revision=revision, token=token)

        config_path = Path(fetch("config.json"))
        config = json.loads(config_path.read_text())
        for name in TOKENIZER_FILES:
            try:
                fetch(name)
                break
            except EntryNotFoundError:
                continue
        for name in cls.graph_files(config, variant).values():
            fetch(name)
        return cls(config_path.parent, variant=variant, **kwargs)

    # -- stages ---------------------------------------------------------------

    def encode_tools(self, tools):
        """[<tools>] + the tokenized tool list, serialized with json.dumps defaults
        (as in training) and truncated to max_tools_len. A JSON string is parsed
        and re-serialized, so its formatting doesn't matter."""
        if isinstance(tools, str):
            tools = json.loads(tools)
        text = json.dumps(tools)
        ids = [self.special["tools"]] + self.tokenizer.encode(text)
        limit = self.config["max_tools_len"]
        if len(ids) > limit:
            logger.warning("tool list is %d tokens; truncated to %d", len(ids), limit)
            ids = ids[:limit]
        return np.asarray([ids], dtype=np.int64)

    def encode_audio(self, waveform):
        """16 kHz mono waveform -> (audio_embeds, audio_len)."""
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = waveform[: int(self.config["max_audio_seconds"] * self.sample_rate)]
        return self.audio_encoder.run(
            None,
            {
                "waveform": waveform[None, :],
                "waveform_len": np.asarray([waveform.shape[0]], dtype=np.int64),
            },
        )

    def selector(self, tools, constrained=True):
        """The token-selection policy for one generation."""
        if not constrained:
            return GreedySelector()
        if self._ll_tokenizer is None:
            self._ll_tokenizer = ll_tokenizer(
                self.tokenizer, self.special["eos"], self.special.values()
            )
        matcher = tool_call_matcher(self._ll_tokenizer, tools)
        return ConstrainedSelector(matcher, self.special["tool_call"], self.special["eos"])

    def generate_ids(
        self, audio, tools, sample_rate=None, constrained=True, max_new_tokens=None, selector=None
    ):
        """Output token ids (EOS excluded). `selector` overrides the decoding
        policy: any object with `select(logits) -> token id`."""
        waveform = prepare_audio(audio, sample_rate, self.sample_rate)
        audio_embeds, audio_len = self.encode_audio(waveform)
        cross_key, cross_value, encoder_mask = self.tool_encoder.run(
            None,
            {
                "audio_embeds": audio_embeds,
                "audio_len": audio_len,
                "tools_ids": self.encode_tools(tools),
            },
        )

        selector = selector or self.selector(tools, constrained)
        c = self.config
        past_key = past_value = np.zeros(
            (c["num_decoder_layers"], 1, c["num_kv_heads"], 0, c["head_dim"]), dtype=np.float32
        )
        token, out = self.special["eos"], []  # the decoder starts from EOS
        for _ in range(max_new_tokens or c["max_answer_len"]):
            logits, past_key, past_value = self.decoder_step.run(
                None,
                {
                    "input_ids": np.asarray([[token]], dtype=np.int64),
                    "past_key": past_key,
                    "past_value": past_value,
                    "cross_key": cross_key,
                    "cross_value": cross_value,
                    "encoder_mask": encoder_mask,
                },
            )
            token = selector.select(logits[0])
            if token == self.special["eos"]:
                break
            out.append(token)
        return out

    def decode(self, ids):
        """Token ids -> answer text, without the structural tokens."""
        ids = list(ids)
        if ids and ids[0] == self.special["tool_call"]:
            ids = ids[1:]
        structural = {self.special[k] for k in ("pad", "eos", "bos", "tool_call", "tools")}
        return self.tokenizer.decode([i for i in ids if i not in structural])

    # -- high level -----------------------------------------------------------

    def generate_text(self, audio, tools, **kwargs):
        """The raw answer text: a JSON list of calls."""
        return self.decode(self.generate_ids(audio, tools, **kwargs))

    def generate(self, audio, tools, max_new_tokens=None, **kwargs):
        """Tool calls for an utterance: a list of {"name", "arguments"} dicts,
        [] when no tool applies.

        audio: a path to a WAV file, or a numpy array (samples,) or
            (samples, channels) -- pass sample_rate= if it isn't 16 kHz.
            Integer arrays (e.g. int16) are scaled to [-1, 1]; float arrays
            should be in [-1, 1] already.
        tools: a list of tool definitions (name, description, JSON-Schema
            parameters), or the same as a JSON string.
        constrained: restrict output to valid calls of the given tools (default True).
        max_new_tokens: output length limit (default: the model's max_answer_len).
            If the answer reaches it, the calls completed before it are
            returned (logged).
        """
        limit = max_new_tokens or self.config["max_answer_len"]
        ids = self.generate_ids(audio, tools, max_new_tokens=limit, **kwargs)
        if len(ids) < limit:  # ended with EOS
            return parse_calls(self.decode(ids))
        calls = complete_calls(self.decode(ids))
        logger.warning(
            "answer reached the %d-token limit; returning the %d call(s) completed before it",
            limit,
            len(calls),
        )
        return calls


def parse_calls(text):
    """A JSON list of objects with a "name", else [] (logged)."""
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        calls = None
    if isinstance(calls, list) and all(isinstance(c, dict) and "name" in c for c in calls):
        return calls
    logger.warning("model output isn't a list of tool calls: %r", text)
    return []


def complete_calls(text):
    """The calls written out in full at the start of a JSON list that was cut off
    part-way, e.g. [{...}, {"na -> [{...}]."""
    decoder, calls = json.JSONDecoder(), []
    text = text.lstrip()
    if not text.startswith("["):
        return calls
    i = 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        try:
            call, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return calls
        if not (isinstance(call, dict) and "name" in call):
            return calls
        calls.append(call)
