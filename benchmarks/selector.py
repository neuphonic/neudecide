"""Reproducible constrained-selector benchmark.

Examples:
    uv run python benchmarks/selector.py --mode both --cores 0
    uv run python benchmarks/selector.py --mode rust --cores 2-3
"""

import argparse
import io
import json
import os
import statistics
import time


def parse_cores(value):
    cores = set()
    for part in value.split(","):
        bounds = part.split("-", 1)
        if len(bounds) == 1:
            cores.add(int(bounds[0]))
        else:
            start, end = map(int, bounds)
            if end < start:
                raise argparse.ArgumentTypeError("CPU range must be ascending")
            cores.update(range(start, end + 1))
    if not cores or min(cores) < 0:
        raise argparse.ArgumentTypeError("CPU set must contain non-negative IDs")
    return sorted(cores)


def pin_process(cores):
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("this benchmark requires Linux CPU affinity support")
    available = os.sched_getaffinity(0)
    unavailable = set(cores) - available
    if unavailable:
        raise RuntimeError(
            f"CPU(s) {sorted(unavailable)} are unavailable; allowed CPUs are {sorted(available)}"
        )
    os.sched_setaffinity(0, cores)


def make_fixture():
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    from neudecide.tokenizer import Tokenizer

    tools = [
        {
            "name": "getWeather",
            "parameters": {
                "type": "object",
                "properties": {"city": {}, "days": {}},
            },
        },
        {"name": "getTime", "parameters": {"type": "object", "properties": {}}},
    ]
    corpus = [json.dumps(tools)] * 20
    output = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_writer=output,
        model_type="bpe",
        vocab_size=400,
        hard_vocab_limit=False,
        byte_fallback=True,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        user_defined_symbols=["<tool_call>", "<tools>"],
        normalization_rule_name="identity",
        minloglevel=2,
    )
    proto = sp_pb2.ModelProto.FromString(output.getvalue())
    normalizer = proto.normalizer_spec
    spec = {
        "model_type": "bpe",
        "unk_id": 3,
        "byte_fallback": True,
        "add_dummy_prefix": normalizer.add_dummy_prefix,
        "remove_extra_whitespaces": normalizer.remove_extra_whitespaces,
        "escape_whitespaces": normalizer.escape_whitespaces,
        "pieces": [
            [p.piece, p.score, sp_pb2.ModelProto.SentencePiece.Type.Name(p.type)]
            for p in proto.pieces
        ],
    }
    return tools, Tokenizer(spec)


def run_batch(grammar, tokenizer, tools, repetitions):
    import numpy as np

    text = '[{"name":"getWeather","arguments":{"city":"Paris","days":3}}]'
    target = tokenizer.encode(text)
    logits = np.zeros(len(tokenizer), dtype=np.float32)
    elapsed = time.perf_counter_ns()
    for _ in range(repetitions):
        selector = grammar.ConstrainedSelector(
            grammar.tool_call_matcher(grammar.ll_tokenizer(tokenizer, 1, [0, 1, 2, 4, 5]), tools),
            4,
            1,
        )
        selector.select(logits)
        for token in target:
            logits.fill(0)
            logits[token] = 10
            logits[1] = 9
            selector.select(logits)
    return (time.perf_counter_ns() - elapsed) / 1e9


def measure(mode, tools, tokenizer, warmups, repetitions, samples):
    import neudecide.grammar as grammar

    if mode == "rust":
        if grammar._rust_masked_argmax is None:
            raise RuntimeError("Rust extension is unavailable; install the package with `uv pip install -e .`")
    else:
        grammar._rust_masked_argmax = None

    for _ in range(warmups):
        run_batch(grammar, tokenizer, tools, repetitions)
    timings = [run_batch(grammar, tokenizer, tools, repetitions) for _ in range(samples)]
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rust", "numpy", "both"), default="both")
    parser.add_argument("--cores", type=parse_cores, default=[0], help="CPU IDs, e.g. 0 or 2-3")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if min(args.warmups, args.samples, args.repetitions) < 1:
        parser.error("warmups, samples, and repetitions must be positive")

    pin_process(args.cores)
    tools, tokenizer = make_fixture()
    modes = ("rust", "numpy") if args.mode == "both" else (args.mode,)
    print(f"affinity={sorted(os.sched_getaffinity(0))} modes={','.join(modes)}")
    for mode in modes:
        timings = measure(mode, tools, tokenizer, args.warmups, args.repetitions, args.samples)
        median = statistics.median(timings)
        spread = statistics.pstdev(timings)
        print(
            f"{mode}: median={median:.6f}s stdev={spread:.6f}s "
            f"per_batch={median / args.repetitions * 1000:.3f}ms "
            f"samples={args.samples} repetitions={args.repetitions}"
        )


if __name__ == "__main__":
    main()