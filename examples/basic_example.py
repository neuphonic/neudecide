import json
from pathlib import Path

from neudecide import NeuDecide

EXAMPLES = Path(__file__).parent


def main(audio, tools_path, constrained=True, repo="neuphonic/neudecide"):
    with open(tools_path, "r") as f:
        tools = json.load(f)

    model = NeuDecide.from_pretrained(repo)
    calls = model.generate(audio, tools, constrained=constrained)
    print(json.dumps(calls, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuDecide Example")
    parser.add_argument(
        "--audio", type=str, default=str(EXAMPLES / "command.wav"), help="Path to a WAV file"
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=str(EXAMPLES / "tools.json"),
        help="JSON file holding a list of tool definitions",
    )
    parser.add_argument(
        "--no-constrain", action="store_true", help="Don't restrict output to valid tool calls"
    )
    parser.add_argument(
        "--repo", type=str, default="neuphonic/neudecide", help="Huggingface repo with the model"
    )
    args = parser.parse_args()
    main(args.audio, args.tools, constrained=not args.no_constrain, repo=args.repo)
