"""CLI entry point for Talk to GRN.

Usage:
  # Router mode (auto-selects environment):
  uv run human2system.py --system lorenzsystem_gradient --prompt "Keep stable"

  # Debug mode (explicit environment):
  uv run human2system.py --system lorenzsystem_gradient --env_name CartPole-v1 --prompt "Keep stable"

  # Verbose to see all intermediate steps:
  uv run human2system.py --system lorenzsystem_gradient --prompt "Keep stable" -v
"""

from __future__ import annotations

import argparse
import os

import dotenv
from openai import OpenAI

from talk import LanguageGame
from talk.policy import _DEFAULT_AGENTS_ROOT


def main():
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    parser = argparse.ArgumentParser(
        description="Talk to a trained RL agent via LLM translation."
    )
    parser.add_argument("--system", type=str, required=True,
                        help="System folder name, e.g. lorenzsystem_gradient")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="The message to send to the agent")
    parser.add_argument("--env_name", "-e", type=str, default=None,
                        help="Environment name for debug (omit for Router mode)")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--agents_root", type=str, default=_DEFAULT_AGENTS_ROOT)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model for LLM translation steps")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for report.md and rendered frames")
    args = parser.parse_args()

    game = LanguageGame(
        client,
        args.system,
        env_name=args.env_name,
        verbose=args.verbose,
        seed=args.seed,
        agents_root=args.agents_root,
        model=args.model,
        device=args.device,
    )

    result = game.chat(args.prompt, output_dir=args.output_dir)
    print(result.reply)
    if args.output_dir:
        print(f"\nOutput saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
