"""neo-agent command line interface."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from typing import Sequence


def _package_version() -> str:
    """Return the installed package version when available."""
    try:
        return version("neo-agent")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="neo-agent",
        description="neo-agent command line interface",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser(
        "ask",
        help="run a simple agent with one prompt",
    )
    ask_parser.add_argument("prompt", help="the user prompt to send to the agent")
    ask_parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="system prompt for the simple agent",
    )
    ask_parser.add_argument("--model", help="override the model name")
    ask_parser.add_argument(
        "--provider",
        default="auto",
        help="provider name, such as openai, deepseek, ollama, or auto",
    )
    ask_parser.add_argument("--base-url", help="override the provider base URL")
    ask_parser.add_argument("--api-key", help="override the API key")

    return parser


def run_ask(args: argparse.Namespace) -> int:
    """Execute a one-shot prompt with the simple agent."""
    from dotenv import load_dotenv

    load_dotenv()

    from neo_agent.agents.simple_agent import SimpleAgent
    from neo_agent.core.llm import NeoAgentLLM

    llm = NeoAgentLLM(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
    )
    agent = SimpleAgent(
        name="neo-agent-cli",
        llm=llm,
        system_prompt=args.system_prompt,
    )
    print(agent.run(args.prompt))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        return run_ask(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
