"""A minimal agent built on the AWS Strands Agents SDK.

The agent loop, in one sentence: the model is given a list of tools, it decides
which to call and with what arguments, the SDK runs them, feeds the results
back, and repeats until the model has an answer.

You write the tools. Strands writes the loop. There is no if/else in this file
deciding which tool to use — that decision is the model's, every turn.

Requires:
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python agent.py                              # interactive chat
    python agent.py --demo                       # scripted demo questions
    python agent.py --ask "What is 17 * 23?"     # one-shot question
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands_tools import current_time

from tools_api import (
    get_cat_fact,
    get_exchange_rate,
    get_public_holidays,
    get_weather,
    web_search,
)
from tools_local import calculate, roll_dice, tell_joke
from tools_rag import search_documents

# Reads ANTHROPIC_API_KEY from Agents/.env — no need to export it.
load_dotenv(Path(__file__).parent / ".env")

MODEL_ID = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Use your tools rather than answering from memory whenever a tool applies —
especially for arithmetic, current time, weather, and exchange rates, where
your own guess would be stale or wrong.

For anything recent, changeable, or that you are not confident about, use
web_search before answering. When you do search, say what you found and cite
the source URL.

For questions about Chiranjeevi or Nandamuri Balakrishna, use search_documents
— we hold private biographies of both. Cite the file and page you used. If the
passages come back without the answer, say so and search again with different
wording rather than filling the gap from memory.

If a tool returns an error, tell the user plainly what failed. Do not invent
the answer it would have given.

Keep responses short and conversational."""

# The full toolbox: our own functions plus one that ships with Strands.
# To the model these are indistinguishable.
TOOLS = [
    calculate,            # local
    tell_joke,            # local
    roll_dice,            # local
    # web_search,           # DuckDuckGo
    search_documents,     # our own RAG index (RAG/data/indexes/)
    get_weather,          # Open-Meteo
    get_exchange_rate,    # open.er-api.com
    get_public_holidays,  # date.nager.at
    get_cat_fact,         # catfact.ninja
    current_time,         # built into strands_tools
]

DEMO_QUESTIONS = [
    "What time is it right now?",
    "What is (1234 * 17) / 3?",
    "Search the web: what is the AWS Strands Agents SDK?",
    "What's the weather in Hyderabad, and how does it compare to London?",
    "How much is 500 USD in Indian rupees?",
    "Roll 3 twenty-sided dice, then tell me a joke.",
    "What are the public holidays in Germany this year?",
    "Give me a cat fact, and tell me what 2 to the power of 40 is.",
    "What awards has Chiranjeevi received, and how old is he today?",
]


def build_agent(callback_handler: object = "default") -> Agent:
    """Wire up the model and the tools. This is the entire setup.

    callback_handler controls where tool activity is printed. The default
    prints to the terminal; pass None to silence it, which is what the
    Streamlit UI does before streaming events itself. It has to be set here at
    construction — assigning to agent.callback_handler afterwards breaks.
    """
    model = AnthropicModel(
        client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
        model_id=MODEL_ID,
        max_tokens=4096,
    )
    kwargs = {} if callback_handler == "default" else {"callback_handler": callback_handler}
    return Agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, **kwargs)


def run_demo(agent: Agent) -> None:
    for question in DEMO_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"USER: {question}")
        print("=" * 70)
        # Calling the agent prints the response (and tool activity) as it goes.
        agent(question)
        print()


def run_interactive(agent: Agent) -> None:
    print("Agent ready. Type a question, or 'exit' to quit.")
    print("Try: weather in Tokyo / 45 squared / 100 EUR in USD / tell me a joke\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        print()
        agent(question)
        print()


def main():
    parser = argparse.ArgumentParser(description="Strands agent demo.")
    parser.add_argument("--demo", action="store_true", help="run scripted questions")
    parser.add_argument("--ask", help="ask a single question and exit")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY not found. Add it to Agents/.env as:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    agent = build_agent()
    print(f"Model: {MODEL_ID}")
    print(f"Tools: {', '.join(t.tool_name for t in agent.tool_registry.registry.values())}\n")

    if args.ask:
        agent(args.ask)
        print()
    elif args.demo:
        run_demo(agent)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
