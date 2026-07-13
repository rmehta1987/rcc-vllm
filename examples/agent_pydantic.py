#!/usr/bin/env python3
"""Build your own agent on an ai-session model, using PydanticAI.

This is a self-contained, runnable example of a *custom* agent whose reasoning
engine is a model you serve yourself on an RCC GPU node through the ai-session
gateway. It talks to the gateway's OpenAI-compatible endpoint, so from the
agent's point of view the local Qwen model looks like any OpenAI chat model.

The agent defines one trivial, read-only local tool (`count_words`) and answers
a single question that requires calling it, which demonstrates end-to-end tool
calling against the served model.

--------------------------------------------------------------------------------
Prerequisites
--------------------------------------------------------------------------------
1. A running ai-session with tool calling enabled. Start one on a login node
   (in tmux or screen), serving the 72B rather than the coder default:

       module load ai-session
       ai-session code --agent --model qwen2.5_72B

   `--agent` starts the server with tool calling enabled, which is what lets the
   model emit the structured tool calls PydanticAI relies on.

2. This script needs the `pydantic-ai` package, which is almost certainly NOT
   installed in the shared environments and MUST be installed by you, into your
   OWN virtual environment, on a LOGIN node (compute nodes have no internet):

       python -m venv ~/agent-venv
       source ~/agent-venv/bin/activate
       pip install pydantic-ai

   If the package is missing this script prints these instructions and exits 0
   without installing anything.

--------------------------------------------------------------------------------
Configuration (environment variables) -- all set by `eval "$(ai-session env)"`
--------------------------------------------------------------------------------
  AISESSION_BASE_URL   gateway base URL, e.g. http://localhost:8421/v1
                       (your own GW_PORT is 8400 + UID % 90)
  AISESSION_API_KEY    the session access key (required)
  MODEL                served model key, default qwen2.5_72B

--------------------------------------------------------------------------------
Run
--------------------------------------------------------------------------------
       source ~/agent-venv/bin/activate
       eval "$(ai-session env)"   # sets AISESSION_BASE_URL / AISESSION_API_KEY
       python /project/rcc/mehta5/vllm/examples/agent_pydantic.py

Model choice: the default is qwen2.5_72B, NOT the coding model. vLLM's `hermes`
tool-call parser does not populate `tool_calls` for Qwen2.5-Coder-32B (vLLM
issue #29192), so agents that rely on native tool calling fail silently against
the coder model. qwen2.5_72B and qwen3_4b emit tool calls the parser matches,
so pick one of those for any tool-calling agent. See docs/coding/agents.md.
"""

import os
import sys

# Where the ai-session gateway listens, and the per-session key it requires.
# The gateway binds to 127.0.0.1 on the login node; from a laptop, open the SSH
# tunnel the start command prints so localhost:<GW_PORT> reaches it.
BASE_URL = os.environ.get("AISESSION_BASE_URL", "http://localhost:8421/v1")
API_KEY = os.environ.get("AISESSION_API_KEY", "")

# Default to the general 72B model, NOT qwen2.5_coder_32B: vLLM's hermes parser
# fails to populate tool_calls for the coder checkpoint (vLLM #29192), so native
# tool calling silently returns no calls there. qwen2.5_72B and qwen3_4b work.
MODEL = os.environ.get("MODEL", "qwen2.5_72B")


SETUP_INSTRUCTIONS = """\
pydantic-ai is not installed in this Python environment, so this example cannot
run yet. It is not installed for you: create your own virtual environment and
install it, on a LOGIN node (compute nodes have no internet access).

  1. Create and activate your own virtual environment (login node):

       python -m venv ~/agent-venv
       source ~/agent-venv/bin/activate

  2. Install pydantic-ai into it:

       pip install pydantic-ai

  3. Start a tool-calling session (login node, in tmux or screen), serving the
     72B rather than the coder default:

       module load ai-session
       ai-session code --agent --model qwen2.5_72B

  4. Load the session URL and key into your shell and run this script:

       eval "$(ai-session env)"
       python /project/rcc/mehta5/vllm/examples/agent_pydantic.py

Nothing has been installed or changed. Exiting.
"""


def main() -> int:
    # Import pydantic-ai lazily so the script always compiles and, when the
    # package is absent, prints setup instructions and exits 0 (installing
    # nothing) rather than crashing with a traceback.
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError:
        print(SETUP_INSTRUCTIONS)
        return 0

    if not API_KEY:
        print(
            'AISESSION_API_KEY is not set. Run `eval "$(ai-session env)"` to '
            "load the session's URL and key (the key is also saved at "
            "<state-dir>/logs/gateway/session_key), then re-run. Exiting."
        )
        return 0

    # Wrap the gateway's OpenAI-compatible endpoint as a PydanticAI model. The
    # provider carries the base URL and the session key; the model name must be
    # the key the session serves under (MODEL above).
    model = OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY),
    )

    agent = Agent(
        model,
        system_prompt=(
            "You are a concise assistant. When a question is about the number "
            "of words in some text, call the count_words tool and report its "
            "result. Do not guess the count yourself."
        ),
    )

    # One trivial, read-only local tool. `@agent.tool_plain` registers a plain
    # function (no agent context) that the model may call; PydanticAI advertises
    # its name, docstring, and typed signature to the model as a tool schema.
    # This stands in for any local capability you want to expose -- keep such
    # tools read-only and side-effect-free unless you have reviewed the risks
    # (see docs/coding/agents.md).
    @agent.tool_plain
    def count_words(text: str) -> int:
        """Return the number of whitespace-separated words in `text`."""
        return len(text.split())

    sentence = "The quick brown fox jumps over the lazy dog"
    prompt = f"How many words are in this sentence: {sentence!r}?"

    print(f"gateway : {BASE_URL}")
    print(f"model   : {MODEL}")
    print(f"prompt  : {prompt}\n")

    result = agent.run_sync(prompt)

    # `result.output` is the model's final answer after any tool round-trips.
    print("answer  :", result.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
