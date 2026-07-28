"""
Minimal example: build a WorldModel from a chess scene description
using the free Gemini 2.0 Flash API.

Requirements:
    pip install retry openai tiktoken

Usage:
    export GOOGLE_API_KEY="your-free-gemini-api-key"
    PYTHONPATH=src:vendor/l2p python examples/llm/chess_world_model_example.py

Get a free API key at https://aistudio.google.com/apikey
"""

import logging
import os
import sys

# Path to the L2P openaiSDK.yaml config
L2P_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "vendor", "l2p", "l2p", "llm", "utils", "openaiSDK.yaml",
    )
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY env var (free at https://aistudio.google.com/apikey)")
        sys.exit(1)

    # L2P's OPENAI class works with any OpenAI-SDK-compatible endpoint,
    # including Gemini's free tier.
    from l2p.llm.openai import OPENAI

    llm = OPENAI(
        model="gemini-2.0-flash",
        provider="google",
        config_path=L2P_CONFIG,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    from empo.llm_world_model import WorldModelBuilder

    builder = WorldModelBuilder(llm=llm)

    scene = (
        "A robot (X) and a human (O) play tic-tac-toe on a 3x3 grid. "
        "Each player can place their mark on one empty cell per turn. "
        "The game proceeds in alternating turns, with X moving first. "
        "A player wins by getting three marks in a row, column, or diagonal."
    )

    print(f"\nScene: {scene}\n")
    print("Building world model (this makes several LLM calls)...\n")

    world_model = builder.build(scene)

    # Inspect the result
    state = world_model.get_state()
    print(f"\nWorld model built successfully!")
    print(f"  Agents: {[a.name for a in builder.domain.agents]}")
    print(f"  Ground atoms: {world_model._num_atoms}")
    print(f"  Actions per agent: {world_model._agent_action_counts}")
    print(f"  State: {state}")


if __name__ == "__main__":
    main()
