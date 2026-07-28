#!/usr/bin/env python3
"""
Two-level airport taxi example using the simple hierarchical LLM modeler.

Scenario
--------
A robot taxi driver and two potential passengers stand in front of the
departure building of an airport.

* **Level 0 (coarse):** High-level decisions – e.g. "offer to drive both
  passengers together", "offer separate rides", "wait for instructions".
* **Level 1 (fine):** Detailed execution of the chosen high-level plan –
  e.g. "open trunk for luggage", "confirm destination", "start driving".

This script can run in two modes:

1. **Mock mode** (default) – uses a deterministic MockLLM so no API key is
   needed.  Useful for testing and understanding the data flow.
2. **Live mode** – uses a real LLM via the L2P OpenAI connector.  Requires
   an API key.

Usage
-----
::

    # Mock mode (no API key needed):
    PYTHONPATH=src:vendor/multigrid:vendor/l2p python examples/hierarchical_llm/airport_taxi_example.py

    # Live mode (requires GOOGLE_API_KEY):
    PYTHONPATH=src:vendor/multigrid:vendor/l2p python examples/hierarchical_llm/airport_taxi_example.py --live
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap

# ---------------------------------------------------------------------------
# Mock LLM – deterministic, no network access
# ---------------------------------------------------------------------------

DEFAULT_SCENARIO = (
    "A robot taxi driver and two potential passengers (Alice and Bob) stand "
    "in front of the departure building of an airport."
)


class AirportMockLLM:
    """Returns plausible canned responses for the airport-taxi scenario."""

    @property
    def context_length(self) -> int:
        return 1_000_000

    def query(self, prompt: str) -> str:
        """Return a deterministic JSON response based on prompt keywords."""
        # --- Batched empowerment estimate (must be checked before humans
        #     reactions because the empowerment prompt also contains the
        #     words "things" and "affected humans") ---
        if "Score all scenarios on the SAME SCALE" in prompt:
            # Count how many scenarios are in the prompt
            n_scenarios = prompt.count("Scenario ")
            return json.dumps([
                {
                    "choices": [{"choice": "destination", "n_options": 3},
                                {"choice": "cooperation mode", "n_options": 2}],
                    "estimate": 12 - i,
                    "rationale": f"Passengers can still choose destinations, routes, and whether to continue together (scenario {i + 1})",
                }
                for i in range(n_scenarios)
            ])

        # --- Robot actions ---
        if "robot can perform" in prompt or "robot can engage in" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "activity": "Offer to drive both passengers together, dropping Bob at the train station first",
                    "rationale": "Maximises both passengers' options by getting them moving quickly and efficiently",
                },
                {
                    "activity": "Ask the passengers about their destinations before deciding",
                    "rationale": "Gathering information preserves flexibility for all parties",
                },
                {
                    "activity": "Offer to drive Alice to the city centre first and come back for Bob",
                    "rationale": "Prioritises Alice but reduces Bob's options by making him wait",
                },
            ]
            return json.dumps(pool[:n])

        # --- Humans reactions ---
        if "things" in prompt and "affected humans" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "reaction": "Both passengers agree and get into the taxi",
                    "rationale": "Cooperative outcome that preserves everyone's empowerment",
                },
                {
                    "reaction": "Alice agrees but Bob decides to take a different taxi",
                    "rationale": "Bob exercises his own choice, reducing coordination but preserving his autonomy",
                },
                {
                    "reaction": "Both passengers decline and decide to share a ride-share instead",
                    "rationale": "Passengers choose an alternative, reducing robot's influence but preserving their options",
                },
            ]
            return json.dumps(pool[:n])

        # --- Consequences ---
        if "ways the situation could look" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "consequence": "The taxi departs smoothly with the agreed passengers",
                    "probability": 0.7,
                    "rationale": "Most likely outcome when passengers have agreed",
                },
                {
                    "consequence": "A traffic jam blocks the route and the passengers must choose an alternative",
                    "probability": 0.3,
                    "rationale": "Traffic is common near the airport, creating new decision points",
                },
            ]
            selected = pool[:n]
            total = sum(c["probability"] for c in selected)
            for c in selected:
                c["probability"] = round(c["probability"] / total, 2)
            return json.dumps(selected)

        # --- Individual empowerment estimate (fallback) ---
        if "meaningfully different futures" in prompt:
            return json.dumps(
                {
                    "estimate": 12,
                    "rationale": "Passengers can still choose destinations, routes, and whether to continue together",
                }
            )

        # --- Hierarchical status ---
        if "success of the higher-level" in prompt:
            return json.dumps({"status": "still in progress"})

        # --- Consequence matching ---
        if "correspond to one of these" in prompt:
            return json.dumps({"match": 1, "new_consequence": None})

        return json.dumps({"error": "unrecognised prompt"})


def _extract_n(prompt: str) -> int:
    m = re.search(r"name\s+(\d+)", prompt)
    return int(m.group(1)) if m else 2


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _indent(text: str, level: int = 1) -> str:
    return textwrap.indent(text, "  " * level)


def print_tree(node, depth: int = 0) -> None:
    prefix = "  " * depth
    tag = node.node_type.upper()
    hist = " → ".join(node.history[-1:]) if node.history else "(root)"
    emp = ""
    if node.empowerment_estimate is not None:
        emp = f"  [empowerment ≈ {node.empowerment_estimate}]"
    print(f"{prefix}[{tag}] {hist}{emp}")
    for label, prob, child in node.children:
        pstr = f" (p={prob:.2f})" if node.node_type == "humansreaction" else ""
        print(f"{prefix}  ├─ {label}{pstr}")
        print_tree(child, depth + 2)


def print_world_model(model) -> None:
    print("\n=== NLWorldModel Summary ===")
    print(f"  States: {len(model.states)}")
    print(f"  Terminal states: {len(model.terminal_states)}")
    print(f"  Robot actions at root: {model.robot_action_labels()}")
    s0 = model.get_state()
    for ra_idx, ra_label in enumerate(model.robot_action_labels()):
        hr_labels = model.humans_reaction_labels(robot_action_index=ra_idx)
        for hr_idx, hr_label in enumerate(hr_labels):
            trans = model.transition_probabilities(s0, [ra_idx, hr_idx])
            if trans:
                outcomes = ", ".join(
                    f"p={p:.2f}→{model.state_description(ns)[:60]}..."
                    for p, ns in trans
                )
                print(f"  ({ra_label}, {hr_label}): {outcomes}")
    for ts in model.terminal_states:
        v = model.V_r_estimate(ts)
        print(f"  Terminal V_r({model.state_description(ts)[:50]}...) = {v:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Airport taxi hierarchical LLM example"
    )
    parser.add_argument(
        "--live", action="store_true", help="Use a real LLM (needs API key)"
    )
    parser.add_argument(
        "--llm", type=str, default="google/gemini-2.5-flash",
        metavar="PROVIDER/MODEL",
        help="LLM to use in live mode as provider/model "
             "(default: google/gemini-2.5-flash). "
             "Providers: google, openai, deepseek, mistral, huggingface, anthropic"
    )
    parser.add_argument(
        "--depth", type=int, default=1, help="Number of full expansion cycles (default: 1)"
    )
    parser.add_argument(
        "--situation", type=str, default=DEFAULT_SCENARIO,
        help="Initial situation description (default: airport taxi scenario)"
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="PATH",
        help="Export tree to file. Format is inferred from extension: "
             ".json (default), .yaml/.yml, or .html"
    )
    args = parser.parse_args()

    scenario = args.situation

    if args.live:
        try:
            import os
            from l2p.llm.openai import OPENAI

            # Parse provider/model from --llm argument
            if "/" in args.llm:
                provider, model = args.llm.split("/", 1)
            else:
                provider, model = "google", args.llm

            # Provider-specific configuration
            _PROVIDER_CONFIG = {
                "google": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "api_key_env": "GOOGLE_API_KEY",
                },
                "openai": {
                    "base_url": "https://api.openai.com/v1/",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "deepseek": {
                    "base_url": "https://api.deepseek.com/v1/",
                    "api_key_env": "DEEPSEEK_API_KEY",
                },
                "mistral": {
                    "base_url": "https://api.mistral.ai/v1/",
                    "api_key_env": "MISTRAL_API_KEY",
                },
                "huggingface": {
                    "base_url": "https://api-inference.huggingface.co/v1/",
                    "api_key_env": "HF_API_KEY",
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            }

            if provider not in _PROVIDER_CONFIG:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Choose from: {', '.join(_PROVIDER_CONFIG)}"
                )

            cfg = _PROVIDER_CONFIG[provider]
            api_key = os.environ.get(cfg["api_key_env"])
            if not api_key:
                raise ValueError(
                    f"Set {cfg['api_key_env']} environment variable for {provider}"
                )

            print(f"Using LLM: {provider}/{model}")
            if provider == "anthropic":
                from empo.simple_hierarchical_llm_modeler.llm_connector import (
                    AnthropicConnector,
                )
                llm = AnthropicConnector(model=model, api_key=api_key)
            else:
                llm = OPENAI(
                    model=model,
                    provider=provider,
                    api_key=api_key,
                    base_url=cfg["base_url"],
                )
        except Exception as exc:
            print(f"Cannot initialise live LLM: {exc}")
            print("Falling back to mock mode.")
            llm = AirportMockLLM()
            args.live = False  # prevent caching mock responses to disk
    else:
        llm = AirportMockLLM()
        print("Running in MOCK mode (deterministic, no API key needed).\n")

    # Import after path setup
    from empo.simple_hierarchical_llm_modeler import (
        CachedLLMConnector,
        LiveTreeRenderer,
        StatsTrackingLLM,
        build_tree,
        build_two_level_model,
        collect_leaves,
        count_nodes,
        NLWorldModel,
        AllSingleStatesGoalGenerator,
    )
    from empo.backward_induction.phase1 import compute_human_policy_prior
    from empo.backward_induction.phase2 import compute_robot_policy

    # Wrap live LLM with a disk cache to avoid re-querying on restarts.
    # Mock mode is deterministic and instant, so no caching needed.
    raw_llm = llm if args.live else None
    if args.live:
        cache_model = args.llm.replace("/", "_")
        llm = CachedLLMConnector(llm, cache_dir=f"outputs/llm_cache/{cache_model}")

    # Wrap with stats tracking (uses real token counts from L2P when available)
    llm = StatsTrackingLLM(llm, raw_llm=raw_llm)

    # ── Coarse-level tree ──────────────────────────────────────────────────
    if not args.live:
        print("=" * 70)
        print("Building coarse-level hierarchical world model")
        print("=" * 70)

    renderer = LiveTreeRenderer(root_label=scenario) if args.live else None

    hmodel = build_two_level_model(
        llm,
        scenario,
        coarse_n_steps=args.depth,
        fine_n_steps=args.depth,
        n_robotactions=3,
        n_humansreactions=2,
        n_consequences=2,
        coarse_time_horizon="one day",
        fine_time_horizon="one hour",
        on_update=renderer.update if renderer else None,
    )
    if renderer:
        renderer.finish()
        import time; time.sleep(2)
        renderer.close()

    coarse_tree = hmodel._coarse_tree
    coarse_wm = hmodel.coarsest()

    if not args.live:
        print(f"\nCoarse tree: {count_nodes(coarse_tree)} nodes, "
              f"{len(collect_leaves(coarse_tree))} leaves.\n")
        print(coarse_tree.render(root_label=scenario))
        print_world_model(coarse_wm)

    # ── Phase 1 + Phase 2 backward induction on coarse level ──────────────
    print(f"\n{'=' * 70}")
    print("Backward induction on coarse-level model")
    print("=" * 70)

    coarse_goal_gen = AllSingleStatesGoalGenerator(coarse_wm)
    coarse_hpp = compute_human_policy_prior(
        coarse_wm,
        human_agent_indices=coarse_wm.human_agent_indices,
        possible_goal_generator=coarse_goal_gen,
        beta_h=10.0,
        quiet=True,
    )
    # V_r_estimate returns empowerment in bits (positive, log2 of options).
    # Phase 2 needs strictly negative V_r values.  The mapping
    #   V_r(s) = -exp(-S(s))   where S = empowerment in bits
    # is always < 0 and increases toward 0 with more empowerment.
    def _neg_vr(state):
        return -math.exp(-coarse_wm.V_r_estimate(state))

    coarse_robot_policy, coarse_Vr, coarse_Vh = compute_robot_policy(
        coarse_wm,
        human_agent_indices=coarse_wm.human_agent_indices,
        robot_agent_indices=coarse_wm.robot_agent_indices,
        possible_goal_generator=coarse_goal_gen,
        human_policy_prior=coarse_hpp,
        beta_r=10.0,
        terminal_Vr=_neg_vr,
        return_values=True,
        quiet=True,
    )
    print("  Coarse V_r by state:")
    for state, vr in sorted(coarse_Vr.items(), key=lambda x: len(x[0])):
        desc = coarse_wm.state_description(state)[:70]
        print(f"    V_r={vr:+.4f}  {desc}")

    # ── Hierarchical rollout ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Hierarchical rollout (coarse + fine)")
    print("=" * 70)

    import numpy as np

    story_lines: list[str] = []          # plain-text story
    html_sections: list[dict] = []       # for HTML export

    coarse_state = coarse_wm.get_state()  # root = ()

    for coarse_step in range(args.depth):
        step_label = f"Coarse step {coarse_step + 1}"
        print(f"\n── {step_label} ──")

        # --- Sample robot action from computed robot policy ---
        ra_profile = coarse_robot_policy.sample(coarse_state)
        ra_idx = ra_profile[0]
        ra_labels = coarse_wm.robot_action_labels(coarse_state)
        ra_label = ra_labels[ra_idx] if ra_idx < len(ra_labels) else f"action_{ra_idx}"
        print(f"  Robot action (sampled): {ra_label}")
        story_lines.append(f"[{step_label}] Robot decides: {ra_label}")

        # --- Sample human reaction from human policy prior ---
        hr_dist = coarse_hpp(coarse_state, 1)  # marginal for agent 1
        hr_idx = int(np.random.choice(len(hr_dist), p=hr_dist))
        hr_labels = coarse_wm.humans_reaction_labels(coarse_state, ra_idx)
        hr_label = hr_labels[hr_idx] if hr_idx < len(hr_labels) else f"reaction_{hr_idx}"
        print(f"  Humans react (sampled): {hr_label}")
        story_lines.append(f"  Humans react: {hr_label}")

        # --- Sample consequence from transition probabilities ---
        trans = coarse_wm.transition_probabilities(coarse_state, [ra_idx, hr_idx])
        if trans:
            probs = np.array([p for p, _ in trans])
            probs = probs / probs.sum()
            cons_idx = int(np.random.choice(len(trans), p=probs))
            cons_prob, cons_state = trans[cons_idx]
        else:
            cons_state = coarse_state
            cons_prob = 1.0
        cons_desc = coarse_wm.state_description(cons_state)
        # Extract just the last event from the state tuple for a concise label
        cons_short = cons_state[-1] if cons_state else cons_desc
        print(f"  Consequence (p={cons_prob:.2f}): {cons_short}")
        story_lines.append(f"  Consequence (p={cons_prob:.2f}): {cons_short}")

        # Add coarse step to HTML sections
        html_sections.append({
            "level": "coarse",
            "step": coarse_step + 1,
            "robot_action": ra_label,
            "human_reaction": hr_label,
            "consequence": cons_short,
            "consequence_prob": cons_prob,
            "fine_events": [],
        })

        # ── Fine-level model for this coarse action ───────────────────────
        print(f"\n  Building fine-level model for: {ra_label}")
        print(f"    Human reaction context: {hr_label}")

        context = [
            f"▸ Situation: {scenario}",
            f"▸ Coarse action: {ra_label}",
            f"▸ Human reaction: {hr_label}",
        ]
        renderer_fine = LiveTreeRenderer(
            root_label=scenario, context_lines=context
        ) if args.live else None

        fine_wm, mapper = hmodel.get_fine_model(
            ra_label,
            coarse_human_reaction_label=hr_label,
            on_update=renderer_fine.update if renderer_fine else None,
        )
        if renderer_fine:
            renderer_fine.finish()
            import time; time.sleep(2)
            renderer_fine.close()

        fine_tree = hmodel._fine_trees.get((ra_label, hr_label))
        if not args.live and fine_tree is not None:
            print(f"    Fine tree: {count_nodes(fine_tree)} nodes, "
                  f"{len(collect_leaves(fine_tree))} leaves.")

        # Phase 1 + Phase 2 on fine level
        print("    Backward induction on fine-level model...")
        fine_goal_gen = AllSingleStatesGoalGenerator(fine_wm)
        fine_hpp = compute_human_policy_prior(
            fine_wm,
            human_agent_indices=fine_wm.human_agent_indices,
            possible_goal_generator=fine_goal_gen,
            beta_h=10.0,
            quiet=True,
        )
        def _neg_vr_fine(state):
            return -math.exp(-fine_wm.V_r_estimate(state))

        fine_robot_policy, fine_Vr, fine_Vh = compute_robot_policy(
            fine_wm,
            human_agent_indices=fine_wm.human_agent_indices,
            robot_agent_indices=fine_wm.robot_agent_indices,
            possible_goal_generator=fine_goal_gen,
            human_policy_prior=fine_hpp,
            beta_r=10.0,
            terminal_Vr=_neg_vr_fine,
            return_values=True,
            quiet=True,
        )

        # Fine-level rollout
        fine_state = fine_wm.get_state()  # root
        fine_events: list[str] = []

        for fine_step in range(args.depth):
            fine_step_label = f"Fine step {fine_step + 1}"

            # Sample fine robot action
            fra_profile = fine_robot_policy.sample(fine_state)
            fra_idx = fra_profile[0]
            fra_labels = fine_wm.robot_action_labels(fine_state)
            fra_label = fra_labels[fra_idx] if fra_idx < len(fra_labels) else f"action_{fra_idx}"

            # Sample fine human reaction
            fhr_dist = fine_hpp(fine_state, 1)
            fhr_idx = int(np.random.choice(len(fhr_dist), p=fhr_dist))
            fhr_labels = fine_wm.humans_reaction_labels(fine_state, fra_idx)
            fhr_label = fhr_labels[fhr_idx] if fhr_idx < len(fhr_labels) else f"reaction_{fhr_idx}"

            # Sample fine consequence
            ftrans = fine_wm.transition_probabilities(fine_state, [fra_idx, fhr_idx])
            if ftrans:
                fprobs = np.array([p for p, _ in ftrans])
                fprobs = fprobs / fprobs.sum()
                fcons_idx = int(np.random.choice(len(ftrans), p=fprobs))
                fcons_prob, fine_next = ftrans[fcons_idx]
            else:
                fine_next = fine_state
                fcons_prob = 1.0
            fcons_short = fine_next[-1] if fine_next else "?"

            print(f"      [{fine_step_label}] Robot: {fra_label}")
            print(f"        Humans: {fhr_label}")
            print(f"        Consequence (p={fcons_prob:.2f}): {fcons_short}")

            fine_events.append(
                f"[{fine_step_label}] Robot: {fra_label} → "
                f"Humans: {fhr_label} → "
                f"Consequence (p={fcons_prob:.2f}): {fcons_short}"
            )
            story_lines.append(f"    {fine_events[-1]}")
            html_sections[-1]["fine_events"].append({
                "step": fine_step + 1,
                "robot_action": fra_label,
                "human_reaction": fhr_label,
                "consequence": fcons_short,
                "consequence_prob": fcons_prob,
            })

            fine_state = fine_next

        # Fine terminal empowerment
        fine_emp = fine_wm.V_r_estimate(fine_state)
        print(f"    Fine terminal V_r_estimate = {fine_emp:.3f}")
        story_lines.append(f"    Fine terminal empowerment (log2): {fine_emp:.3f}")
        html_sections[-1]["fine_terminal_emp"] = fine_emp

        # Advance coarse state to the sampled consequence
        coarse_state = cons_state

    # ── Print story ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ROLLOUT STORY")
    print("=" * 70)
    print(f"Scenario: {scenario}\n")
    for line in story_lines:
        print(line)
    print()

    # ── Export ─────────────────────────────────────────────────────────────
    export_path = args.export or "outputs/airport_taxi_rollout.html"
    _export_rollout_html(export_path, scenario, html_sections, coarse_tree, hmodel)
    print(f"Rollout exported to {export_path}")

    # ── Export trees if requested in non-HTML format ──────────────────────
    if args.export and not args.export.endswith(".html"):
        ext = args.export.rsplit(".", 1)[-1].lower() if "." in args.export else "json"
        fmt = {"json": "json", "yaml": "yaml", "yml": "yaml"}.get(ext, "json")
        coarse_tree.export(args.export, fmt=fmt, root_label=scenario)
        print(f"Tree exported to {args.export} ({fmt})")

    # Print token usage statistics
    llm.print_report()

    print("\nDone!")


def _export_rollout_html(
    path: str,
    scenario: str,
    sections: list[dict],
    coarse_tree,
    hmodel,
) -> None:
    """Write a pretty HTML file with collapsible fine-level details."""
    import html as html_mod
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    parts: list[str] = []
    parts.append(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Airport Taxi – Hierarchical Rollout</title>"
        "<style>"
        "body{font-family:'Segoe UI',system-ui,sans-serif;"
        "background:#1a1a2e;color:#e0e0e0;max-width:900px;margin:2em auto;padding:0 1em}"
        "h1{color:#e94560;border-bottom:2px solid #e94560;padding-bottom:.3em}"
        "h2{color:#0f3460;background:#16213e;padding:.5em .8em;border-radius:6px;"
        "border-left:4px solid #e94560;margin-top:1.5em}"
        ".scenario{color:#a0a0c0;font-style:italic;margin-bottom:1.5em}"
        ".coarse-step{background:#16213e;border-radius:8px;padding:1em 1.2em;"
        "margin:1em 0;border:1px solid #0f3460}"
        ".coarse-step h3{color:#e94560;margin:0 0 .5em}"
        ".event{margin:.3em 0;padding:.2em 0}"
        ".robot{color:#5ec4c4;font-weight:bold}"
        ".humans{color:#e6c84c;font-weight:bold}"
        ".consequence{color:#8fbcbb}"
        ".prob{color:#888;font-size:.85em}"
        "details{background:#0f3460;border-radius:6px;padding:.6em 1em;margin:.6em 0;"
        "border:1px solid #1a3a6a}"
        "details[open]{padding-bottom:.8em}"
        "summary{cursor:pointer;color:#88c0d0;font-weight:bold;padding:.3em 0}"
        "summary:hover{color:#5ec4c4}"
        ".fine-step{margin:.4em 0 .4em 1em;padding:.3em 0;"
        "border-left:2px solid #333;padding-left:.8em}"
        ".emp{color:#8f8;font-size:.9em;margin-top:.5em}"
        "</style></head><body>"
    )
    parts.append(f"<h1>Airport Taxi – Hierarchical Rollout</h1>")
    parts.append(f'<div class="scenario">{html_mod.escape(scenario)}</div>')

    for sec in sections:
        parts.append(f'<div class="coarse-step">')
        parts.append(f'<h3>Coarse Step {sec["step"]}</h3>')
        parts.append(
            f'<div class="event">'
            f'<span class="robot">Robot:</span> {html_mod.escape(sec["robot_action"])}'
            f'</div>'
        )
        parts.append(
            f'<div class="event">'
            f'<span class="humans">Humans:</span> {html_mod.escape(sec["human_reaction"])}'
            f'</div>'
        )
        parts.append(
            f'<div class="event">'
            f'<span class="consequence">Consequence:</span> '
            f'{html_mod.escape(str(sec["consequence"]))}'
            f' <span class="prob">(p={sec["consequence_prob"]:.2f})</span>'
            f'</div>'
        )

        if sec["fine_events"]:
            parts.append(
                f'<details><summary>Fine-level execution '
                f'({len(sec["fine_events"])} steps)</summary>'
            )
            for fe in sec["fine_events"]:
                parts.append(f'<div class="fine-step">')
                parts.append(
                    f'<span class="robot">Robot:</span> '
                    f'{html_mod.escape(fe["robot_action"])}'
                )
                parts.append(
                    f' → <span class="humans">Humans:</span> '
                    f'{html_mod.escape(fe["human_reaction"])}'
                )
                parts.append(
                    f' → <span class="consequence">'
                    f'{html_mod.escape(str(fe["consequence"]))}</span>'
                    f' <span class="prob">(p={fe["consequence_prob"]:.2f})</span>'
                )
                parts.append("</div>")
            emp = sec.get("fine_terminal_emp")
            if emp is not None:
                parts.append(
                    f'<div class="emp">Fine terminal empowerment '
                    f'(log₂): {emp:.3f}</div>'
                )
            parts.append("</details>")
        parts.append("</div>")

    parts.append("</body></html>")

    with open(path, "w") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
