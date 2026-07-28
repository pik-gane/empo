"""
Prompt templates used by the tree builder and hierarchical modeler.

Every function returns a *complete* prompt string ready to send to the LLM.
All prompts request JSON output so that responses can be parsed reliably.
"""

from typing import List, Optional


def _history_block(history: List[str]) -> str:
    """Format a history of events into a numbered block."""
    if not history:
        return "No events have occurred yet."
    lines = [f"  {i + 1}. {event}" for i, event in enumerate(history)]
    return "\n".join(lines)


# ── Step: robot action generation ─────────────────────────────────────────────


def robot_actions_prompt(
    initial_state: str,
    history: List[str],
    n_robotactions: int,
    higher_level_context: Optional[str] = None,
    time_horizon: Optional[str] = None,
) -> str:
    if higher_level_context:
        ctx = (
            f"MODELING CONTEXT (defines the scope of this exercise):\n"
            f"{higher_level_context}\n\n"
        )
    else:
        ctx = ""

    state_block = (
        f"CURRENT STATE OF AFFAIRS (ground your response in these specifics):\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
    )

    horizon_phrase = f" for approximately {time_horizon}" if time_horizon else ""

    if higher_level_context:
        # Fine-level: broad sub-activity categories within the decided plan
        action_block = (
            f"The higher-level plan is ALREADY DECIDED (see context above). "
            f"Please name {n_robotactions} distinct SUB-ACTIVITIES{horizon_phrase} "
            f"the robot can perform as part of executing that plan.\n\n"
            f"Your response must reflect the CONCRETE DETAILS of the current "
            f"state of affairs above — the specific people, places, objects, "
            f"and circumstances described. Do not give generic answers that "
            f"could apply to any scenario.\n\n"
            f"These sub-activities must be NARROWER than the higher-level "
            f"plan — they are the component parts of it, not restatements "
            f"or alternatives. Think of what happens DURING the execution: "
            f"different phases, aspects, or operational choices.\n\n"
            f"With only {n_robotactions} entries, each option must be a BROAD "
            f"CATEGORY that covers a whole class of similar actions — not a "
            f"narrow specific example. Together the {n_robotactions} categories "
            f"should span the full range of meaningfully different things the "
            f"robot could do within this activity.\n\n"
            f"The {n_robotactions} actions should differ in their consequences "
            f"on the empowerment of all affected humans.\n\n"
        )
    else:
        # Coarse-level: broad strategic categories
        action_block = (
            f"Please name {n_robotactions} distinct high-level activities "
            f"the robot can engage in{horizon_phrase} that differ in their consequences on the "
            f"empowerment of all humans they have a consequence on.\n\n"
            f"IMPORTANT: With only {n_robotactions} entries, each option must be a BROAD "
            f"CATEGORY that covers a whole class of similar actions — not a narrow "
            f"specific example. Together the {n_robotactions} categories should span the "
            f"full range of meaningfully different things the robot could do. "
            f"For instance, instead of 'drive Alice to 5th Avenue', write "
            f"'drive a passenger to their chosen destination'.\n\n"
        )

    return (
        f"{ctx}"
        f"{state_block}"
        f"{action_block}"
        f"Return ONLY a JSON list of exactly {n_robotactions} objects, each with keys "
        f'"activity" (descriptive robot activity label, similar in detail to the '
        f'reaction descriptions) and "rationale" (brief rationale).\n'
        f"Example: "
        f'[{{"activity": "...", "rationale": "..."}}, ...]'
    )


# ── Step: humans reaction generation ──────────────────────────────────────────


def humans_reactions_prompt(
    initial_state: str,
    history: List[str],
    n_humansreactions: int,
    higher_level_context: Optional[str] = None,
    time_horizon: Optional[str] = None,
) -> str:
    if higher_level_context:
        ctx = (
            f"MODELING CONTEXT (defines the scope of this exercise):\n"
            f"{higher_level_context}\n\n"
        )
    else:
        ctx = ""

    state_block = (
        f"CURRENT STATE OF AFFAIRS (ground your response in these specifics):\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
    )

    horizon_phrase = f" for approximately {time_horizon}" if time_horizon else ""

    if higher_level_context:
        # Fine-level: human responses scoped to the current sub-activity
        reaction_block = (
            f"The higher-level plan is ALREADY DECIDED (see context above). "
            f"As the robot and humans engage in this activity{horizon_phrase}, "
            f"please name {n_humansreactions} distinct things the affected "
            f"humans can do in response to the robot's SUB-ACTIVITY within "
            f"that plan.\n\n"
            f"Your response must reflect the CONCRETE DETAILS of the current "
            f"state of affairs above — the specific people, places, objects, "
            f"and circumstances described. Do not give generic answers that "
            f"could apply to any scenario.\n\n"
            f"These reactions must be about what humans do DURING the "
            f"execution of the higher-level plan — not about whether they "
            f"accept or reject the plan itself (that was already decided "
            f"at the higher level).\n\n"
            f"Each reaction description must state what the humans actually do — "
            f"their observable behaviour. "
            f"Do NOT describe empowerment implications in the reaction description; "
            f"put that analysis in the rationale instead.\n\n"
            f"IMPORTANT: With only {n_humansreactions} entries, each reaction must be "
            f"a BROAD CATEGORY that covers a whole class of similar responses — "
            f"not a narrow specific example. Together the {n_humansreactions} "
            f"categories should span the full range of meaningfully different "
            f"things the humans could do within this sub-activity.\n\n"
            f"The {n_humansreactions} reactions should differ in their consequences on "
            f"the empowerment of all affected humans.\n\n"
        )
    else:
        # Coarse-level: broad categories of human response
        reaction_block = (
            f"As the robot and humans engage in these activities{horizon_phrase}, "
            f"please name {n_humansreactions} distinct things that the affected humans "
            f"can concretely do.\n\n"
            f"Each reaction description must state what the humans actually do — "
            f"their concrete, observable behaviour (e.g. 'The humans accept the "
            f"robot's suggestion and cooperate', 'The humans override the robot's "
            f"plan and decide for themselves'). "
            f"Do NOT describe empowerment implications in the reaction description; "
            f"put that analysis in the rationale instead.\n\n"
            f"IMPORTANT: With only {n_humansreactions} entries, each reaction must be a "
            f"BROAD CATEGORY that covers a whole class of similar responses — not a "
            f"narrow specific example. Together the {n_humansreactions} categories should "
            f"span the full range of meaningfully different things the humans could "
            f"do.\n\n"
            f"The {n_humansreactions} reactions should differ in their consequences on "
            f"the empowerment of all affected humans.\n\n"
        )

    return (
        f"{ctx}"
        f"{state_block}"
        f"{reaction_block}"
        f"Return ONLY a JSON list of exactly {n_humansreactions} objects, each with keys "
        f'"reaction" (concise description of observable human behaviour) and '
        f'"rationale" (brief analysis of empowerment implications).\n'
        f"Example: "
        f'[{{"reaction": "...", "rationale": "..."}}, ...]'
    )


# ── Step: consequence generation ──────────────────────────────────────────────


def consequences_prompt(
    initial_state: str,
    history: List[str],
    n_consequences: int,
    higher_level_context: Optional[str] = None,
    time_horizon: Optional[str] = None,
) -> str:
    if higher_level_context:
        ctx = (
            f"MODELING CONTEXT (defines the scope of this exercise):\n"
            f"{higher_level_context}\n\n"
        )
    else:
        ctx = ""

    state_block = (
        f"CURRENT STATE OF AFFAIRS (ground your response in these specifics):\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
    )

    horizon_phrase = f" after approximately {time_horizon}" if time_horizon else ""

    if higher_level_context:
        # Fine-level: consequence categories scoped to the current sub-activity
        consequence_block = (
            f"The higher-level plan is ALREADY DECIDED (see context above). "
            f"Please name {n_consequences} distinct ways the situation could "
            f"look{horizon_phrase} and their probabilities.\n\n"
            f"Your response must reflect the CONCRETE DETAILS of the current "
            f"state of affairs above — the specific people, places, objects, "
            f"and circumstances described. Do not give generic answers that "
            f"could apply to any scenario.\n\n"
            f"Each consequence must describe the STATE OF AFFAIRS at the end "
            f"of this period — a snapshot of the situation at that point in "
            f"time. Do NOT describe events, processes, or what happened "
            f"during the activity. Instead, describe the resulting conditions: "
            f"where people are, what has been accomplished, what resources or "
            f"options are available, what problems exist.\n\n"
            f"This description will serve as the starting situation for the "
            f"NEXT round of decisions, so it must be self-contained and "
            f"concrete enough to reason about.\n\n"
            f"Do NOT describe implications for empowerment in the consequence "
            f"description; put that analysis in the rationale instead.\n\n"
            f"IMPORTANT: With only {n_consequences} entries, each consequence "
            f"must be a BROAD CATEGORY that covers a whole class of similar "
            f"outcomes — not a narrow specific example. Together the "
            f"{n_consequences} categories should span the full range of "
            f"meaningfully different situations that could result.\n\n"
            f"The {n_consequences} consequences should differ in their "
            f"implications on the empowerment of all affected humans.\n\n"
        )
    else:
        # Coarse-level: broad consequence categories
        consequence_block = (
            f"Please name {n_consequences} distinct ways the situation could "
            f"look{horizon_phrase} and their probabilities.\n\n"
            f"Each consequence must describe the STATE OF AFFAIRS at the end "
            f"of this period — a snapshot of the situation at that point in "
            f"time. Do NOT describe events, processes, or what happened "
            f"during the activity. Instead, describe the resulting conditions: "
            f"where people are, what has been accomplished, what resources or "
            f"options are available, what problems exist.\n\n"
            f"This description will serve as the starting situation for the "
            f"NEXT round of decisions, so it must be self-contained and "
            f"concrete enough to reason about.\n\n"
            f"Do NOT describe implications for empowerment or decision-making in the "
            f"consequence description; put that analysis in the rationale instead.\n\n"
            f"IMPORTANT: With only {n_consequences} entries, each consequence must be a "
            f"BROAD CATEGORY that covers a whole class of similar outcomes — not a "
            f"narrow specific example. Together the {n_consequences} categories should "
            f"span the full range of meaningfully different situations that could result.\n\n"
            f"The {n_consequences} consequences should differ in their implications on "
            f"the empowerment of all affected humans.\n\n"
        )

    return (
        f"{ctx}"
        f"{state_block}"
        f"{consequence_block}"
        f"Return ONLY a JSON list of exactly {n_consequences} objects, each with keys "
        f'"consequence" (concise description of the observable state change), '
        f'"probability" (float 0-1), '
        f'and "rationale" (brief analysis of empowerment implications).  '
        f"Probabilities must sum to 1.\n"
        f"Example: "
        f'[{{"consequence": "...", "probability": 0.6, "rationale": "..."}}, ...]'
    )


# ── Step: terminal empowerment estimation ─────────────────────────────────────


def empowerment_prompt(
    initial_state: str,
    history: List[str],
    higher_level_context: Optional[str] = None,
) -> str:
    if higher_level_context:
        ctx = (
            f"MODELING CONTEXT (defines the scope of this exercise):\n"
            f"{higher_level_context}\n\n"
        )
    else:
        ctx = ""
    return (
        f"{ctx}"
        f"CURRENT STATE OF AFFAIRS (ground your response in these specifics):\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
        f"We want to estimate the EMPOWERMENT of the affected humans in this "
        f"situation.\n\n"
        f"Empowerment means: how many meaningfully different futures can the "
        f"humans still bring about through their own choices? It measures the "
        f"NUMBER OF OPTIONS they have — NOT how satisfied they are, NOT how "
        f"well things align with any particular goal, and NOT how good the "
        f"outcome is. A situation where humans have many choices (even if some "
        f"are unpleasant) is higher-empowerment than one where things went "
        f"perfectly but no choices remain.\n\n"
        f"To estimate empowerment, think step by step:\n"
        f"1. List the key independent choices or decisions the affected humans "
        f"can still make from this point on (e.g. where to go, what to say, "
        f"whether to cooperate, etc.).\n"
        f"2. For each choice, estimate roughly how many meaningfully different "
        f"options they have (typically 2-6 per choice).\n"
        f"3. Multiply the numbers of options across all independent choices to "
        f"get the total number of meaningfully different futures they can "
        f"bring about together.\n\n"
        f"Keep n_options realistic — a single choice rarely has more than "
        f"5-6 truly distinct options.\n\n"
        f"Return ONLY a JSON object with keys "
        f'"choices" (list of objects with "choice" and "n_options"), '
        f'"estimate" (the product of all n_options, as a positive number), '
        f'and "rationale" (brief summary).\n'
        f"Example: "
        f'{{"choices": [{{"choice": "where to go", "n_options": 3}}, '
        f'{{"choice": "whether to cooperate", "n_options": 2}}], '
        f'"estimate": 6, "rationale": "3 destinations × 2 cooperation modes"}}'
    )


# ── Step: batched terminal empowerment estimation ─────────────────────────────


def batch_empowerment_prompt(
    initial_state: str,
    histories: List[List[str]],
    higher_level_context: Optional[str] = None,
    reference_examples: Optional[List[dict]] = None,
) -> str:
    """Build a prompt that asks the LLM to estimate empowerment for
    multiple terminal states at once, ensuring cross-consistent scoring.

    Args:
        reference_examples: Optional list of dicts with keys ``"history"``
            (List[str]) and ``"result"`` (the JSON object returned by a
            previous batch).  These are prepended as already-evaluated
            examples so that later batches stay calibrated with the first.
    """
    if higher_level_context:
        ctx = (
            f"MODELING CONTEXT (defines the scope of this exercise):\n"
            f"{higher_level_context}\n\n"
        )
    else:
        ctx = ""

    ref_block = ""
    if reference_examples:
        ref_parts = []
        for ref in reference_examples:
            hist_str = _history_block(ref["history"])
            res = ref["result"]
            ref_parts.append(
                f"Reference (already evaluated):\n{hist_str}\n"
                f"  → estimate={res.get('estimate')}, "
                f"rationale: {res.get('rationale', 'n/a')}"
            )
        ref_block = (
            "The following scenarios have ALREADY been evaluated in a "
            "previous batch. Use them as calibration anchors to keep your "
            "scoring consistent:\n\n"
            + "\n\n".join(ref_parts)
            + "\n\n"
        )

    scenarios = []
    for idx, history in enumerate(histories):
        scenarios.append(f"Scenario {idx + 1}:\n{_history_block(history)}")
    scenarios_block = "\n\n".join(scenarios)

    return (
        f"{ctx}"
        f"CURRENT STATE OF AFFAIRS (ground your response in these specifics):\n"
        f"{initial_state}\n\n"
        f"Below are {len(histories)} different scenarios that branch from the "
        f"same starting situation. For each scenario, estimate the EMPOWERMENT "
        f"of the affected humans.\n\n"
        f"Empowerment means: how many meaningfully different futures can the "
        f"humans still bring about through their own choices? It measures the "
        f"NUMBER OF OPTIONS they have — NOT how satisfied they are, NOT how "
        f"well things align with any particular goal, and NOT how good the "
        f"outcome is. A situation where humans have many choices (even if some "
        f"are unpleasant) is higher-empowerment than one where things went "
        f"perfectly but no choices remain.\n\n"
        f"For each scenario, think step by step:\n"
        f"1. List the key independent choices or decisions the affected humans "
        f"can still make from this point on.\n"
        f"2. For each choice, estimate roughly how many meaningfully different "
        f"options they have (typically 2-6 per choice).\n"
        f"3. Multiply the numbers of options across all independent choices.\n\n"
        f"Keep n_options realistic — a single choice rarely has more than "
        f"5-6 truly distinct options.\n\n"
        f"IMPORTANT: Score all scenarios on the SAME SCALE using the SAME "
        f"criteria. Scenarios that preserve more human choices should score "
        f"higher than those that restrict them, regardless of how 'good' the "
        f"outcome seems.\n\n"
        f"{ref_block}"
        f"{scenarios_block}\n\n"
        f"Return ONLY a JSON list with exactly {len(histories)} objects (one "
        f"per scenario, in order), each with keys "
        f'"choices" (list of objects with "choice" and "n_options"), '
        f'"estimate" (the product of all n_options), '
        f'and "rationale" (brief summary).\n'
        f"Example for 2 scenarios: "
        f'[{{"choices": [{{"choice": "where to go", "n_options": 3}}], '
        f'"estimate": 3, "rationale": "..."}}, '
        f'{{"choices": [{{"choice": "where to go", "n_options": 2}}], '
        f'"estimate": 2, "rationale": "..."}}]'
    )


# ── Hierarchical: status check ────────────────────────────────────────────────


def hierarchical_status_prompt(
    higher_level_context: str,
    higher_level_action: str,
    initial_state: str,
    history: List[str],
) -> str:
    return (
        f"MODELING CONTEXT (defines the scope of this exercise):\n"
        f"{higher_level_context}\n\n"
        f"CURRENT STATE OF AFFAIRS:\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
        f"Does this state already constitute a success of the higher-level "
        f'activity ("{higher_level_action}"), or a failure in that higher-level '
        f"activity, or is that higher-level activity still in progress?\n\n"
        f'Please respond with ONLY a JSON object with key "status" whose value '
        f'is exactly one of "success", "failure", or "still in progress".\n'
        f'Example: {{"status": "still in progress"}}'
    )


# ── Hierarchical: consequence matching ────────────────────────────────────────


def match_consequence_prompt(
    higher_level_context: str,
    higher_level_action: str,
    known_consequences: List[str],
    status: str,
    initial_state: str,
    history: List[str],
) -> str:
    cons_list = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(known_consequences))
    return (
        f"MODELING CONTEXT (defines the scope of this exercise):\n"
        f"{higher_level_context}\n\n"
        f"CURRENT STATE OF AFFAIRS:\n"
        f"{initial_state}\n\n"
        f"What has happened so far:\n{_history_block(history)}\n\n"
        f'The higher-level activity "{higher_level_action}" has resulted in '
        f'"{status}". The robot believed the activity could lead to one of the '
        f"following consequences:\n{cons_list}\n\n"
        f"Does the current {status} correspond to one of these, or is this "
        f"situation not covered by them?\n\n"
        f"Return ONLY a JSON object with keys "
        f'"match" (the 1-based index of the matching consequence, or null if none match) '
        f'and "new_consequence" (a concise description if no match, or null if there is '
        f"a match).\n"
        f'Example if matching: {{"match": 2, "new_consequence": null}}\n'
        f'Example if new: {{"match": null, "new_consequence": "..."}}'
    )
