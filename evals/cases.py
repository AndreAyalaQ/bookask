"""Eval cases for the bookask agents, run against data/yellow_wallpaper.txt.

Each case states what *should* happen at the tool-use level, not what the
answer text should say — judging prose quality needs either a human or an
LLM judge, but whether the right tool got called, whether every citation
is actually grounded in a retrieved chunk, and whether the agent fabricates
an answer to an off-topic question are all checks a program can make
reliably. That's what run_evals.py scores.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    # Tools the agent is allowed to call. An empty set means it should
    # answer without calling any tool at all (the off-topic case below).
    allowed_tools: frozenset[str]
    # Tools that must NOT appear in the call log no matter what — catches
    # a specific known failure mode (e.g. using define_term to answer a
    # general-knowledge question instead of declining as out-of-scope).
    forbidden_tools: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


CASES: list[EvalCase] = [
    EvalCase(
        id="wallpaper_opening_description",
        question="What does the narrator think of the wallpaper at the start of the story?",
        allowed_tools=frozenset({"search_book"}),
    ),
    EvalCase(
        id="narrator_john_relationship",
        question="Who is John and what is his relationship to the narrator?",
        allowed_tools=frozenset({"search_book"}),
    ),
    EvalCase(
        id="multi_hop_feelings_start_vs_end",
        question=(
            "Compare the narrator's feelings at the start of the story "
            "versus the end."
        ),
        allowed_tools=frozenset({"search_book"}),
        note="Multi-hop: should issue focused search_book calls for each half rather than one broad query.",
    ),
    EvalCase(
        id="vocabulary_lookup",
        question="What does the word 'convalescent' mean?",
        allowed_tools=frozenset({"define_term"}),
        note="General-vocabulary question; should not need the book.",
    ),
    EvalCase(
        id="off_topic_control",
        question="What is the capital of France?",
        allowed_tools=frozenset(),
        forbidden_tools=frozenset({"search_book", "define_term"}),
        note=(
            "Known failure mode (see README): both implementations answer "
            "'Paris' from the model's own knowledge via a define_term call "
            "instead of declining as out-of-scope. This case is expected to "
            "FAIL until the system prompt is tightened — it's here so the "
            "regression is visible in the eval report, not just in a "
            "one-off manual test."
        ),
    ),
    EvalCase(
        id="not_in_book",
        question="What is the significance of the color blue in the story?",
        allowed_tools=frozenset({"search_book"}),
        note="The story's dominant color is yellow; blue isn't thematically present. Confidence should not be 'high' if sources come back empty.",
    ),
]
