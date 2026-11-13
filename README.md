# bookask

A tool-using RAG agent that answers questions about a book, implemented
**twice** — once by hand against the raw `google-genai` SDK, once with
LangChain + LangGraph — so the two can be compared side by side on
identical questions. Built as a hands-on exercise in agentic flows: tool
calling, retrieval grounding, structured output, and the control-flow
differences between driving your own loop versus using a framework's.

Both implementations answer questions about ["The Yellow
Wallpaper"](data/yellow_wallpaper.txt) (public domain, via Project
Gutenberg) using two tools:

- `search_book(query)` — retrieves relevant passages from a Chroma index of
  the book.
- `define_term(word)` — a plain-dictionary lookup for general-vocabulary
  questions that don't depend on the book's content.

Every answer is returned as structured JSON: `{answer, tool_calls, sources,
confidence}`, with `sources` citing the specific retrieved chunk(s) the
answer is grounded in.

## Project layout

```
src/bookask/
  bookask.py                      CLI: index the book, or ask without an agent (plain RAG)
  bookagent_raw.py                CLI: ask via the raw agent (agent_loop.py)
  bookagent_langchain_langgraph.py CLI: ask via the LangGraph agent
  agent_loop.py                   Raw agent: hand-written while-loop over google-genai
  tool_schemas.py                 Raw agent: hand-written tool JSON schemas
  output_schema.py                Structured {answer, tool_calls, sources, confidence} synthesis (shared by both agents)
  tools.py                        search_book / define_term implementations (shared by both agents)
  retriever.py, embedder.py,      RAG pipeline: chunking, embedding, Chroma storage/retrieval
  chunker.py, chroma_store.py
  book_loader.py                  Project Gutenberg boilerplate stripping
  prompts.py, generator.py        Plain (non-agentic) RAG answer generation, used by bookask.py
  config.py, gemini_client.py     .env / settings loading, Gemini client setup
evals/
  cases.py                        Fixed question set with tool-use expectations
  checks.py                       Deterministic checks: tool usage, citation grounding, confidence consistency
  run_evals.py                    Runs both agents against cases.py and scores them
data/yellow_wallpaper.txt         The indexed book (public domain)
```

## Setup

Requires Python 3.11+ and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

## Usage

Build the index once (re-run after changing the book or chunking settings —
it upserts by chunk id, so re-indexing the same book just overwrites):

```bash
python -m bookask.bookask --book data/yellow_wallpaper.txt --index
```

Ask a question via either agent:

```bash
python -m bookask.bookagent_raw --book data/yellow_wallpaper.txt \
  --ask "What does the narrator think of the wallpaper at the start of the story?"

python -m bookask.bookagent_langchain_langgraph --book data/yellow_wallpaper.txt \
  --ask "What does the narrator think of the wallpaper at the start of the story?"
```

Or skip the agent entirely and ask the plain (non-tool-using) RAG pipeline
directly:

```bash
python -m bookask.bookask --book data/yellow_wallpaper.txt \
  --ask "What does the narrator think of the wallpaper at the start of the story?"
```

## Evals

```bash
python -m evals.run_evals --book data/yellow_wallpaper.txt
```

Runs a fixed set of questions (book-content, multi-hop, vocabulary, and an
off-topic control) against both agents and checks, per answer:

1. **Tool-use correctness** — did it call the tools it was supposed to,
   and nothing else?
2. **Citation grounding** — does every `sources[].chunk_id` in the
   structured answer actually come from a `search_book` result, and does
   the cited excerpt substantially overlap with that chunk's real text (vs.
   a hallucinated quote)?
3. **Confidence/sources consistency** — a `"high"`-confidence answer that
   cites zero sources on a book-content question is treated as a
   contradiction, not a pass.

No LLM-as-judge — every check above is a plain Python assertion, so the
suite is free to run and fully repeatable.

One case (`off_topic_control`, "What is the capital of France?") is
expected to **fail**: both agents currently answer it anyway by routing
through `define_term` instead of declining as out-of-scope (see the
analysis below — it's a system-prompt gap shared by both
implementations, not something either framework causes or fixes on its
own). It's left failing on purpose so the eval suite demonstrates it caught
a real, specific defect rather than only reporting all-green results.

Pass `--impl raw` or `--impl langgraph` to run just one side, and
`--report evals/report.json` to write the full per-case results as JSON.

## Raw vs. framework: what the comparison found

Tested both on 8 questions against "The Yellow Wallpaper", including a
multi-hop one ("compare the narrator's feelings at the start of the story
versus the end") and an off-topic control ("what is the capital of
France?").

**Behavior:** identical on every question — same tools chosen, same
number of calls, same confidence level, materially the same answer text.
Both batch multiple `search_book` calls into a single turn for multi-part
questions rather than spreading them across iterations. Both also made
the same mistake on the off-topic control: instead of recognizing the
question was unrelated to the book, they called `define_term` and
answered "Paris" from the model's own general knowledge — a prompt-design
gap in `AGENT_SYSTEM_PROMPT`, not something either framework caused or
would fix on its own. (The `evals/` suite above now catches this
automatically instead of relying on a manual test run.)

**Latency:** a wash. Raw averaged 12.3s/question, framework 12.0s —
within API round-trip noise, no systematic gap. The abstraction layers
(schema generation, `bind_tools`, graph dispatch) add negligible overhead
next to the dominant cost of the Gemini network calls themselves.

**What you gain from the framework:**
- Tool schemas for free from type hints + docstrings (`@tool`) instead of
  hand-written `types.Schema` objects — less boilerplate, and it can't
  drift out of sync with the function signature.
- The loop's control flow (state accumulation, tool dispatch, routing)
  is declarative — a graph of nodes/edges instead of an imperative
  while-loop with manual list bookkeeping. Adding a third or fourth tool
  is a one-line change to the tool list; no dispatch dict, no schema, no
  loop logic to touch.
- `ToolNode` handles multi-call batching and per-call exception handling
  that the raw version wrote by hand.

**What you lose / what still costs you:**
- The forced-final-answer fallback (when the iteration cap is hit) isn't
  free — LangGraph's `recursion_limit` just raises `GraphRecursionError`;
  recovering the partial transcript to answer gracefully still required
  wiring up a checkpointer and reimplementing that fallback by hand.
- One more layer between you and the actual API call, which matters when
  debugging: the raw version's `contents` list and `generate_content()`
  call are one stack frame away; the framework version's request is
  built by `ChatGoogleGenerativeAI` several layers down.
- Two more dependencies (`langchain`, `langgraph`) with their own release
  cadence and breaking changes to track, versus one (`google-genai`).

**When you'd choose each:** raw for a small, fixed number of tools where
you want the full request/response cycle visible and minimal dependency
surface (this project's actual size). Framework once you're past ~3-4
tools, expect to add more, or want the loop's routing logic to be
inspectable/extensible (e.g. adding a human-approval step, retries, or
parallel branches) without hand-rolling that control flow again.

## License

MIT — see [LICENSE](LICENSE).
