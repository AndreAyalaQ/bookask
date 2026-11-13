"""Framework version of the agent, for side-by-side comparison with
agent_loop.py + tool_schemas.py + output_schema.py + try_agent.py (the
"raw" version, which calls google-genai directly and drives its own
while-loop). This file swaps in LangChain for tool definitions and
structured output, and LangGraph for orchestration.

It deliberately does NOT touch the raw version or reimplement the RAG
logic: tools.py's search_book()/define_term() (which themselves wrap
retriever.py's Chroma query and the embed/generate calls) are imported
and reused unchanged. Only the "how do we describe a tool to the model"
and "how do we loop until the model is done" layers are replaced.

Usage:
    python bookagent_langchain_langgraph.py --book <path> --ask "<question>"

Each place below where a framework replaces something the raw version
did by hand is marked with a "RAW VERSION vs FRAMEWORK" comment.
"""

import json
import logging
import uuid

import click
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .chroma_store import get_client, get_or_create_collection
from .config import load_settings, setup_logging
from .gemini_client import get_gemini_client
from .output_schema import AgentAnswer, SYNTHESIS_SYSTEM_PROMPT, _format_tool_call_log
from .tools import define_term as raw_define_term
from .tools import search_book as raw_search_book

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = ".bookask_chroma"
DEFAULT_COLLECTION = "book"

AGENT_MODEL = "gemini-3.1-flash-lite"
AGENT_TEMPERATURE = 0.2
SYNTHESIS_MODEL = "gemini-3.1-flash-lite"
SYNTHESIS_TEMPERATURE = 0.0
MAX_ITERATIONS = 5

AGENT_SYSTEM_PROMPT = """\
You are a research agent that answers questions about a specific book \
using two tools:

- search_book(query): retrieves excerpts from the book relevant to a \
query. Use this for anything about the book's plot, characters, themes, \
or content.
- define_term(word): looks up a plain, general-knowledge definition of a \
word. Use this only when a question needs the ordinary meaning of a \
word, not the book's content.

You have no built-in knowledge of this book's content — only what \
search_book returns to you. Never answer a book-content question from \
memory or assumption; call search_book first.

If a question has multiple parts (for example, comparing two points in \
the story, or covering two different characters), call search_book \
separately for each part with a focused query, rather than one broad \
query for the whole question.

Once you have enough information to answer, stop calling tools and \
respond with your answer in plain text. If, after searching, the book \
does not contain enough information to answer, say so plainly rather \
than guessing.
"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
#
# RAW VERSION vs FRAMEWORK: tool_schemas.py hand-wrote a types.FunctionDeclaration
# per tool — name, description, and a types.Schema spelling out each
# parameter's type and description by hand, then bundled them into a
# types.Tool. Here, @tool reads the function's type hints (query: str) and
# its docstring and generates that same JSON schema automatically. The
# docstrings below are copied near-verbatim from tool_schemas.py's
# `description=` strings, because that wording is still doing real work —
# swapping the schema-building mechanism doesn't remove the need to tell
# the model when (and when not) to use each tool.
#
# gemini_client and collection are runtime context the model can never
# supply (it only ever produces the args a tool's schema declares, e.g.
# `query`) — build_tool_dispatch() in agent_loop.py bound these by hand
# with a dict of lambdas. Here the same binding happens via closures: this
# factory function is called once per run with the live client/collection,
# and returns @tool-decorated functions that already have them baked in.
def build_tools(gemini_client, collection):
    @tool
    def search_book(query: str) -> dict:
        """Search the indexed book for passages relevant to a specific query.
        Use this for any question about the book's plot, characters,
        themes, setting, or specific content — anything the answer to
        which depends on what the book actually says. If a question has
        multiple distinct parts (e.g. comparing two points in the story,
        or asking about two different characters), call this tool once
        per part with a focused query for that part, rather than one
        broad query covering everything.
        """
        return raw_search_book(gemini_client, collection, query)

    @tool
    def define_term(word: str) -> dict:
        """Look up a plain, general-knowledge definition of a single word
        or term. Use this only when the question needs the ordinary
        meaning of a word and that meaning does NOT depend on this book's
        content — for example, an unfamiliar vocabulary word used in the
        question itself. Do not use this for questions about how the book
        uses a word, what a character means by something, or anything
        that requires the book's text to answer — use search_book for
        those.
        """
        return raw_define_term(gemini_client, word)

    return [search_book, define_term]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
#
# RAW VERSION vs FRAMEWORK: agent_loop.py kept the conversation itself —
# the growing `contents` list of types.Content — and manually appended two
# entries per pass (the model's turn, then a turn holding tool results).
# MessagesState replaces that: it's a TypedDict with one field, `messages`,
# whose reducer (`add_messages`) appends new messages to the list for you.
# Every node below just returns the *new* messages it produced; the state
# machinery handles growing the running history.
def build_graph(api_key, tools):
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL, google_api_key=api_key,
        temperature=AGENT_TEMPERATURE,
    )
    # RAW VERSION vs FRAMEWORK: agent_loop.py put `tools=[AGENT_TOOLS]` in
    # the GenerateContentConfig passed to every generate_content() call.
    # bind_tools() is the same idea expressed as a wrapped runnable: it
    # returns a new model object that always advertises these tools,
    # without needing a config object threaded through every call site.
    llm_with_tools = llm.bind_tools(tools)

    system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)

    def agent_node(state: MessagesState) -> dict:
        # The system message is prepended fresh on every call rather than
        # stored in state — only the AIMessage this node returns gets
        # merged into state["messages"] by MessagesState's reducer, so it
        # never actually accumulates in the persisted history.
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    # RAW VERSION vs FRAMEWORK: the block in agent_loop.py that looped over
    # `function_calls`, dispatched each to search_book/define_term via the
    # hand-built `dispatch` dict, caught exceptions and turned them into
    # {"error": ...} results, and packed everything into a
    # types.Part.from_function_response — all of that is ToolNode. Give it
    # the same tool objects the model was bound to, and for every
    # AIMessage.tool_calls in the incoming state, it runs the matching
    # tool, catches exceptions itself, and appends one ToolMessage per
    # call.
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    # RAW VERSION vs FRAMEWORK: this replaces the loop's
    # `if not function_calls: return ... else: <run tools and loop again>`.
    # tools_condition inspects the last message; if it has tool_calls it
    # routes to the "tools" node, otherwise it routes to END. That
    # if-statement is now a graph edge instead of a Python branch.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    # RAW VERSION vs FRAMEWORK: this edge is the "and repeat" part of the
    # loop — after tools run, control always goes back to the agent so it
    # can see the results and decide whether it's done.
    graph.add_edge("tools", "agent")

    # A checkpointer is only here so that if we hit the recursion limit
    # below, app.get_state() has something to read back — without one,
    # invoke() raising GraphRecursionError would take the partial
    # transcript down with it, with no way to recover what had already
    # been gathered (the raw version never had this problem, since
    # `contents` was a plain Python list already in scope when the loop
    # hit MAX_ITERATIONS).
    return graph.compile(checkpointer=MemorySaver())


def _extract_tool_call_log(messages: list) -> list[dict]:
    """Rebuilds the flat {"tool", "input", "result"} log that
    output_schema.py's _format_tool_call_log() expects, by reading it back
    out of the LangGraph message history. agent_loop.py built this list
    incrementally as it went (tool_call_log.append(...) right after each
    call); here it's reconstructed after the fact from the transcript
    ToolNode already assembled, by matching each ToolMessage back to the
    AIMessage.tool_calls entry with the same id.
    """
    calls_by_id = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                calls_by_id[call["id"]] = {"tool": call["name"], "input": call["args"]}

    log = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in calls_by_id:
            entry = calls_by_id[msg.tool_call_id]
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                result = {"error": str(msg.content)}
            # Match agent_loop.py's shape: "input" is a plain dict of args
            # (e.g. {"query": "..."}), consumed by _format_tool_call_log
            # via call['tool'](call['input']).
            entry = {"tool": entry["tool"], "input": entry["input"].get("query") or entry["input"].get("word") or entry["input"], "result": result}
            log.append(entry)
    return log


def run_agent_graph(gemini_client, api_key, collection, question: str) -> dict:
    """Framework counterpart to agent_loop.run_agent_loop(). Same return
    shape ({final_text, tool_call_log, iterations_used, hit_iteration_cap})
    so the rest of the pipeline (structured-output synthesis, CLI printing)
    doesn't need to know which version produced it.

    Takes both `gemini_client` (a genai.Client, needed only because
    tools.py's search_book()/define_term() — reused unchanged — expect
    one) and `api_key` (needed separately because ChatGoogleGenerativeAI
    is a different SDK wrapper and wants a plain string, not a genai.Client
    instance).
    """
    tools = build_tools(gemini_client, collection)
    app = build_graph(api_key, tools)

    # RAW VERSION vs FRAMEWORK: MAX_ITERATIONS in agent_loop.py counted
    # full ask/respond passes, and on hitting it, deliberately made one
    # more call with `tools` stripped from the config so the model was
    # forced to answer in text using whatever had been gathered so far.
    # LangGraph's recursion_limit is the closest built-in analogue, but it
    # counts graph *steps* (each agent visit and each tools visit both
    # count), and hitting it raises GraphRecursionError rather than
    # forcing a graceful final answer — that's still on us, so the
    # except block below reimplements the raw version's fallback by hand.
    recursion_limit = MAX_ITERATIONS * 2 + 1
    run_config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        result = app.invoke({"messages": [HumanMessage(content=question)]}, config=run_config)
        messages = result["messages"]
        return {
            "final_text": messages[-1].content,
            "tool_call_log": _extract_tool_call_log(messages),
            "iterations_used": sum(1 for m in messages if isinstance(m, AIMessage)),
            "hit_iteration_cap": False,
        }
    except GraphRecursionError:
        logger.warning("Agent graph hit the recursion limit; forcing a final answer.")
        # Pull whatever transcript exists via the graph's own state
        # snapshot (recovered via the checkpointer above, keyed by the
        # same thread_id the failed invoke() used), then make one more
        # call with a plain (non-tool-bound) model, mirroring
        # agent_loop.py's forced-final-call fallback.
        state = app.get_state(config=run_config)
        messages = state.values.get("messages", []) if state.values else []

        llm = ChatGoogleGenerativeAI(
            model=AGENT_MODEL, google_api_key=api_key,
            temperature=AGENT_TEMPERATURE,
        )
        forced_system = SystemMessage(
            content=AGENT_SYSTEM_PROMPT
            + "\n\nYou have used all available tool calls. Answer now using "
            "only what you have already gathered above — you cannot call "
            "any more tools. If that isn't enough to fully answer, say what "
            "you were able to find and what's missing."
        )
        forced_response = llm.invoke([forced_system] + messages)
        return {
            "final_text": forced_response.content,
            "tool_call_log": _extract_tool_call_log(messages),
            "iterations_used": MAX_ITERATIONS,
            "hit_iteration_cap": True,
        }


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
#
# RAW VERSION vs FRAMEWORK: output_schema.py made its own generate_content()
# call with response_mime_type="application/json" and response_schema=
# AgentAnswer set directly on the config, then read `.parsed`. with_
# structured_output(AgentAnswer) is LangChain's wrapper around that same
# idea — give it a Pydantic model, get back a runnable that returns an
# already-validated instance of it. Same AgentAnswer/ToolCallRecord/
# SourceRecord models from output_schema.py are reused untouched (down to
# the additionalProperties fix that made ToolCallRecord.input a plain str
# — that Gemini API constraint doesn't go away just because LangChain is
# calling it), and so is _format_tool_call_log(), since turning the tool
# log into readable text for the prompt isn't something either framework
# piece does for you.
def synthesize_answer(api_key, question: str, draft_answer: str, tool_call_log: list[dict], hit_iteration_cap: bool) -> AgentAnswer:
    llm = ChatGoogleGenerativeAI(
        model=SYNTHESIS_MODEL, google_api_key=api_key,
        temperature=SYNTHESIS_TEMPERATURE,
    )
    structured_llm = llm.with_structured_output(AgentAnswer)

    prompt = (
        f"Question: {question}\n\n"
        f"Agent's draft answer:\n{draft_answer}\n\n"
        f"Tool call log:\n{_format_tool_call_log(tool_call_log)}\n"
    )
    if hit_iteration_cap:
        prompt += (
            "\nNote: the agent hit its safety limit on tool calls before "
            "confirming it was done. Treat the draft answer as potentially "
            "incomplete when setting confidence.\n"
        )

    return structured_llm.invoke([SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT), HumanMessage(content=prompt)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--book", "book_path", required=True, type=click.Path(), help="Path to the book's .txt file.")
@click.option("--ask", "question", required=True, help="Ask a question about the indexed book.")
@click.option("--chroma-dir", default=DEFAULT_CHROMA_DIR, show_default=True, help="Where Chroma persists its data.")
@click.option("--collection", "collection_name", default=DEFAULT_COLLECTION, show_default=True, help="Chroma collection name.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(book_path, question, chroma_dir, collection_name, verbose):
    setup_logging(verbose)

    try:
        settings = load_settings(book_path, chroma_dir, collection_name)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        client = get_client(settings.chroma_persist_dir)
        collection = get_or_create_collection(client, settings.collection_name)
    except Exception as exc:
        raise click.ClickException(f"Could not connect to Chroma: {exc}")

    gemini_client = get_gemini_client(settings.gemini_api_key)

    click.echo(f"Question: {question}\n")

    loop_result = run_agent_graph(gemini_client, settings.gemini_api_key, collection, question)
    click.echo(
        f"[graph] iterations_used={loop_result['iterations_used']} "
        f"hit_iteration_cap={loop_result['hit_iteration_cap']} "
        f"tool_calls={len(loop_result['tool_call_log'])}"
    )
    for call in loop_result["tool_call_log"]:
        click.echo(f"  - {call['tool']}({call['input']})")
    click.echo()

    answer = synthesize_answer(
        settings.gemini_api_key,
        question,
        loop_result["final_text"],
        loop_result["tool_call_log"],
        loop_result["hit_iteration_cap"],
    )
    click.echo("=== Structured answer ===")
    click.echo(json.dumps(answer.model_dump(), indent=2))


if __name__ == "__main__":
    main()
