import os
from typing import Optional

import click
from dotenv import dotenv_values, load_dotenv
from google import genai
from pydantic import BaseModel


class ParsedLine(BaseModel):
    """Raw parsed entry — holds the real value. Never leaves this process."""
    key: str
    value: Optional[str] = None


class RedactedEntry(BaseModel):
    """What's actually safe to send to the API — masked value only."""
    key: str
    masked_value: str
    is_secret: bool

class Issue(BaseModel):
    key: str
    problem: str
    severity: str
    category: str


class RiskAssessment(BaseModel):
    overall_risk: str
    issues: list[Issue]
    summary: str


# Step 4: secret detection heuristics — key-name pattern matching.
# Simple substring check against common naming conventions. Not exhaustive
# (no entropy check on the value itself), but a reasonable first pass.
SECRET_KEY_PATTERNS = (
    "key",
    "secret",
    "token",
    "password",
    "pwd",
    "credential",
    "auth",
    "private",
)


def looks_like_secret(key: str) -> bool:
    lowered = key.lower()
    return any(pattern in lowered for pattern in SECRET_KEY_PATTERNS)


# Step 5: redaction/masking. Short values are fully masked (nothing useful
# to reveal); longer values keep first/last 2 chars so a human reviewing
# the tool's output can still recognize *which* secret is being discussed.
def mask_value(value: Optional[str]) -> str:
    if value is None:
        return "<not set>"  # e.g. a bare `KEY` line with no `=` at all
    if value == "":
        return "<empty>"  # e.g. `KEY=` with nothing after the `=`
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def parse_env_file(file_path: str) -> list[ParsedLine]:
    config = dotenv_values(file_path)
    return [ParsedLine(key=key, value=value) for key, value in config.items()]


def build_redacted_entries(parsed_lines: list[ParsedLine]) -> list[RedactedEntry]:
    return [
        RedactedEntry(
            key=line.key,
            masked_value=mask_value(line.value),
            is_secret=looks_like_secret(line.key),
        )
        for line in parsed_lines
    ]


# Step 7: build the actual text block that gets embedded in a prompt.
# Only ever touches RedactedEntry objects — masked_value, never .value.
def format_entries_for_prompt(entries: list[RedactedEntry]) -> str:
    lines = []
    for entry in entries:
        flag = " [FLAGGED AS POSSIBLE SECRET]" if entry.is_secret else ""
        lines.append(f"{entry.key} = {entry.masked_value}{flag}")
    return "\n".join(lines)


# Step 6: the tool's OWN credentials (GEMINI_API_KEY), loaded from the
# project's .env — separate from --file-path, which is the file being audited.
def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException("GEMINI_API_KEY is not set in your .env file.")
    return genai.Client(api_key=api_key)


MODEL_NAME = "gemini-3.1-flash-lite"


def run_structured_assessment(client: genai.Client, prompt_payload: str) -> RiskAssessment:
    """Step 9: non-streamed call, output constrained to the RiskAssessment schema."""
    instructions = (
        "You are a security reviewer auditing a local .env-style configuration file. "
        "You are given a list of redacted config entries — key names and partially "
        "masked values only; you never see full secret values. For each entry, decide "
        "whether it represents a security or configuration risk (e.g. an unset value, "
        "a key name suggesting a credential, a weak-looking value, or anything else "
        "worth flagging).\n\n"
        "Config entries:\n"
        f"{prompt_payload}\n\n"
        "Return an overall_risk rating, a list of issues (one per problematic entry, "
        "each with the entry's key, a short problem description, a severity, and a "
        "category), and a short summary."
    )

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=instructions,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RiskAssessment.model_json_schema(),
        },
    )
    return RiskAssessment.model_validate_json(interaction.output_text)


def run_streamed_walkthrough(client: genai.Client, prompt_payload: str) -> None:
    """Step 10: streamed call for a plain-English walkthrough of the same entries."""
    instructions = (
        "You are a security reviewer auditing a local .env-style configuration file. "
        "You are given a list of redacted config entries (key names and partially "
        "masked values only). Write a short, plain-English walkthrough of what looks "
        "concerning and why, as if explaining it to the developer who wrote this config.\n\n"
        "Config entries:\n"
        f"{prompt_payload}"
    )

    stream = client.interactions.create(
        model=MODEL_NAME,
        input=instructions,
        stream=True,
    )
    for event in stream:
        if event.event_type == "step.delta" and event.delta.type == "text":
            click.echo(event.delta.text, nl=False)
    click.echo()


@click.command()
@click.option('--file-path', required=True, help='Path to the file to be checked.')
def main(file_path):
    if not os.path.isfile(file_path):
        raise click.ClickException(f"The file '{file_path}' does not exist.")

    parsed_lines = parse_env_file(file_path)
    redacted_entries = build_redacted_entries(parsed_lines)
    prompt_payload = format_entries_for_prompt(redacted_entries)

    click.echo("Redacted entries sent to Gemini:")
    click.echo(prompt_payload)
    click.echo()

    client = get_gemini_client()

    try:
        click.echo("=== Plain-English walkthrough ===")
        run_streamed_walkthrough(client, prompt_payload)
        click.echo()

        click.echo("=== Structured assessment ===")
        assessment = run_structured_assessment(client, prompt_payload)
    except Exception as exc:
        # Broad on purpose for now — narrow this once you've seen which
        # exceptions the SDK actually raises (network errors, bad key,
        # rate limits, etc. likely each have their own type).
        raise click.ClickException(f"Gemini API call failed: {exc}")

    click.echo(f"Overall risk: {assessment.overall_risk}")
    click.echo(f"Summary: {assessment.summary}")
    if assessment.issues:
        click.echo("Issues:")
        for issue in assessment.issues:
            click.echo(
                f"  - [{issue.severity}] {issue.key} ({issue.category}): {issue.problem}"
            )
    else:
        click.echo("No issues found.")


if __name__ == "__main__":
    main()