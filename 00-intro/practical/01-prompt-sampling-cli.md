# Project 0.1 — Prompt & Sampling Playground

## Target

Create a Python CLI named `01-prompt-sampling-cli.py` that calls a Gemini model through LangChain and runs the same prompt multiple times with different sampling settings.

The CLI should let us change:

- model
- system instruction
- user prompt
- temperature
- top-p
- max output tokens
- repeat count

It should save every run to JSONL and SQLite, display readable CLI output, and provide a compare command for reviewing differences between saved generations.

## What We Learnt

- A model call is shaped by roles: system instructions guide behavior, while the user prompt asks for the task.
- Temperature changes how adventurous the next-token selection is.
- Top-p limits token selection to the smallest likely set whose probability mass reaches the chosen value.
- Lower sampling values usually make outputs more stable, but they do not guarantee identical responses across every provider and model.
- Max output tokens is a budget, not a promise that the answer will use all tokens.
- Repeating the same prompt is useful because it makes stochastic behavior visible.
- Saving prompt, settings, output, token usage, latency, and errors makes experiments comparable instead of relying on memory.
- Structured JSON output can fail when the model returns invalid JSON or values that do not match the schema.
- Pydantic validation gives us a clear pass/fail check for structured output.
- A repair retry can turn a malformed structured answer into a valid one, but robust programs must still store the original error.

## Goal

By the end, we are able to run controlled prompt experiments from the terminal and explain why outputs differ when sampling settings, system instructions, model, or token limits change.

We are also able to understand how to make an LLM API call more robust by adding persistence, formatted output, validation, error capture, and retry/repair logic.

## How To Run

Use the project virtual environment from the repository root:

```powershell
.\.venv\Scripts\python.exe 00-intro/practical/01-prompt-sampling-cli.py --dry-run
```

Dry-run mode does not call Gemini. It is useful for checking the CLI, output formatting, JSONL saving, SQLite saving, and comparison flow.

For a real Gemini call, set `GOOGLE_API_KEY` in `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Then run without `--dry-run`:

```powershell
.\.venv\Scripts\python.exe 00-intro/practical/01-prompt-sampling-cli.py
```

Run a smaller controlled experiment:

```powershell
.\.venv\Scripts\python.exe 00-intro/practical/01-prompt-sampling-cli.py --prompt "Explain top-p sampling" --temperatures 0,0.7 --top-ps 1,0.9 --max-tokens 120 --repeat 2
```

Run structured output validation and repair in dry-run mode:

```powershell
.\.venv\Scripts\python.exe 00-intro/practical/01-prompt-sampling-cli.py --structured --induce-schema-failure --dry-run
```

Compare saved outputs:

```powershell
.\.venv\Scripts\python.exe 00-intro/practical/01-prompt-sampling-cli.py compare --db runs.sqlite
```

Useful flags:

- `--prompt`: user prompt to test
- `--system`: system instruction; use multiple flags or separate instructions with `||`
- `--models`: Gemini model names to compare
- `--temperatures`: comma-separated temperature values
- `--top-ps`: comma-separated top-p values
- `--max-tokens`: comma-separated output token limits
- `--repeat`: repeat the full experiment grid
- `--structured`: request JSON and validate it with Pydantic
- `--induce-schema-failure`: intentionally trigger the repair path
- `--quiet`: show only the final summary
- `--db`: SQLite output path
- `--jsonl`: JSONL output path

## What The Output Looks Like

Each run prints a formatted block with the model settings, latency, token usage, output hash, and generated text:

```text
[1/1] model=gemini-1.5-flash temp=0.7 top_p=0.9 max=80

--------------------------------------------- Run 1/1 ----------------------------------------------
  status      : ok
  model       : gemini-1.5-flash
  sampling    : temperature=0.7, top_p=0.9, max_tokens=80
  latency     : 0ms
  token usage : prompt=12, completion=26, total=38
  output hash : 1ccd080977df26c8
  output      :
    [dry run] With temperature 0.7 and top-p 0.9, this answer should feel lively, compact,
    surprising. The model 'gemini-1.5-flash' would receive a max token budget of 80.
----------------------------------------------------------------------------------------------------
```

After all runs finish, the CLI prints a summary table:

```text
------------------------------------------- Run Summary --------------------------------------------
#    status        model              temp    top_p   max    latency   tokens   hash
---- ------------- ------------------ ------- ------- ------ --------- -------- ----------------
01   ok            gemini-1.5-flash   0.7     0.9     80     0ms       38       1ccd080977df26c8
----------------------------------------------------------------------------------------------------

Saved JSONL: C:\Personal\Practicing-Agentic-AI\runs.jsonl
Saved SQLite: C:\Personal\Practicing-Agentic-AI\runs.sqlite
```

The output hash makes it easy to spot whether two outputs are identical. The saved JSONL and SQLite files make it possible to compare runs later.

## Original Brief

Build: a CLI playground that sends the same prompt repeatedly while varying temperature, top-p, max output tokens, system instructions, and model. Save every run to JSON/SQLite and compare outputs.

Learn while building: tokens, tokenization intuition, context windows, next-token prediction, temperature/top-p, deterministic vs stochastic behavior, system/user/assistant roles, streaming, rate limits, retries, token cost, and latency.

Upgrade: add structured JSON output validated with Pydantic. Deliberately cause schema failures and implement retry/repair logic.

Done when: you can explain why two calls differ, why context length matters, when structured outputs fail, and how to make an LLM API call robust.
