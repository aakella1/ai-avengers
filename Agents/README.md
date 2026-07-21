# Agent demo — AWS Strands Agents SDK

A single agent with nine tools. Three run locally as Python functions, five
call free public APIs (no keys), one ships with the SDK.

## Setup

From the folder that contains `RAG/` and `Agents/`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-..." > Agents/.env
cd Agents
```

The key is read from `Agents/.env` automatically — nothing to export. That
file is not included in this bundle; create it yourself.

## Run

```bash
python agent.py                            # interactive chat (terminal)
python agent.py --demo                     # scripted questions
python agent.py --ask "What is 17 * 23?"   # one-shot

streamlit run chat_app.py                  # browser chat UI
```

## Files

| File | Contents |
|------|----------|
| `agent.py` | Model setup, tool list, system prompt, the run modes |
| `tools_local.py` | `calculate`, `tell_joke`, `roll_dice` — plain Python |
| `tools_api.py` | `web_search`, `get_weather`, `get_exchange_rate`, `get_public_holidays`, `get_cat_fact` — real HTTP calls, no API keys needed |
| `tools_rag.py` | `search_documents` — queries the FAISS index built by the RAG demo |
| `chat_app.py` | Streamlit chat UI over the same agent |

| Tool | Backed by |
|------|-----------|
| `calculate` | local — parses the expression with `ast`, never `eval` |
| `tell_joke` | local — a hardcoded list |
| `roll_dice` | local — `random` |
| `search_documents` | the RAG pipeline's FAISS index in `../RAG/data/indexes/` |
| `web_search` | DuckDuckGo via the `ddgs` package |
| `get_weather` | Open-Meteo (geocoding + forecast) |
| `get_exchange_rate` | open.er-api.com |
| `get_public_holidays` | date.nager.at |
| `get_cat_fact` | catfact.ninja |
| `current_time` | built into `strands_tools` |

## Things worth demonstrating in class

**The docstring is the interface.** `@tool` builds the schema the model sees
from the function name, the type hints, and the docstring. Nothing else. Break
a docstring on purpose — change `get_weather`'s to `"""Does a thing."""` — and
watch the model stop calling it. Tool selection is a prompt-engineering
problem, not a code problem.

**There is no dispatch logic anywhere.** Grep `agent.py` for `if`. Nothing
decides which tool runs; the model does, every turn. That is the difference
between an agent and a workflow.

**Multi-step reasoning.** `"What's the weather in Hyderabad, and how does it
compare to London?"` requires two calls to the same tool with different
arguments, then a comparison the model makes itself. `"Roll 3 twenty-sided
dice, then tell me a joke"` chains two unrelated tools in one turn.

**Search closes the knowledge-cutoff gap.** Ask something the model cannot
know from training alone:

```
Who is the current CEO of OpenAI, and what is the weather where they are
headquartered?
```

It searches, reads the company's location out of the results, then feeds that
into `get_weather` — two different tools, the second one's argument produced by
the first one's output. Nothing in the code connects them; the model does.

## The chat UI

```bash
streamlit run chat_app.py       # opens http://localhost:8501
```

`chat_app.py` imports the same `build_agent()` and the same `TOOLS` as
`agent.py`. It swaps the terminal for a browser and adds nothing to the agent
itself — worth stating plainly in class, because the interface is where people
tend to think the intelligence lives.

What the UI gives you that the terminal does not:

- **Tool calls render live**, with the arguments the model chose. Ask a
  two-part question and watch two tools appear in the panel before any text
  does. The panel stays attached to each message, so you can scroll back
  through a session and see exactly what fired where.
- **A sidebar index switcher.** Flip `search_documents` between the OpenAI and
  local indexes mid-conversation and re-ask the same question.
- **Memory is visible.** Ask "what's the capital of France?", then "what's the
  weather there?" — the second question has no city in it. The agent holds its
  own message history, so `there` resolves to Paris and the weather tool gets
  called with the right argument.
- **Clear conversation** resets to a fresh agent, for a clean run between
  cohorts.

Two implementation details worth pointing at, since both are easy to get wrong:

`build_agent(callback_handler=None)` silences the SDK's terminal printing so
the UI can render the event stream itself. It has to be passed at construction
— assigning `agent.callback_handler = None` afterwards raises `TypeError` on
the next call.

Tool events repeat as the model streams its arguments in, so the UI keys on
`toolUseId` to update one entry rather than appending the same call a dozen
times.

## Agentic RAG — `search_documents`

`search_documents` wraps the RAG demo's retriever. It needs an index first:

```bash
cd ../RAG && python 01_extract.py && python 02_chunk.py \
          && python 03_embed.py  && python 04_load_index.py
```

It follows `latest` by default. Pin a specific one with an env var — useful for
demoing the OpenAI and local indexes back to back through the same agent:

```bash
RAG_INDEX_ID=idx_openai_20260718_153200 python agent.py
```

**This is the payoff if you teach RAG and agents in the same session.** Same
index, same embeddings, same FAISS call as `RAG/06_generate.py`. The only
difference is who controls the retrieval — and it changes the answers.

Ask both versions the identical question:

```bash
cd ../RAG   && python 06_generate.py --index latest \
               --query "Compare the political careers of Chiranjeevi and Balakrishna"
cd ../Agents && python agent.py \
               --ask "Compare the political careers of Chiranjeevi and Balakrishna."
```

The pipeline answers: *"A meaningful comparison is not possible... the context
contains no information about any political career for Balakrishna."* That is
wrong — he has been the MLA for Hindupur since 2014 — but it is a truthful
report of what retrieval handed over. One top-6 search across a mixed corpus
returned five Chiranjeevi chunks and one Balakrishna chunk, and the Balakrishna
chunk was the wrong one. The model never saw the fact.

The agent runs **two** searches, one per person, and produces the full
comparison: party founded vs. party inherited, national office vs. state seat,
one term vs. three consecutive wins.

Nobody wrote "if the question mentions two people, search twice." The model
worked out that one search would not cover the question. That is the entire
difference between a workflow and an agent, visible in one query.

**Retrieval as one tool among many.** `"How old is Chiranjeevi today, and
what's the weather in his birthplace?"` runs four tools in sequence:
`search_documents` (get the birth date and place from the PDF) → `current_time`
→ `calculate` (age) → `get_weather` (that place). The output of each becomes
the argument to the next. When the weather lookup failed on the small village,
it said so and offered the nearest larger town rather than inventing a
temperature.

**The tool description is the retrieval trigger.** The docstring of
`search_documents` lists what the corpus actually contains — two named actors,
their films, awards, politics. That is what tells the model when to reach for
it. Vague it out to `"""Search documents."""` and the model stops using it,
because it has no idea what is in there. In fixed-pipeline RAG this problem
does not exist: you always retrieve, so nothing has to decide. Give the model
the choice and describing the corpus becomes load-bearing.

**Failure is visible, not silent.** Try `RAG_INDEX_ID=nope python agent.py` and
ask a biography question. The tool returns "index not found" as text, and the
agent reports a configuration problem instead of falling back to whatever it
half-remembers about a Telugu film star.

Note `web_search` scrapes DuckDuckGo, so it can rate-limit if you hammer it.
The tool returns a plain "rate limited, try again" message rather than raising,
so a class demo degrades gracefully instead of crashing. If you need something
sturdier for a big room, swap in Tavily or Brave — both have free tiers and an
API key, and the tool body is about ten lines to rewrite.

**Tools fail, and that is normal.** Every remote tool returns an error *string*
rather than raising. Try `"weather in Zzzznotacity"` or `"public holidays in
India"` (that API has no Indian data — it answers HTTP 204). The model reads
the error and tells the user, instead of the process crashing or the model
inventing a plausible answer.

**The sandbox matters.** Ask it to `calculate __import__("os").system("ls")`.
The calculator walks the parsed expression tree and allows arithmetic only, so
it refuses. Had it used `eval()`, the model would have had a shell. Tool inputs
are model output — treat them as untrusted.

## Switching to Amazon Bedrock

The demo uses the Anthropic API directly because it needs one env var. Strands
defaults to Bedrock instead if you want AWS credentials to do the work — swap
the model in `build_agent()`:

```python
from strands.models import BedrockModel

model = BedrockModel(model_id="global.anthropic.claude-opus-4-8",
                     region_name="us-west-2")
```

That needs `aws configure` plus model access enabled in the Bedrock console.
Everything else in the file stays identical — which is itself a decent point
about where the provider boundary sits.
