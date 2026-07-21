# RAG demo — six scripts, one per stage

> The agent demo can query this same index as a tool — see
> `../Agents/README.md` → *Agentic RAG*. Running the two side by side on the
> same question is the sharpest way to show what an agent adds.

Each stage writes its output to disk, so you can run them one at a time, stop
and inspect what came out, and re-run a single stage after changing a setting.

## Setup (once)

From the folder that contains `RAG/` and `Agents/`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd RAG
```

Keys are read from a `.env` file automatically — nothing to export. Create
them yourself; they are not included in this bundle.

| Key | Put it in | Needed for |
|-----|-----------|------------|
| `OPENAI_API_KEY` | `RAG/.env` | stage 3 with `--provider openai` |
| `ANTHROPIC_API_KEY` | `Agents/.env` (or `RAG/.env`) | stage 6 |

```bash
echo "OPENAI_API_KEY=sk-..." > RAG/.env
echo "ANTHROPIC_API_KEY=sk-ant-..." > Agents/.env
```

Stages 1, 2, 4 and 5 need no key at all. With `--provider local`, stage 3
doesn't either — only stage 6 does.

## The pipeline

| # | Script | Class | What it does | Output |
|---|--------|-------|--------------|--------|
| 1 | `01_extract.py` | `PDFExtractor` | PDF → plain text, per page | `data/extracted/*.json` |
| 2 | `02_chunk.py` | `Chunker` | Text → overlapping word windows | `data/chunks.json` |
| 3 | `03_embed.py` | `Vectorizer` | Chunks → vectors (provider of your choice) | `data/embeddings.npy` |
| 4 | `04_load_index.py` | `FaissIndexLoader` | Vectors → a **new** FAISS index | `data/indexes/<index_id>/` |
| 5 | `05_retrieve.py` | `SemanticRetriever` | Question → most similar chunks | prints results |
| 6 | `06_generate.py` | `RAGGenerator` | Chunks + question → answer from Claude | prints answer |

```bash
python 01_extract.py
python 02_chunk.py
python 03_embed.py                    # OpenAI by default
python 04_load_index.py
```

Stage 4 prints the id of the index it just created, e.g. `idx_openai_20260718_151654`.

## Choosing an embedding model

Stage 3 takes `--provider`. Everything downstream adapts automatically.

| | `--provider openai` (default) | `--provider local` |
|---|---|---|
| Model | `text-embedding-3-small` | `all-MiniLM-L6-v2` |
| Dimensions | 1536 | 384 |
| Runs | OpenAI's servers | your CPU |
| Needs | `OPENAI_API_KEY` | nothing |
| Cost | ~$0.02 / million tokens (this corpus: a fraction of a cent) | free |
| Index size | 630 KB | 158 KB |

```bash
python 03_embed.py --provider local  && python 04_load_index.py
python 03_embed.py --provider openai && python 04_load_index.py
python 03_embed.py --provider openai --model text-embedding-3-large
```

Each run of stage 3 overwrites `embeddings.npy`, so the order is always
**embed, then immediately index**. Once an index exists it is frozen and
self-contained — later runs of stage 3 cannot corrupt it.

**You cannot mix providers.** A query embedded by one model against an index
built by another produces vectors in unrelated spaces and returns noise. The
provider and model are written into the index metadata and read back at query
time, so stages 5 and 6 always rebuild the right embedder on their own. Worth
showing the class: `data/indexes/<id>/meta.json` is what makes that safe.

## Querying

Every run of stage 4 creates a **new** index and leaves the old ones in place.
Stages 5 and 6 take `--index` to choose one — either an explicit id, or the
word `latest`.

```bash
python 04_load_index.py --list                       # see what you have

python 05_retrieve.py --index latest --query "What awards has he received?"
python 05_retrieve.py --index idx_local_20260718_151707 --query "political career" -k 3

python 06_generate.py --index latest --query "Compare their political careers"
python 06_generate.py --index latest --query "..." --show-prompt
```

`--list` shows which embedding model built each index:

```
INDEX ID                         VECTORS   DIM  EMBEDDING MODEL          CREATED
idx_local_20260718_151707            105   384  all-MiniLM-L6-v2         2026-07-18T15:17:07
idx_openai_20260718_151654           105  1536  text-embedding-3-small   2026-07-18T15:16:54
```

`--show-prompt` prints the exact text sent to the model. This is the single most
useful thing to project on screen — it makes clear that RAG is just string
assembly plus a normal API call, not a special model feature.

## Things worth demonstrating in class

**Semantic ≠ keyword.** Search for `"awards and honours received"`. The top hit
is a passage about the Padma Vibhushan that contains none of those words.

**Bigger embedding model ≠ better retrieval.** Build both indexes and run the
same five questions through each, scoring how many genuinely relevant chunks
land in the top 5:

| Question | local (384d) | openai (1536d) |
|---|---|---|
| who did he marry and how many children | **2/2** | 1/2 |
| what awards has he won | 1/14 | 1/14 |
| which political party did he start | 3/9 | **4/9** |
| when and where was he born | 1/3 | **2/3** |
| what was his film debut | **2/9** | 1/9 |
| **total** | **9** | **9** |

A dead heat on this corpus, with each winning different questions. That is a
more useful lesson than either model winning: on short factual chunks from
clean Wikipedia text, the local model is already good enough, and the 4× larger
vectors buy nothing here. The honest rule is to measure on *your* data before
paying per token — which is exactly what the two-provider setup lets students
do in five minutes.

(This also shows why chunking dominates. Every "awards" question scores 1/14
regardless of model, because the award mentions are scattered across fourteen
chunks and no top-5 can contain them all. No embedding model fixes that —
better chunking or a higher `-k` does.)

**Chunk size changes the results.** Build two indexes and query both:

```bash
python 02_chunk.py --chunk-size 80 --overlap 20 && python 03_embed.py
python 04_load_index.py --name small-chunks

python 02_chunk.py --chunk-size 400 --overlap 80 && python 03_embed.py
python 04_load_index.py --name big-chunks

python 05_retrieve.py --index small-chunks --query "who is his father?"
python 05_retrieve.py --index big-chunks   --query "who is his father?"
```

Small chunks are precise but lose context; big chunks carry context but dilute
the vector. There is no universally correct value — that is the point.

**Retrieval failure is answer failure.** Ask something the PDFs do not cover
(`"what is his favourite food?"`). Stage 5 still returns five chunks, ranked by
similarity, because it always returns the *k* nearest — "nearest" does not mean
"relevant". Stage 6 should then say it cannot answer from the context. That gap
between stage 5 and stage 6 is where most production RAG bugs live.

## Notes

- `data/indexes/<id>/` is self-contained: vectors, chunk text, and the metadata
  recording which embedding model and chunk size built it. Deleting the rest of
  `data/` will not break querying.
- The index is `IndexFlatIP` — brute-force exact search. It is the right choice
  up to roughly a million vectors, and it keeps the demo honest: no approximate
  search behaviour to explain away.
- Vectors are normalized at stage 3, so an inner-product search is a cosine
  similarity search. Scores run 0 → 1, higher is better.
