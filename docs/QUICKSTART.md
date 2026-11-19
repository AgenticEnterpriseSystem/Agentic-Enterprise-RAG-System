# Quick start: run Redwood on your computer

This guide takes you from nothing to a working search page in ten steps. Copy
each command block into a terminal. After each step there is a **Check**. Don't
go on to the next step until the check passes.

The first run is **free**. It searches documents and shows sources but does not
call a paid AI model. Generated answers are an optional extra
([step 11](#11-optional-turn-on-ai-answers)).

For more background on any step, see the [full reference guide](WEB_APP.md).

## What you need

- A Mac or Linux computer with 16 GiB of RAM.
- Free disk space of at least twice the size of the Drive download plus 20 GiB.
  The data is stored once in this folder and copied once into Docker.
- [Git](https://git-scm.com/downloads) and Python 3.11 or newer.
- [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Docker
  with Compose v2.10 or later. In Docker Desktop's settings, give Docker at
  least 8 GiB of memory.
- Access to the team's
  [Google Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).
  Ask a teammate if it says you need access.

## 1. Check your tools

Start Docker first, then run:

```sh
git --version
python3 --version
docker compose version
```

**Check:** all three print a version. Python must be 3.11 or newer. If
`docker compose version` fails, Docker isn't running or is too old.

## 2. Get the code

```sh
git clone https://github.com/AgenticEnterpriseSystem/Agentic-Enterprise-RAG-System.git
cd Agentic-Enterprise-RAG-System
```

Run every remaining command from this folder.

**Check:** `ls compose.yaml` prints `compose.yaml`.

## 3. Download the document bundle

The documents and search indexes are too large for Git. Open the
[Google Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing),
download the `files_web_app` folder, and unzip it. If Drive gives you several
ZIP files, unzip all of them into the same place.

Move the files so that the layout inside this repository looks exactly like this:

```text
.cache/hybrid-artifacts/
  documents.parquet
  indexes/
    bm25/
      manifest.json
      ...other files...
    dense-bge-base/
      manifest.json
      ...other files...
```

Keep every file. Don't rename anything. Don't store the folder in iCloud,
OneDrive, or Dropbox, because they can swap files for online-only placeholders.

**Check:** this prints three paths and no errors:

```sh
ls .cache/hybrid-artifacts/documents.parquet \
   .cache/hybrid-artifacts/indexes/bm25/manifest.json \
   .cache/hybrid-artifacts/indexes/dense-bge-base/manifest.json
```

## 4. Download the embedding model

The search uses a small open-source model (not a paid AI service) to understand
questions. This downloads the exact version the indexes were built with. The
first run takes several minutes.

```sh
mkdir -p .cache/models
docker build --build-arg INSTALL_HYBRID=true --target app -t redwood-rag:models .
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/.cache/hybrid-artifacts/indexes/dense-bge-base,dst=/index,readonly" \
  --mount "type=bind,src=$PWD/.cache/models,dst=/app/.cache/models" \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  redwood-rag:models python -c \
  "from pathlib import Path; from src.retrieval.corpus_dense import EmbeddingModel; model = EmbeddingModel.for_index(Path('/index'), device='cpu', batch_size=1); print(model.identity)"
```

**Check:** the last command prints the model name, and this prints a folder:

```sh
ls -d .cache/models/models--BAAI--bge-base-en-v1.5
```

## 5. Download the dataset source

This public repository provides the original Jira fields.

```sh
git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git .cache/local-source
git -C .cache/local-source checkout --detach d36685e273713975ee20299bbf1ab64165575b3c
```

**Check:** this prints `d36685e273713975ee20299bbf1ab64165575b3c`:

```sh
git -C .cache/local-source rev-parse HEAD
```

## 6. Create your private settings file

```sh
python3 scripts/local_stack.py --env-file .env.local init \
  --artifacts-dir "$PWD/.cache/hybrid-artifacts" \
  --models-dir "$PWD/.cache/models" \
  --source-repo-dir "$PWD/.cache/local-source" \
  --operator-dir "$PWD/.cache/local-operator" \
  --project redwood --port 8000
python3 scripts/local_stack.py --env-file .env.local plan
```

This creates `.env.local` with new random database passwords. Only your user
account can read it. Never share or commit this file.

**Check:** `plan` prints a line containing `"accounts_seeded": false`.

If `init` prints `Local stack operation failed safely` and `.env.local` already
exists, you've done this step before. Keep the existing file and run only `plan`.
If it says the prepared files must already exist, go back to steps 3–5.

## 7. Build and start the app

This is the slowest step. It builds the app, copies the data into Docker, and
imports about 512,000 documents. Expect it to take a while; leave the terminal
open.

```sh
python3 scripts/local_stack.py --env-file .env.local bootstrap
```

**Check:** both commands print a short JSON reply:

```sh
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/ready
```

If `ready` fails, wait a minute and try again. If it still fails, see
[If something goes wrong](#if-something-goes-wrong).

## 8. Create your login

Replace `hira` and `Hira` with your own username and name. You'll be asked to
type a password of at least 12 characters. It isn't shown as you type.

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.auth.accounts create hira --display-name Hira --role employee
```

**Check:** it prints a `user_id`. Save it somewhere private; you'll need it only
for the optional Observability page.

## 9. Give your login access to the documents

A new login can't see any documents. These commands give it read access to every
document in the fictional test dataset. Use your own name for `--reviewer`
and any label for `--approval`. They're saved with the change as a record of who
approved it.

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.benchmark_grants prepare --login hira --reviewer "Your Name" --approval local-qa-001
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.benchmark_grants dry-run --login hira
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.benchmark_grants apply --login hira
```

`prepare` only writes review files to `.cache/local-operator/benchmark-grants/hira/`.
`dry-run` shows what would change. `apply` makes the change.

**Check:** `prepare` reports `"documents": 511958`. `apply` reports a
`changed_grants` number above zero. If you run `apply` again, it reports `0`,
which is expected.

To remove this access later:

```sh
python3 scripts/local_stack.py --env-file .env.local operator -- \
  python -m src.access.benchmark_grants revoke --login hira
```

## 10. Open the app

Go to **<http://127.0.0.1:8000>**, sign in with the login from step 8, and wait
until the page shows **Ready**. Then ask a question.

**Check:** you see source cards with document passages. In this free mode you
won't get a written AI answer; that's expected.

You're done. Everything below is optional.

## 11. Optional: turn on AI answers

This sends your question and selected document excerpts to Anthropic, and it
costs money. Agree on a spending limit with your team first.

1. Get an Anthropic API key from your team. Don't use someone else's.
2. Open `.env.local` in a text editor. Change these lines if they already exist,
   and add any that are missing. Don't create duplicates:

   ```dotenv
   RAG_MODE=live
   RAG_RETRIEVAL=hybrid
   RAG_ALLOW_PAID=true
   ANTHROPIC_API_KEY=<your key>
   RAG_MODEL=claude-haiku-4-5-20251001
   RAG_INPUT_PRICE=1
   RAG_OUTPUT_PRICE=5
   RAG_BUDGET_USD=0.50
   RAG_QUESTION_BUDGET_USD=0.10
   ```

3. Restart the app:

   ```sh
   python3 scripts/local_stack.py --env-file .env.local up
   ```

`RAG_BUDGET_USD` caps spending only until the app restarts. It isn't a total
bill limit. See [live-mode details](WEB_APP.md#enable-live-generation).

## Stop and start again later

```sh
python3 scripts/local_stack.py --env-file .env.local stop
python3 scripts/local_stack.py --env-file .env.local up
```

Your documents, login, and access are kept. Never run `docker compose down -v`:
it deletes the database.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| A step-3 check says `No such file or directory` | A file is in the wrong place. Compare your folders with the layout in step 3. A common mistake is an extra `files_web_app/` folder level. |
| `bootstrap` reports a disk-space error | Free up disk space, then run `bootstrap` again. Don't delete Docker volumes. |
| `bootstrap` stopped partway through | Run it again. If it mentions existing data, see [Troubleshooting](WEB_APP.md#validation-and-troubleshooting) before using `--adopt-existing`. |
| `ready` keeps failing | Check that steps 3–5 passed, then look at the logs: `docker compose --env-file .env.local logs --tail 100 app verify` |
| Signing in works but Search says 403 / no access | Step 9 hasn't been applied for this login. |
| `benchmark_grants prepare` says `already exists` | You've already prepared it. Run `dry-run` and `apply`, or pass a new `--directory`. |
| `benchmark_grants` says `No active human login` | Create the login first (step 8) and check the spelling. |
| Search says busy (429) | Only one search runs at a time. Wait for the current one to finish. |
| Port 8000 is already in use | In step 6, use `--port 8001` and a different `--project` name, then open that port. |

Never post logs or `.env.local` contents publicly. They can contain passwords.
