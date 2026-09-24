# UKULELE — Knowledge Universe

*Unsupervised Knowledge Universe for Leftovers and Loose Ends.*

Most experiments that don't work never get published, so lab after lab repeats the same dead ends. This project gives negative and inconclusive results a home. It is a shared, searchable, citable record of what has already been tried.

It has three parts:

| Part | What it does | Where |
|---|---|---|
| **Intake web app** | A password-protected web page. A scientist describes a result in plain language. An AI (Claude by default) turns the description into a structured entry. It records gaps honestly rather than guessing, and it shows the scientist everything before saving. | `intake_app.py` |
| **Database** | One YAML file per result in `entries/`. Each file is checked against a schema, reviewed through GitHub pull requests, and given a permanent ID on merge. | `entries/`, `schema/`, `tools/`, `.github/workflows/` |
| **Search** | Finds related entries using a free embedding model from Hugging Face. It warns you when results that *look* similar can't actually be compared. | `retrieval/` |

This guide assumes no prior experience. Follow the parts in order.

- [Before you start](#before-you-start)
- [Part 1 — Run it on your computer](#part-1--run-it-on-your-computer)
- [Part 2 — Put the web app online](#part-2--put-the-web-app-online)
- [Part 3 — Turn submissions into database entries](#part-3--turn-submissions-into-database-entries)
- [Part 4 — Search the database (Hugging Face)](#part-4--search-the-database-hugging-face)
- [Optional — Use a Hugging Face model instead of Claude](#optional--use-a-hugging-face-model-instead-of-claude)
- [Settings reference](#settings-reference)
- [Troubleshooting](#troubleshooting)

---

## Before you start

You need these accounts:

- **GitHub**: to get the code, and to host the database.
- **Anthropic**: to get the API key the intake AI uses. Sign up at [platform.claude.com](https://platform.claude.com), then go to **Settings → API keys → Create key**. The key starts with `sk-ant-`. It is shown only once, so paste it somewhere safe. Also set a monthly spending cap under **Limits**.
- **Render** (to host the web app for testers) *or* **Hugging Face** (a free alternative with limits). [Part 2](#part-2--put-the-web-app-online) helps you choose.

> [!WARNING]
> Treat your API key like a password. Never put it in a file in this repo, and never commit it to git. Enter it only in your terminal or in your hosting service's settings page.

You also need these programs on your computer:

- **Git**. macOS: run `git --version` in Terminal and accept the prompt to install it. Windows: install from [git-scm.com](https://git-scm.com). Linux: `sudo apt install git` (or your distribution's equivalent).
- **Miniforge**, which provides the `conda` command and installs Python and every library for you. See step 1 below.

---

## Part 1 — Run it on your computer

Commands are for **Terminal** on macOS or Linux. On **Windows**, open **Miniforge Prompt** from the Start menu after step 1, and write `set NAME=value` wherever this guide says `export NAME=value`.

### 1. Install Miniforge

macOS / Linux:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Press Enter to page through the license. Answer `yes` to each question, then **close and reopen Terminal**.

Windows: download `Miniforge3-Windows-x86_64.exe` from the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest) and run it with the default options.

If you already have Anaconda or Miniconda, skip this step. It works too.

### 2. Download the code

```bash
git clone https://github.com/QVEU/UKULELE.git
cd UKULELE
```

If the repository is private, GitHub will ask you to sign in. Run every later command from inside this `UKULELE` folder.

### 3. Create the environment

```bash
conda env create -f environment.yml
conda activate knowledge-universe
```

The first command downloads about 400 MB and takes up about 2 GB on disk, so it may take several minutes. Run `conda activate knowledge-universe` again in every new Terminal window. Your prompt shows `(knowledge-universe)` when it's active.

### 4. Check that everything works

```bash
pytest
```

The last line should say `151 passed` (the number grows as tests are added). The tests need no internet and no API key.

### 5. Try the web app on your own machine

```bash
export KU_APP_PASSWORD=pick-any-password
export KU_LLM=anthropic
export ANTHROPIC_API_KEY=sk-ant-...your-key...
uvicorn intake_app:app --port 8000
```

Open <http://localhost:8000> in your browser, type the password you picked, and describe a real negative result. Try **Review & confirm**, then **Submit entry**. The entry is saved in the `entries/` folder. Press **Ctrl+C** in Terminal to stop the server.

> [!NOTE]
> Delete the test entries you create here before committing anything, unless you mean to add them to the database. While they sit in `entries/`, `pytest` also checks them, and it fails on incomplete ones.

There is also a command-line version of the same conversation: `python intake/agent.py`. It uses the same `export` settings as above. Each line you type is one message, and the agent replies after each line. Press **Ctrl+D** (Windows: **Ctrl+Z**, then Enter) to move to review, and type `approve` to save.

---

## Part 2 — Put the web app online

When the app runs on a server, testers only need a web link and the password. Your API key stays on the server and is never sent to their browsers.

Pick one host:

| | **Render** (recommended) | **Hugging Face Spaces** |
|---|---|---|
| Cost | A few dollars a month: a paid instance plus a small disk. Check [render.com/pricing](https://render.com/pricing). | Free |
| Are submitted entries kept? | **Yes**, on a persistent disk | **No.** They're erased whenever the Space restarts or sleeps, unless you pay for storage. |
| Can you download the saved files? | Yes, from the **Shell** tab | Not easily on the free tier |
| Who can find the page? | Only people you give the link to | Public Spaces are listed and searchable. Private Spaces require every tester to have a Hugging Face account that you add. |

`TESTER_README.md` tells testers "anything you submitted is safe". That promise only holds on Render with a disk attached. Use Spaces just for a quick demo.

### Option A — Render (recommended)

1. **Sign up** at [render.com](https://render.com) using **Sign in with GitHub**.
2. In the dashboard, click **New → Web Service**.
3. **Connect the repository** `QVEU/UKULELE`. If it isn't listed, follow Render's link to give it access to that repository on GitHub, then come back.
4. **Fill in the form:**

   | Field | Value |
   |---|---|
   | Name | anything, e.g. `ku-intake`. Your address becomes `https://ku-intake.onrender.com`. |
   | Branch | `main` |
   | Language | `Python 3` |
   | Build Command | `pip install fastapi uvicorn pyyaml anthropic openai` |
   | Start Command | `uvicorn intake_app:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | the cheapest **paid** type. The free type can't have a disk, so it loses every entry when it goes to sleep, which happens after 15 minutes without visitors. |

   The build command deliberately doesn't use `requirements.txt`. That file includes the search tools, which pull in PyTorch (several gigabytes), and the web app doesn't need them.

5. **Environment Variables:** add these four, using **Add Environment Variable** for each:

   | Key | Value |
   |---|---|
   | `KU_APP_PASSWORD` | a long password you'll share with testers |
   | `KU_LLM` | `anthropic` |
   | `ANTHROPIC_API_KEY` | your `sk-ant-...` key |
   | `KU_ENTRIES_DIR` | `/var/data/entries` |

6. **Add a disk.** This is in the **Advanced** section of the form, or on the service's **Disks** page after creation. Use Mount Path `/var/data` and Size `1 GB`, and give it any name. `KU_ENTRIES_DIR` above points inside this disk, and that is why entries survive restarts and redeploys.
7. Click **Create Web Service**. Watch the log until you see `Application startup complete`. This takes a couple of minutes the first time.
8. **Test it:** open your `https://<name>.onrender.com` address, enter the password, and submit a test entry. Then open the service's **Shell** tab and run `ls /var/data/entries`. Your file should be listed.
9. **Invite testers.** Send each person the link, the password (by a separate message), and the file [`TESTER_README.md`](TESTER_README.md).

Things to know while it runs:

- Every push to `main` redeploys the app automatically. That includes merging entries in Part 3, and the bot's ID commit that follows. A redeploy restarts the server, which ends any conversation in progress, so testers who are mid-conversation will need to reload. Submitted entries are safe on the disk. To avoid interrupting testers, merge when nobody is using the app, or turn off **Auto-Deploy** in the service's settings and use **Manual Deploy** when you change the app.
- Keep the start command as written. Don't add `--workers`, because conversations are held in the server's memory, and extra workers can't see each other's conversations.
- To change the password or key, edit the value on the service's **Environment** page. Render restarts the service with the new value.
- The app limits each message to 8,000 characters and each conversation to 40 messages, which caps what any one tester can spend. Your Anthropic monthly cap is the real safety net.

### Option B — Hugging Face Spaces (free demo)

1. Sign up at [huggingface.co](https://huggingface.co), then open [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose any **Space name**. For **SDK**, pick **Docker**, then the **Blank** template. For **Hardware**, pick **CPU basic** (free). Choose **Public** or **Private** (see the table above), then click **Create Space**.
3. In the Space's **Files** tab, click **Add file → Upload files** and upload `intake_app.py` from this repo.
4. Click **Add file → Create a new file**, name it `Dockerfile`, paste the following, and commit. If the template already created a `Dockerfile`, open that one, click **edit**, and replace everything in it.

   ```dockerfile
   FROM python:3.11-slim
   RUN pip install --no-cache-dir fastapi uvicorn pyyaml anthropic openai
   RUN useradd -m -u 1000 user
   USER user
   WORKDIR /home/user/app
   COPY --chown=user intake_app.py .
   CMD ["python", "-m", "uvicorn", "intake_app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

   The auto-created `README.md` in the Space already says `sdk: docker`. Port 7860 is Spaces' default, so no other file needs changing.

5. Open **Settings → Variables and secrets**. Add the **secrets** `KU_APP_PASSWORD` and `ANTHROPIC_API_KEY`, and the **variable** `KU_LLM` = `anthropic`.
6. The Space rebuilds on its own. When the status turns **Running**, the app appears on the Space's page.

Limits of this option: free Spaces go to sleep after 48 hours without visitors, and wake when someone opens the page. Every sleep, restart, or file change **erases submitted entries**. To keep entries, ask each tester to copy the text shown after **Submit entry** and send it to you. The alternative is to buy persistent storage for the Space and add the variable `KU_ENTRIES_DIR` = `/data/entries`.

---

## Part 3 — Turn submissions into database entries

Submissions are drafts. They become permanent database entries only after you review them and merge them through GitHub.

1. **Get the file.** On Render, open the **Shell** tab, run `ls /var/data/entries` to list the files, then run `cat /var/data/entries/<file>.yaml` and copy the text. On Hugging Face, use the text a tester sent you.
2. **Save it on your computer** in the `entries/` folder, under a descriptive name ending in `.yaml`, for example `entries/ptbp2-poliovirus-ires.yaml`. Start a branch for your changes first:

   ```bash
   git checkout main && git pull
   git checkout -b add-new-entries
   ```

3. **Review it.** The comment at the top lists the fields the scientist left unknown. Check with them about anything that looks wrong. `evidence_layer` must be filled in with one of `binding`, `functional`, `phenotypic`, `computational`, or `observational`. Leave `id: ku-pending-00000000` exactly as it is.
4. **Check it against the schema:**

   ```bash
   python tools/validate.py entries/ptbp2-poliovirus-ires.yaml
   ```

   Fix whatever it reports and rerun until it prints `OK`.

5. **Propose it:**

   ```bash
   git add entries/ptbp2-poliovirus-ires.yaml
   git commit -m "Add entry: PTBP2 has no effect on poliovirus IRES translation"
   git push -u origin add-new-entries
   ```

   GitHub prints a link to open a pull request. Open it. Two checks, **Validate entries** and **Tests**, run automatically. This pull request is also where others can review the entry.

6. **Merge** once both checks pass. Within a minute or two, a bot called `ku-bot` commits a permanent ID (like `ku-molecularvir-95a03d44`) to the entry on `main`. Run `git checkout main && git pull` to get it.

---

## Part 4 — Search the database (Hugging Face)

Search needs an **embedding model**, which is a program that turns text into numbers so that similar texts can be found. This project uses [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a small, free, public model from the Hugging Face Hub. It runs on your own computer. It needs **no account, no token, and no payment**. The web app doesn't use it at all.

### 1. Build the search index

From the `UKULELE` folder, with the environment active:

```bash
python retrieval/build_index.py
```

The first run downloads the model (about 90 MB) from Hugging Face automatically. When it finishes, you should see `Wrote FAISS index.` and `Indexed N entries with all-MiniLM-L6-v2.` The index is saved in `retrieval/index/`. Git ignores that folder, so it never gets committed.

**Rebuild the index whenever entries are added or changed.** The index doesn't update itself.

### 2. Ask questions

```bash
python retrieval/query.py "does PTBP2 affect poliovirus IRES translation?"
python retrieval/query.py "PTBP2 poliovirus IRES" --layer functional
python retrieval/query.py "does PTBP2 bind the poliovirus IRES?" --layer binding
```

`--layer` shows only one kind of evidence. `--claim` filters by claim type, and `--k 10` returns more results. The `[RAFT ASSESSMENT]` line tells you whether the results can be treated as agreeing:

- `genuine_raft`: the entries use comparable conditions.
- `weak_raft`: the conditions only partly match.
- `topical_only`: the entries are on the same topic but were never shown to be comparable.
- `mixed_layers`: the entries answer different kinds of question.
- `no_match`: nothing matched your filters. That is a real answer, not an error. The third query above gives it while the only entry is the functional PTBP2 result, because a functional null must not be passed off as an answer to a binding question.

### 3. Get a written answer (uses the LLM)

```bash
cd retrieval
python -c "from synthesize import synthesize; print(synthesize('does PTBP2 affect poliovirus IRES translation?'))"
cd ..
```

This sends only the matching entries to the LLM, which must cite an entry ID for every claim. It uses the same `KU_LLM` / API key settings as the intake. With `export KU_LLM=echo`, it prints the exact prompt instead of calling any model, so you can check what would be sent for free.

### Hugging Face details (only if something goes wrong)

- **Where the model is stored:** in `~/.cache/huggingface` (Windows: `C:\Users\<you>\.cache\huggingface`). To store it somewhere else, set `export HF_HOME=/path/to/folder` before building.
- **Download errors or rate limits:** make a free account, create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then run `hf auth login` and paste the token. Alternatively, set `export HF_TOKEN=hf_...`.
- **No internet, or a network that blocks Hugging Face:** build the index once on a machine that can download the model, then set `export HF_HUB_OFFLINE=1` to use the saved copy. To prepare a second machine, copy the `~/.cache/huggingface` folder across.
- **Using a different model:** change `MODEL_NAME` near the top of `retrieval/build_index.py`, then rebuild the index. `query.py` reads the model name from the index, so the two always match.

---

## Optional — Use a Hugging Face model instead of Claude

Hugging Face **Inference Providers** can run open models, such as `openai/gpt-oss-120b`, behind an OpenAI-compatible address. Both the web app and the command-line tools can use it without code changes. Billing then goes through your Hugging Face account instead of Anthropic.

1. Create a **fine-grained** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with the permission **Make calls to Inference Providers**.
2. Use these settings instead of the Anthropic ones, either with `export` locally or on your host's environment page:

   | Key | Value |
   |---|---|
   | `KU_LLM` | `openai` |
   | `OPENAI_BASE_URL` | `https://router.huggingface.co/v1` |
   | `OPENAI_API_KEY` | your `hf_...` token |
   | `KU_LLM_MODEL` | a chat model ID from the Hub, e.g. `openai/gpt-oss-120b` |

   Hugging Face picks the fastest provider automatically. To force one, add a suffix such as `openai/gpt-oss-120b:groq`.

> [!IMPORTANT]
> The intake is only worth running if it **never records things the scientist didn't say**. Before switching testers to a different model, test it yourself: give it vague, incomplete notes and check whether it invents controls, numbers, or conditions (see `TESTER_README.md`). Smaller models are more likely to fill gaps. They also sometimes return badly formatted replies. The app safely ignores a badly formatted reply, but the conversation won't move forward.

The same three `OPENAI_*`/`KU_LLM_MODEL` settings also work with OpenAI itself (leave `OPENAI_BASE_URL` unset), or with a model on your own computer through [Ollama](https://ollama.com). For Ollama, set `OPENAI_BASE_URL=http://localhost:11434/v1`, and set `OPENAI_API_KEY` to any text, such as `ollama`: the client refuses to start without one.

---

## Settings reference

All settings are environment variables.

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `KU_APP_PASSWORD` | web app | *(required)* | The password testers type. The app refuses to start without it. |
| `KU_LLM` | web app, `intake/agent.py`, `synthesize` | web app: `anthropic`; others: `echo` | `anthropic` or `openai`. `echo` (command line only) prints prompts instead of calling a model. |
| `ANTHROPIC_API_KEY` | when `KU_LLM=anthropic` | none | Your Anthropic key |
| `OPENAI_API_KEY` | when `KU_LLM=openai` | none | Your OpenAI key, or a Hugging Face token |
| `OPENAI_BASE_URL` | when `KU_LLM=openai` | OpenAI | Another OpenAI-compatible server (Hugging Face, Ollama, …) |
| `KU_LLM_MODEL` | all LLM calls | `claude-haiku-4-5-20251001` / `gpt-4o-mini` | Which model to use |
| `KU_ENTRIES_DIR` | web app | `entries` | Folder where submissions are saved |
| `HF_TOKEN`, `HF_HOME`, `HF_HUB_OFFLINE` | search | none | See [Hugging Face details](#hugging-face-details-only-if-something-goes-wrong) |

---

## Troubleshooting

| You see | What it means / what to do |
|---|---|
| `RuntimeError: Set KU_APP_PASSWORD before starting.` | Set `KU_APP_PASSWORD` locally, or on your host's environment page. |
| `Bad or missing password` in the page | The password is wrong. Reload the page and type it again. |
| `Unknown session — start a new one.` | The server restarted (after a redeploy, a crash, or sleeping), which ends conversations in progress. Reload the page. Submitted entries are safe as long as they were saved to a disk. |
| An error after clicking **Send** | The server couldn't reach the LLM. Check the API key, your credit balance, and `KU_LLM`. The details are in the host's **Logs** tab, or in your Terminal. |
| Entries disappear on Render | Either no disk is attached, or `KU_ENTRIES_DIR` doesn't start with the disk's mount path (`/var/data`). |
| `uvicorn: command not found` on Render | The build command didn't run. Copy it again exactly from Part 2. |
| `conda: command not found` | Close and reopen Terminal after installing Miniforge. On Windows, use **Miniforge Prompt**. |
| `validate.py` says a date `is not of type 'string'` | Put quotes around the date, e.g. `date: "2026-09-24"`. |
| `validate.py` says `None is not of type 'string'` at `evidence_layer` | Fill in the kind of evidence (see Part 3, step 3). |
| Can't download the model / `couldn't connect to 'https://huggingface.co'` | Your network blocks Hugging Face. See the offline instructions in Part 4. |
| `python intake/agent.py` never records anything | The command-line intake needs a real model. Set `KU_LLM=anthropic` (or `openai`), not `echo`. |

For developers: run `pytest` before every push. The same tests run automatically on each pull request.
