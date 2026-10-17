# AAR Gateway

**Use less of your Claude plan in Claude Code, with the same login, tools and skills.**

AAR Gateway runs on your computer, between Claude Code and Anthropic. For every
message you type it picks the cheapest Claude model that should answer it about
as well as the one you chose — Haiku or Sonnet for the easy ones, your model
for the hard ones — and sends the request on your own login. Subagents
(Claude Code's helpers for searching and side tasks) get a model by the kind
of work they do. A local dashboard shows what it saved.

In our own use so far: **30–50% less plan usage** than always using the model
we picked, and **~97% less** on search subagents moved to Haiku. On 198 graded
test prompts it keeps about 94% of Opus's answer quality at 39% less usage; a
perfect router could save 59% at no loss, and closing that gap is the work in
progress. Your own numbers show up on the dashboard as you work.

## What you need

* [Claude Code](https://claude.com/claude-code), installed and logged in
  (a Claude subscription or an API key both work)
* macOS or Linux (Windows is supported in the code but not yet tested)

## Install

```bash
# 1. uv, a fast Python installer that brings its own Python (skip if you have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. the gateway (release v0.1.0, a beta), in its own isolated environment
uv tool install "agentic-llm-router[gateway] @ git+https://github.com/wkukka1/agentic-llm-router@v0.1.0"

# 3. first-time setup: installs the router and writes ~/.aar-gateway/config.yaml
aar-gateway setup
```

Setup asks one question: **use the gateway in every Claude Code session?**
Answer `y` and plain `claude` goes through the gateway from then on. The
gateway starts by itself (at login on macOS and Linux, and whenever a
session starts if it isn't running), so you never run a command for it.
Press Enter (the default is no) and nothing about Claude Code changes: start
it with `aar-gateway code` whenever you want routing. You can change your
mind any time with `aar-gateway enable` or `aar-gateway disable`.

The router comes with the package: a small trained network (0.4 MB). To use
one you built yourself, `aar-gateway setup --router <file>`
([docs/RUNBOOK.md](docs/RUNBOOK.md)). `aar-gateway setup` ends by running
`aar-gateway doctor`, which checks everything and says how to fix anything
missing.

To update to a newer release, run step 2 again with its tag and `--force`
(`uv tool install --force "agentic-llm-router[gateway] @ git+https://github.com/wkukka1/agentic-llm-router@v0.2.0"`),
then `aar-gateway restart`. To remove it, run
`aar-gateway disable` first (if you enabled it), then `uv tool uninstall
agentic-llm-router`; your settings stay in `~/.aar-gateway`.

## Use it

If you enabled it at setup, just run `claude` as always. Otherwise:

```bash
aar-gateway code            # Claude Code, through the gateway (starts it if needed)
aar-gateway dashboard       # your savings, live
```

Use Claude Code exactly as before. Anything after `code` goes to Claude Code:
`aar-gateway code -c` continues your last session.

The **status line** at the bottom of Claude Code shows which model actually
answered and how much less plan usage this session used. (Asking Claude which
model it is won't tell you: it's always told it is the model you picked.)

## Choose how much to save

On the dashboard, under **Cost preference**:

* **Best quality** — stays on your model unless a cheaper one is almost as good
* **Balanced** (default) — cheaper models when they're predicted close in quality
* **Save more** — prefers cheaper models more often

**Advanced settings** let you pin every request to one model, turn routing
off, let hard questions move *up* to a stronger model, or change which model
each kind of subagent uses. Changes apply from your next message.

## Optional: a free local model

Not part of the standard setup. If you'd like a model that runs on your own
computer (free, private, much weaker than Claude), add one through
[Ollama](https://ollama.com):

```bash
# 1. install Ollama (once)
brew install ollama && brew services start ollama       # macOS
curl -fsSL https://ollama.com/install.sh | sh           # Linux
# Windows: the installer from https://ollama.com/download

# 2. add a model to the gateway (downloads it the first time)
aar-gateway local add llama3.1
```

The router never sends anything to it on its own. Use it by pinning it on
the dashboard (Advanced settings → Pin a model) or with
`aar-gateway code --model llama3.1:latest`. `aar-gateway local` shows what's
installed; `aar-gateway local remove llama3.1` takes it out again. Pick a
model that handles tool calls well — `llama3.1` does; some coding models
don't.

## Commands

| Command | What it does |
|---|---|
| `aar-gateway code` | run Claude Code through the gateway |
| `aar-gateway enable` / `disable` | use the gateway in every `claude` session, or stop |
| `aar-gateway dashboard` | open the dashboard |
| `aar-gateway status` | is it running, is the router loaded, how much has it saved |
| `aar-gateway start` / `stop` / `restart` | run or stop the gateway in the background |
| `aar-gateway logs -f` | follow the gateway's log |
| `aar-gateway doctor` | check your setup |
| `aar-gateway setup` | install the router and write a config |
| `aar-gateway service install` | start the gateway automatically at login |
| `aar-gateway local add <model>` | optional: add a local model through Ollama |
| `aar-gateway share [on/off/show/delete]` | optional: anonymous usage sharing (off by default) |

## Sharing usage data (optional, off by default)

Setup asks whether you'd like to share anonymous usage data to help improve
routing; only a `y` turns it on. When on, each message you type adds one
record: which model was asked for and which answered, tokens, cost, speed,
whether you corrected the answer, a rounded prompt length, and a compressed
embedding of the message (64 small numbers). Never your prompts, error
messages, file paths, or session ids, and no time finer than the day.

For now nothing is sent anywhere: records stay in `~/.aar-gateway/share/`
on your computer. `aar-gateway share show` prints them, `aar-gateway share
off` stops collecting, `aar-gateway share delete` removes them.

## What it never does

* **Never moves you to a pricier model** unless you turn that on.
* **Never leaves you stuck:** if a rerouted request fails, it's resent on the
  model you asked for. If the router can't load, every request goes through
  as you asked.
* **Never changes your request** beyond the model name — your tools, skills,
  MCP servers and settings reach Claude unchanged.
* **Never sends your data anywhere new:** Claude requests go to Anthropic on
  your own login, as before. The gateway keeps a local log of models, token
  counts and costs — never your prompts or credentials. It only accepts
  connections from programs on your own computer, not from web pages.

## If something looks wrong

* **If the gateway stops answering** while you use Claude Code with it
  enabled, sessions switch to Anthropic directly, with one line saying so,
  and try to start the gateway again. Routing comes back on with the next
  message once it answers. Nothing needs doing; `aar-gateway doctor` says
  why it stopped.
* **Run `aar-gateway doctor`.** It checks Claude Code, your config, the
  router and the port, and says how to fix each problem.
* **"busy" in the dashboard's Status column** means Anthropic asked Claude
  Code to slow down (your plan's rate limit). Claude Code retries by itself.
* **A red "+N%"** on one request means switching model cost more that
  once: the new model had to read the whole conversation in fresh. It should
  be rare. A "—" means the request went to the model you asked for. Judge
  savings by the session totals, which include these.
* **Settings:** `aar-gateway config` shows which config file is in use
  (default `~/.aar-gateway/config.yaml`). Every option is explained in
  [configs/gateway.example.yaml](configs/gateway.example.yaml).

## More

* [docs/gateway.md](docs/gateway.md) — how routing decides, the cost model,
  subagents, using other providers (OpenAI-compatible APIs, Ollama)
* [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — the research code: data pipeline,
  router training, evaluation
* [docs/RUNBOOK.md](docs/RUNBOOK.md) — every pipeline, step by step
