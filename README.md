# Aptic

by Deuce Labs

**Use less of your Claude plan in Claude Code, with the same login, tools and skills.**

Aptic runs on your computer, beside Claude Code. For every message you type it
picks the cheapest Claude model that should answer about as well as the one you
chose: Haiku or Sonnet for the easy ones, your model for the hard ones. The
request still goes to Anthropic on your own login. Subagents (Claude Code's
helpers for searching and side tasks) get a model by the kind of work they do.

In our own use so far: **30–50% less plan usage** than always using the model
we picked, and **~97% less** on search subagents moved to Haiku. On 198 graded
test prompts it keeps about 94% of Opus's answer quality at 39% less usage; a
perfect router could save 59% at no loss, and closing that gap is the work in
progress. Your own numbers show up in Claude Code as you work.

## Quick start

You need:

* [Claude Code](https://claude.com/claude-code) 2.1.287 or later, logged in
  (a Claude subscription or an API key both work)
* [uv](https://docs.astral.sh/uv/), which the plugin uses to run the router.
  macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
  Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.

Then, in Claude Code:

```
/plugin install aptic --marketplace wkukka1/agentic-llm-router
```

Start a new session. The plugin starts the router in the background (the
first time, uv downloads it in a few seconds) and opens the Aptic pane. From
then on each turn runs on the model the router picks. Nothing else on your
machine changes: no settings are edited and no proxy sits between Claude Code
and Anthropic. If the router stops, Claude Code keeps the model you chose.

## Using it in Claude Code

Use Claude Code as you always do. Aptic shows its work in four places.

### The Aptic pane

Opens by itself at the start of a session; `/aptic-pane` brings it back.

* **This session**: how much plan usage you saved against always using the
  model you asked for, and why (for example "2 of 5 turns routed to Sonnet 5.5").
* **Model mix**: a bar of how many requests ran on your model and how many
  were routed down.
* **Requests, Cache hit, Turns** for this session.
* **All sessions**: total requests routed and how much less plan usage they took.
* **Turns**: one row per message you typed, with its time, the model that
  answered, the decision ("Kept · router agreed", "Routed down from Opus 5.5"),
  its request count and cache hit. Subagents are listed under their turn.
  **Newest first** switches the order.
* **View all sessions** opens the dashboard in your browser.

The chevron in the header collapses the pane to the two savings figures and a
requests-per-turn chart. The gear shows the pane's settings. The pane
remembers whether it is collapsed and how turns are sorted.

### The line under each answer

After every answer: the model that answered, its cache use and its request
count, for example `Aptic: sonnet 5.5 · cache 98% hit (120k read, 2k written) · 4 requests`.

### The status line

Every minute: plan usage saved across all sessions, total requests and this
session's cache hit. (Asking Claude which model it is won't tell you: it is
always told it is the model you picked.)

### Commands

| In Claude Code | What it does |
|---|---|
| `/aptic` | print the savings now; starts the router if it isn't running |
| `/aptic-pane` | open the Aptic pane |
| `/config` | the plugin's "Route in Claude Code" setting turns routing on or off |
| `/plugin` | update or remove the plugin |

## The dashboard

The router serves a dashboard at http://127.0.0.1:8787/dashboard, with every
session, every request and what each one cost. Open it from the pane's
**View all sessions**, or with `aar-gateway dashboard` if you installed the
command-line tool (see [Proxy mode](#proxy-mode-without-the-plugin)).

### Choose how much to save

On the dashboard, under **Cost preference**:

* **Best quality**: stays on your model unless a cheaper one is almost as good
* **Balanced** (default): cheaper models when they're predicted close in quality
* **Save more**: prefers cheaper models more often, using the savings router
  (`manifold-nirt-claude-judged`), which scores the Claude models directly

**Advanced settings** let you pin every request to one model, turn routing
off, let hard questions move *up* to a stronger model, or change which model
each kind of subagent uses. Changes apply from your next message.

## Update or remove

* **Update**: `/plugin`, then Marketplaces, `deuce-labs`, Update; then
  Installed, `aptic`, Update. Or from a terminal:
  ```
  claude plugin marketplace update deuce-labs
  claude plugin update aptic@deuce-labs
  ```
  Start a new session to load the new version.
* **Pause routing** but keep the pane: turn off "Route in Claude Code" in `/config`.
* **Remove**: `/plugin`, then Installed, `aptic`, Uninstall. Claude Code then
  uses the model you choose for every turn. `~/.aar-gateway/` keeps your
  config and history until you delete it.

## Proxy mode (without the plugin)

The router can also sit between Claude Code and Anthropic as a local proxy:
for Claude Code versions without plugins, or to route every session on the
machine.

```bash
uv tool install "aar-gateway @ git+https://github.com/wkukka1/agentic-llm-router@main#subdirectory=aar-gateway"
aar-gateway setup
```

Setup asks whether to use the gateway in every Claude Code session (default
no) and whether to add the plugin too. Then:

```bash
aar-gateway code            # Claude Code through the gateway (starts it if needed)
aar-gateway code -c         # anything after `code` goes to Claude Code: -c continues your last session
aar-gateway dashboard       # open the dashboard
```

If you chose to use it in every session, just run `claude`. `aar-gateway
disable` undoes that.

| Command | What it does |
|---|---|
| `aar-gateway code` | run Claude Code through the gateway |
| `aar-gateway enable` / `disable` | use the gateway in every `claude` session, or stop |
| `aar-gateway dashboard` | open the dashboard |
| `aar-gateway status` | is it running, is the router loaded, how much has it saved |
| `aar-gateway start` / `stop` / `restart` | run or stop the gateway in the background |
| `aar-gateway logs -f` | follow the gateway's log |
| `aar-gateway doctor` | check your setup and say how to fix anything missing |
| `aar-gateway setup` | install the router and write a config |
| `aar-gateway service install` | start the gateway automatically at login |
| `aar-gateway config` | show which config file is in use |
| `aar-gateway local add <model>` | optional: add a local model through Ollama |
| `aar-gateway history ...` | look at, export or delete the history kept on this computer |
| `aar-gateway share ...` | optional: anonymous usage sharing (off by default) |

## Optional: a free local model

If you'd like a model that runs on your own computer (free, private, much
weaker than Claude), add one through [Ollama](https://ollama.com):

```bash
# 1. install Ollama (once)
brew install ollama && brew services start ollama       # macOS
curl -fsSL https://ollama.com/install.sh | sh           # Linux
# Windows: the installer from https://ollama.com/download

# 2. add a model to the gateway (downloads it the first time)
aar-gateway local add llama3.1
```

The router never sends anything to it on its own. Use it by pinning it on the
dashboard (Advanced settings, then Pin a model) or with
`aar-gateway code --model llama3.1:latest`. `aar-gateway local` shows what's
installed; `aar-gateway local remove llama3.1` takes it out again. Pick a
model that handles tool calls well: `llama3.1` does; some coding models don't.

## Your data

**What stays on your computer.** The router keeps a log of models, token
counts and costs (never your credentials) and, in `~/.aar-gateway/history/`,
each message you type with the model it got and how it went, so routing can
be personalized to your own work. Nothing in the history is ever uploaded.
Nothing is deleted automatically (old data is compressed).
`aar-gateway history status|show|export <file>` look at it,
`aar-gateway history off` stops keeping messages, `aar-gateway history delete`
removes everything kept. The plugin itself stores no prompt text.

**Sharing (optional, off by default).** Setup asks whether you'd like to share
anonymous usage data to help improve routing; only a `y` turns it on. When on,
each message you type adds one record: which model was asked for and which
answered, tokens, cost, speed, whether you corrected the answer, a rounded
prompt length, and a compressed embedding of the message (64 small numbers).
Never your prompts, error messages, file paths or session ids, and no time
finer than the day. Records wait in `~/.aar-gateway/share/` until a collection
endpoint is set (`aar-gateway share endpoint <https address>`); then they go
once a day to the project's collection server, filed under a random id that
isn't tied to you. `aar-gateway share show` prints what's waiting,
`share status` what's been sent, `share off` stops it, `share delete` removes
what hasn't been sent. (The server is in [`src/ingest`](src/ingest/README.md).)

## What it never does

* **Never moves you to a pricier model** unless you turn that on.
* **Never leaves you stuck:** if a routed request fails, it's resent on the
  model you asked for, and that model isn't routed to again in the session.
  If the router can't load, every request goes through as you asked.
* **Never changes your request** beyond the model name: your tools, skills,
  MCP servers and settings reach Claude unchanged.
* **Never sends your data anywhere new:** Claude requests go to Anthropic on
  your own login, as before. The router only accepts connections from
  programs on your own computer, not from web pages.

## If something looks wrong

* **The pane says "Aptic's router isn't running".** Run `/aptic`: it starts
  the router and says what went wrong if it can't. Without uv, it says how to
  install it.
* **The pane didn't change after an update.** Plugins load when a session
  starts: open a new one.
* **View all sessions does nothing.** Open http://127.0.0.1:8787/dashboard in
  your browser yourself; if that fails too, the router isn't running (see above).
* **In proxy mode, if the gateway stops answering**, sessions switch to
  Anthropic directly, with one line saying so, and try to start it again.
  `aar-gateway doctor` checks Claude Code, your config, the router and the
  port, and says how to fix each problem.
* **"busy" in the dashboard's Status column** means Anthropic asked Claude
  Code to slow down (your plan's rate limit). Claude Code retries by itself.
* **A red "+N%" on one request** means switching model cost more that once:
  the new model had to read the whole conversation in fresh. It should be
  rare. Judge savings by the session totals, which include these.
* **Settings**: every option in `~/.aar-gateway/config.yaml` is explained in
  [gateway.example.yaml](aar-gateway/src/aar_gateway/gateway.example.yaml).

## More

* [aar-gateway/docs/gateway.md](aar-gateway/docs/gateway.md): how routing
  decides, the cost model, subagents, other providers (OpenAI-compatible APIs, Ollama)
* [mods/aptic](mods/aptic/README.md): the Claude Code plugin
* [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): the research code (data
  pipeline, router training, evaluation)
* [docs/RUNBOOK.md](docs/RUNBOOK.md): every pipeline, step by step

## License

Apache License 2.0. See [LICENSE](LICENSE).
