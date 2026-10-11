# Gateway: route Claude Code through the router

`router.gateway` is a local process that sits between Claude Code and the
model providers, in the same place [claude-code-router](https://github.com/musistudio/claude-code-router)
sits. Claude Code keeps working exactly as before — same login, same tools —
and at every new message the trained router decides which model answers that
turn. A dashboard shows what that cost against two baselines.

```
Claude Code ──► gateway (127.0.0.1:8787) ──► router picks a model for this turn
  (your login)        │                          │
                      │                          ├─ Claude model → Anthropic, on your own login
                      │                          └─ other model  → its API / Ollama, tools translated
                      └─► request log ──► /dashboard
```

## Quick start

```bash
pip install -e ".[gateway,research]"          # research = the trained router
cp configs/gateway.example.yaml configs/gateway.yaml
export AAR_GATEWAY_CONFIG=$PWD/configs/gateway.yaml

aar-gateway code          # starts the gateway if needed, then runs Claude Code through it
aar-gateway dashboard     # opens http://127.0.0.1:8787/dashboard
```

`aar-gateway code` passes every further argument to `claude`
(`aar-gateway code -p "..."`, `aar-gateway code -c`). It is the same as
running `ENABLE_TOOL_SEARCH=true ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude`
with the gateway already up (`aar-gateway serve`). It also sets Claude Code's
status line to show **which model actually answered** and this session's
cost against no router, e.g.
`aar router · claude-sonnet-5-5 (routed) · $0.21 · saved $0.26 (55%)`.
That line is the reliable answer to "which model is this". Asking the model
isn't: Claude Code's system prompt tells it that it is the model *you*
picked, so a rerouted Sonnet still says "Opus". If you already have a
status line it is left alone (or set `AAR_GATEWAY_STATUSLINE=0`).

**Don't drop `ENABLE_TOOL_SEARCH=true`.** For any base URL that isn't
Anthropic's, Claude Code stops loading MCP tools on demand and sends every
tool's schema with every request. With a lot of MCP servers that means
~245k tokens per request instead of ~40k: several times the cost, and
compaction on every turn.

**If `aar-gateway` says `No module named 'router'`:** your virtualenv is in
an iCloud-synced folder (`~/Documents`, `~/Desktop`). macOS keeps marking its
`.pth` files hidden, and Python 3.13+ skips hidden `.pth` files. Either put
the venv outside iCloud (`python3 -m venv ~/.venvs/aar`) or
`export PYTHONPATH=$PWD/src`. Run from the repo root: the router loads its
trained run from `data/processed/nirt_runs/` (or set `interceptor.router.runs_dir`).

## Check it works

```bash
curl -s localhost:8787/health          # router on? graded prompts + calibration offsets
aar-gateway code -p "Read src/router/gateway/graded.py and explain min_prompts in two sentences."
tail -3 ~/.aar-gateway/requests.jsonl  # requested vs routed model, why, cost vs no-router
tail -1 ~/.aar-gateway/turns.jsonl     # the turn was captured (capture_prompts: true)
aar-gateway-grade --n 2                # grade pending turns on the real tiers
curl -s localhost:8787/health          # graded_prompts went up; offsets recomputed, no restart
```

The status line in Claude Code and the dashboard's "Routed" column show which
model answered. Asking the model can't tell you: Claude Code's system prompt
names the model you picked.

## How one request flows

[`router/gateway/interceptor.py`](../src/router/gateway/interceptor.py):

1. **Is this a new turn?** A request whose last user message carries new
   human text starts a turn. Claude Code's `<system-reminder>` text, and the
   hook output it appends as a trailing `system` message, are not the human's
   words and are ignored. A request carrying only tool results continues the
   current turn and **stays on the turn's model**, so a tool loop never changes
   model halfway through.
2. **Which model?** On a new turn of the agent conversation, the picker
   decides (next section). Claude Code's side requests (title generation and
   similar, which carry no tools) are left alone: they cost almost nothing
   and some models reject their settings.
3. **Send it.** A Claude model: forwarded to Anthropic close to untouched —
   the client's own `Authorization`, every tool, thinking and beta settings,
   real streaming. Only `model` may change. Any other model: translated (see
   *Other models*).
4. **Fail open.** If the routed model errors, the request is retried on the
   model Claude Code asked for. A 400 is remembered per client version, so a
   model that client can't use is not tried again. Anthropic rejects models
   newer than the Claude Code version supports, for example.
5. **Log it** (see *Dashboard*).

The gateway never stores a credential for the Claude path. It forwards the
one Claude Code sent, and never writes headers, credentials or prompt text
to its log.

## How the router decides

[`router/gateway/routing.py`](../src/router/gateway/routing.py) composes the
existing pieces: a trained `RouterModel` scores the prompt, and
`MarginRoutingPolicy` takes **the cheapest candidate within the user's
quality margin of the best**.

**Candidates** are the models in `interceptor.models`, minus any that:

* are pricier than the model the user picked (the router only ever saves);
* can't fit the prompt (`context_window`). Claude Code's prompt is ~200k
  tokens, so Haiku 4.5's 200k window usually rules it out;
* reject the request's settings (`rejects`);
* were rejected for this client before;
* are non-Claude models that can't take the request without losing something.

**Quality** comes from a router fitted on **LLMRouterBench**. That
benchmark scores 13 models from 2025 (Claude Sonnet 4, GPT-5, Gemini 2.5
Pro/Flash, DeepSeek V3/R1, Qwen3, GLM-4.6, Kimi K2, Intern-S1) on 12,336
prompts across 10 benchmarks, including SWE-bench and τ²-bench (tool use).
See *Training data and router choice* below.

Each configured model names a `proxy`: the pool model whose predicted
quality stands in for it. No public dataset scores today's Claude tiers on
the same prompts, so:

* **Opus** reads **GPT-5**, the pool's strongest;
* **Sonnet** reads **Claude Sonnet 4**, the same family one generation back;
* **Haiku** reads **Gemini 2.5 Flash**, the pool's fast tier.

The router therefore answers "how much does this prompt need the top tier
versus the mid or fast tier", measured on 2025 models. That's much closer
than the 2023 pool the first version used, but it is still not a measurement
of Claude 5.x itself.

**Cost** is estimated without predicting tokens per model. The gateway keeps
a running average of the tokens each request in a conversation actually used:

* new uncached input;
* new cache *writes*: what the conversation added since the last request;
* output, thinking included.

It prices that same measured profile on each candidate model. A new
conversation starts from the average over all agent traffic. Per model, two
things then differ:

* **Price.** Claude Code's prompt is mostly cache reads, so the decisive
  difference is usually the price of the per-request cache writes (2x input:
  $8/MTok on Opus 5.5, $4 on Sonnet 5.5). The read prices can be equal: $0.20
  on both. A reads-only estimate made a warm Opus look as cheap as Sonnet, so
  nothing switched at any margin.
* **Cache state.** Caches are per model. A model not holding the conversation
  rewrites it once at the write price.

A model counts as warm if it served this conversation, or cached the same
system prompt and tools for any session in the last hour (Claude Code's
prefix is shared across sessions). The estimate is the average per request
over the next `horizon_requests` (default 20; one turn's tool loop is often
5–20 requests).

Without this the router would "save" money by switching and pay more in
cache rewrites than it saved.

## Cost preference

The margin is the user's cost preference: how much predicted quality a turn
may give up for a cheaper model. The presets are calibrated on
LLMRouterBench's held-out split (1,738 prompts, 13 models), against always
using GPT-5 (quality 0.596, $28.30 per 1k prompts):

| Preset | Margin | Quality | Cost / 1k prompts |
|---|---|---|---|
| Best quality | 0.01 | 0.615 | $24.40 (−14%) |
| Balanced (default) | 0.05 | 0.596 | $17.25 (−39%) |
| Save more | 0.08 | 0.580 | $12.06 (−57%) |

`0` (always the top prediction) is allowed, but on the benchmark it cost
*more* than always using GPT-5 for no measurable gain.

Three ways to set it:

* **From the dashboard.** Saved in `~/.aar-gateway/preferences.json` and kept
  across restarts.
* **With `POST /api/preferences`**, sending `{"preset": "save_more"}` or
  `{"quality_margin": 0.1}`.
* **For one Claude Code session only**, with a header:

```bash
ANTHROPIC_CUSTOM_HEADERS="x-aar-quality-margin: 0.1" aar-gateway code
```

## Dashboard

`/dashboard` reads `GET /stats`. One JSON line per request goes to
`AAR_GATEWAY_LOG` (default `~/.aar-gateway/requests.jsonl`), recording:

* the asked-for and routed model, and why;
* predicted quality;
* tokens: input, cache read, cache write, output;
* latency, status, and whether it fell back;
* three costs:
    * **cost**: what this request cost at API prices;
  * **if no router**: the same request on the model Claude Code asked for;
  * **if always strongest**: the same request on `interceptor.strongest`.

The dashboard and the status line show the result as **plan usage**: percent
less (or more) than without the router, and than always using the strongest
model. On a subscription the dollars are never billed. Scaling by API price
gives the relative cost of the same work on each model, which is the best
available proxy for how much of the plan it used. The dollars are still
shown, in small print.

The two baselines reuse this request's token counts, adjusted for what that
model would have had cached if it had served all the traffic. On a Claude
subscription nothing is billed per token, so read the dollars as plan usage.
Prices are in [`router/gateway/pricing.py`](../src/router/gateway/pricing.py)
(Anthropic API rates, including each model's own cache-read rate) and can be
overridden per model.

### Switching routing yourself

The dashboard's Routing card:

* **Auto** (default): the router picks per turn, within your cost preference.
  * **Let hard turns go up a tier** (off by default): the router may pick a
    model pricier than the one you asked for, e.g. a Haiku session sends a
    hard question to Opus. Off, routing only ever moves you to a cheaper model.
  * **Subagent tiers**: change any role's model (or "session's model");
    "default" undoes it.
* **Pin a model**: every agent turn and subagent goes to the model you pick,
  even a pricier one. A request the pinned model can't take keeps the model
  Claude Code asked for.
* **Off**: every request keeps the model Claude Code asked for.

Choices apply from the next turn and survive restarts
(`~/.aar-gateway/preferences.json`). Each change is also appended to
`~/.aar-gateway/overrides.jsonl`: if a role keeps being moved up, its default
tier is wrong.

## Other models (API keys, Ollama)

A model with `lane: openai` is any OpenAI-compatible endpoint: OpenAI,
Gemini's compatible endpoint, OpenRouter, Groq, vLLM, Ollama.
[`router/gateway/translate.py`](../src/router/gateway/translate.py) converts the
Anthropic request and converts the answer back, so the agent's tool loop runs
unchanged. It carries tools, tool calls, tool results, images,
mid-conversation system messages and streaming.

The rule is **lossless or not at all**. A request that wouldn't survive the
trip is never routed there:

* more tools than `max_tools`;
* server tools;
* documents;
* images inside tool results.

Thinking blocks and cache markers are dropped, because no other provider can
use them.

Keys come from the environment variable named in `api_key_env`; a missing one
fails at startup. A model on this lane needs no Anthropic login: name it as
the model (`aar-gateway code --model llama3.1:latest`) or let the router pick it.

What this means for Claude Code, checked live:

* With `ENABLE_TOOL_SEARCH=true`, Claude Code sends ~12 ordinary tools and
  loads the rest itself through its `ToolSearch` tool, so its requests *are*
  translatable. The prompt is ~40k tokens: it fits API models (DeepSeek, GPT,
  Gemini), not a 32k local model, which the context-window filter enforces.
* Without tool search, the same request carries 180+ tool schemas (~245k
  tokens) and fits nothing but Claude.
* Local models under Ollama: `llama3.1` returns real tool calls (a
  tool → result → answer loop through the gateway works); `qwen2.5-coder:14b`
  writes tool calls as plain JSON text, which the agent can't act on.

## Training data and router choice

**Searched for:** routing data covering current Claude models (Hugging Face
Hub, October 2026).

* **No public dataset scores Opus, Sonnet and Haiku 4.x/5.x on the same
  prompts.** That data only comes from running them, or from graded gateway
  traffic.
* **LLMRouterBench** (`NPULH/LLMRouterBench`, ACL 2026) is the closest broad
  pool: 13 models from 2025, including Claude Sonnet 4, with real per-token
  costs. The repo already had its pipeline (`configs/llmrouterbench.yaml`);
  it had never been run. Running it found:
  * the archive layout had changed (`results/bench-release/` now);
  * 10 records carry negative token counts (now unknown, not dropped);
  * one excluded 14 MB file takes ~15 minutes to `json.loads`, so files
    outside the setting are now skipped by path before being read.
* **SWE-Router** (`SWE-Router/swebench-verified-*`) has agentic SWE-bench
  Verified runs, with pass/fail and cost, for Claude Opus 4.7, GPT-5.2,
  GPT-5 mini/nano, Gemini 3/3.1/2.5 Pro and DeepSeek V3.2/V4 Flash. Not
  merged into the training pool: its agent harness differs from
  LLMRouterBench's single-shot SWE-bench, so the two mix up "better model"
  with "better harness". On its own it shows the headroom on agentic coding:

  | | Solved | Cost (500 tasks) |
  |---|---|---|
  | Always Opus 4.7 | 78% | $158 |
  | Perfect router | 88% | $67 |

  DeepSeek V4 Flash, 22x cheaper per task, also solves 76% of the tasks
  Opus 4.7 solves.

**Results** on LLMRouterBench's held-out split, same decision rule as the
gateway:

| Policy | Trained NIRT | kNN (k=25) |
|---|---|---|
| margin 0.01 | 0.610 @ $29.06 | **0.615 @ $24.40** |
| margin 0.05 | 0.575 @ $11.37 | **0.596 @ $17.25** |
| margin 0.08 | 0.533 @ $0.93 (collapses to the cheapest) | **0.580 @ $12.06** |

Always GPT-5 scores 0.596 @ $28.30; the oracle 0.817 @ $3.96.

The trained NIRT (AUC 0.78, ECE 0.044) does not beat nearest neighbours, so
the gateway uses **kNN** (`interceptor.router.kind: knn`). This agrees with
the repo's earlier finding that the decision rule matters more than the
predictor. kNN loads in ~7 s and scores a prompt in ~17 ms.

To reproduce:

```bash
python scripts/data/download_llmrouterbench.py --config configs/llmrouterbench.yaml
python scripts/data/build_response_matrix.py   --config configs/llmrouterbench.yaml
python scripts/data/build_splits.py             --config configs/llmrouterbench.yaml
OMP_NUM_THREADS=1 python -m training.phase0 --config configs/llmrouterbench.yaml \
    --sources llmrouterbench --embeddings --taxonomy --query-bank
OMP_NUM_THREADS=1 python scripts/margin_sweep.py --config configs/llmrouterbench.yaml --router knn --k 25
```

On Apple silicon add `ROUTER_EMBED_DEVICE=mps` for the BERT step. Keep the
6.5 GB of results out of iCloud-synced folders (`~/Documents`, `~/Desktop`);
syncing them stalls every file operation in the project.

## Grading your own traffic

The benchmark stand-ins are a starting point. The real fix is routing data
from your own Claude Code work, graded on the actual Claude tiers.

1. **Capture.** Set `interceptor.capture_prompts: true`. For every new turn
   of an agent conversation, the gateway appends your message (only your
   words, never the conversation or tool results) and the folder Claude Code
   ran in to `~/.aar-gateway/turns.jsonl`.
2. **Grade.** Run `aar-gateway-grade --n 20`
   ([`evaluation/gateway/grade.py`](../src/evaluation/gateway/grade.py)).
   * **Replay:** each captured message is replayed on Opus 5.5, Sonnet 5.5
     and Haiku 5.5 through the `claude` CLI, on your own login, in the
     original folder, read-only (`--permission-mode plan`). The calls go
     straight to Anthropic, not through the gateway.
   * **Judge:** one call per message, with the answers anonymised and
     shuffled, scores each 0–10: "would a careful senior engineer accept
     this as correct and complete?"
   * **Output:** one row per (message, model) in `~/.aar-gateway/graded.jsonl`.
   * **Cost:** each graded message is 4 calls on your plan. `--n` bounds it,
     and nothing is graded twice. `aar-gateway-grade --summary` shows the
     per-tier results.
3. **Route on it.** Once `router.min_graded_prompts` (20) messages are
   graded, a new message with at least 3 similar graded ones (cosine ≥ 0.5,
   top 10) is routed on their grades for the real tiers: no stand-ins.
   * Otherwise the benchmark router decides, as before.
   * The dashboard marks each decision "your data" or "benchmark".
   * New grades are picked up without a restart.

**A bigger pool than your own traffic.** One person's grades are a
calibration layer, not a training set.
[`evaluation/gateway/pool.py`](../src/evaluation/gateway/pool.py) builds a
shared pool of real coding requests in `~/.cache/aar/routing-pool/`:

* `python -m evaluation.gateway.pool arena --n 120`: single-turn English
  coding prompts from LMArena (2025, CC-BY-4.0), replayed in an empty folder.
* `python -m evaluation.gateway.pool swe --n 80`: SWE-bench Verified issues,
  each replayed in its repository at the issue's commit (git worktrees),
  asking for root cause and fix. The maintainers' merged fix goes to the
  judge as a reference, so these labels rest on a known-good answer.

Grade the pool into its own file, not the one the gateway routes on:

```bash
aar-gateway-grade --turns ~/.cache/aar/routing-pool/turns.jsonl \
    --out ~/.cache/aar/routing-pool/graded.jsonl --n 25 --timeout 1200
```

Each row also gets `matches_best`: whether the judge found that answer as
good as the best of the three. "Did Haiku match the best?" is the label a
router is trained on. Rows carry `group` (the repository, or `arena`) so
train/test splits can be made by group.

Known limits:

* The replay sees the repository as it is now, not as it was at the turn.
* Read-only means an edit request is graded on its plan, not on an edit.
* The judge is a Claude model, so a family preference is possible.
  Anonymising and shuffling limit that; they don't remove it.

## Subagents

Claude Code's subagents (Agent tool spawns) go through the gateway like
everything else, as conversations of their own. Each gets a fresh context
and returns only its final message to the main agent, so the work that
fills a context window can run on a cheaper model without the main
conversation paying for it. [`subagents.py`](../src/router/gateway/subagents.py)
gives each one a tier by role.

**How a subagent is recognised.** Its system prompt starts with Claude
Code's billing line, which carries `cc_is_subagent=true` (seen on Claude Code
2.1.293; the main agent's does not). Its first message is the task the main
agent wrote.

**Its role**, first match wins:

1. **Definition.** `aar-gateway code` hands Claude Code six roles
   (`--agents`), each prompt marked `aar-role: <name>`.
2. **Built-in.** Claude Code's Explore and Plan, by their system prompts.
3. **Task keywords**, from the first 400 characters of the task, in priority
   order: root cause/debug/architecture → architect; review/audit → reviewer;
   write tests → tester; propose/plan/design → planner; label/extract/
   summarize/commit message → extractor; find/locate/search/explore → scout.
   A task that opens with write/generate/create/build is generation and
   gets no keyword role below tester.

A subagent with no role is routed by the general router, like a normal turn.

| Role | Tier | Output budget in its prompt |
|---|---|---|
| scout | Haiku 5.5 | ≤ 10 bullets, `path:line` + one sentence |
| extractor | Haiku 5.5 | only the structured result; big results to a file |
| Explore (built-in) | Haiku 5.5 | Claude Code's own prompt |
| Plan (built-in) | Sonnet 5.5 | Claude Code's own prompt |
| reviewer | Sonnet 5.5 | verdict + ≤ 10 one-line findings with severity |
| planner | Sonnet 5.5 | ≤ 12 steps + ≤ 5 risks, no prose |
| tester | Sonnet 5.5 | files changed, pass/fail counts, one line per failure |
| architect | session's model | conclusion, ≤ 8 findings with evidence, next step |

**Rules.**

* A role is a ceiling on cost, never an upgrade. A Sonnet-tier role in a
  Haiku session stays on Haiku.
* If the role's model can't take the request (it rejects its settings, is
  blocked for this Claude Code version, or the prompt doesn't fit), the
  general router decides instead.
* The subagent's tool loop stays on the model its first request got.
* Moving a subagent never costs the main conversation a cache rewrite: its
  system prompt and tools differ, so its prompt cache is separate.

**The definitions work without the gateway too.** Each pins its tier with
Claude Code's own `model:` field. `aar-gateway agents --install` writes them
to `~/.claude/agents/` (existing files are kept), and `aar-gateway agents`
prints them as `--agents` JSON. `AAR_GATEWAY_AGENTS=0` stops `aar-gateway
code` from passing them; passing your own `--agents` does the same.

**Config.** `interceptor.subagents.roles` maps a role to a model (`null`
keeps the session's); `enabled: false` routes subagents like any turn.

**What it shows.** Every request is logged with `subagent`, `agent_role` and
`role_source`. The dashboard's Subagents table shows, per role: spawns,
requests, the models that answered, cost against the session's model, and
cache-hit rate.

**Baseline from subagents you already ran.** `aar-gateway-subagents` reads
Claude Code's own subagent transcripts
(`~/.claude/projects/*/<session>/subagents/`), gives each the role the
gateway would, and prices its recorded tokens on the model it ran on and on
the role's model. This measures usage, not quality: whether the cheaper
model's answers would have been as good is for grading to show.

**Grading subagent tasks.** `aar-gateway-grade --from-subagents --subagents
only --n 10` adds past subagent tasks that can still be replayed (their
folder and every file their task names still exist) and grades them on each
tier, like your own messages. Tasks that read deleted scratch files are
skipped. With `capture_prompts: true`, the gateway also captures each new
subagent's task, tagged with its role, so the pool grows on its own.

## Configuration

Everything is in `configs/gateway.example.yaml`, commented. Environment
variables:

* `AAR_GATEWAY_CONFIG`: path to the config.
* `AAR_GATEWAY_HOST` / `AAR_GATEWAY_PORT`: default `127.0.0.1:8787`.
* `AAR_GATEWAY_LOG`: the request log.
* `AAR_GATEWAY_UPSTREAM`: default `https://api.anthropic.com`.

A request with no Anthropic credentials, for a model not on an `openai`
lane, gets a 401: the gateway has nothing of its own to answer with.

## Verified end to end

Each of these was run through real Claude Code on a Claude subscription,
with the gateway moving the turn from Opus 5.5 to Sonnet 5.5:

| What | Claude Code did | Result |
|---|---|---|
| Multi-step tool loop | `Bash` (count files), then `Read` (a source file) | correct answer; every request of the loop stayed on Sonnet ("same turn") |
| Skill | `Skill(aar-architecture)` | skill loaded and answered |
| On-demand tool | `ToolSearch(select:WebFetch)`, then `WebFetch` | fetched and answered |
| Non-Claude model | tool call → tool result → streamed answer, via local `llama3.1` | correct answer |

## Compared with claude-code-router

claude-code-router (CCR) routes with fixed rules, per request:

* `provider,model` in the model name: that model;
* a prompt over `longContextThreshold` (60k): the long-context model;
* a `<CCR-SUBAGENT-MODEL>` tag in a subagent's system prompt: that model;
* a Haiku request (Claude Code's background work): the background model;
* a `web_search` tool: the web-search model;
* `thinking` present: the think model;
* otherwise: the default.

It also has per-model request transformers, and a status line. This gateway
borrows the status line and the session id (read from `metadata.user_id`,
as CCR does). For the rest, the trained router replaces the rules:

* **Difficulty.** A per-prompt quality prediction stands in for the coarse
  "thinking on → big model" rule.
* **Long context and web search.** Hard filters (context window, lossless
  translation) cover them.
* **Cost.** Cost is estimated with the prompt cache in it, which no CCR rule
  sees.
* **Background requests** stay on the model Claude Code asked for: they are
  ~1k tokens, and some models reject their settings.

## Not done yet

* **Your graded data starts empty.** Until about 20 of your turns are
  graded (`aar-gateway-grade`), decisions rest on 2025 benchmark models
  through stand-ins.
* **No grading of live answers.** Grading replays captured turns afterwards.
  It does not check, in the moment, whether the answer you got held up.
* **No multi-user hosting.** One local process per person, like
  claude-code-router; it binds to `127.0.0.1`.
* **Haiku 4.5 can't serve Claude Code.** It rejects the `effort` setting
  Claude Code sends. Haiku 5.5, which accepts it, is the fast tier.
