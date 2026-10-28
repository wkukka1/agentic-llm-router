# Aptic for Claude Code

Use less of your Claude plan with the same login, tools and skills. For each
message you type, Aptic picks the cheapest Claude model that should answer
it about as well as the model you chose, and switches the turn to it from inside
Claude Code. Subagents get a model by the kind of work they do.

## Install

In Claude Code:

```
/plugin install aptic --marketplace wkukka1/agentic-llm-router
```

## What you get

* **Routing per turn**, by a small trained router that runs on your computer.
  Turn it off in `/config` (the plugin's "Route in Claude Code" setting).
* **A pane** (`/aptic-pane`): this session's saving against always using the
  model you asked for, the mix of models its requests ran on, its requests,
  cache hit and turns, and all sessions' plan usage; then a table of turns
  (newest or oldest first) with each turn's time, model, decision, requests
  and cache hit. Under each turn are the subagents it handed work to: their
  kind, the model they ran on, and whether their answer came back to this
  session (`↩ answered ~29 tok`) or they are still working. When the router
  keeps prompt history, each subagent's task and the start of its answer show
  too (see below). A Subagents row sums them up: spawned, answered, and their share of the
  session's cost. A chevron collapses the pane to the two savings and a
  requests-per-turn chart; the pane remembers your choice.
* **A line under each answer** with the model that answered and its cache use,
  and the totals on the status line. `/aptic` prints them.

## Requirements

* Claude Code 2.1.287 or later.
* [uv](https://docs.astral.sh/uv/). The plugin uses it to run the router
  (`uvx`), the first time downloading it in a few seconds. Without uv the
  plugin says so and Claude Code works as normal.

## What it does on your machine

* **Downloads and runs the router.** The first time, `uv` downloads the router
  from GitHub (`wkukka1/agentic-llm-router`, release `v0.7.4`) and its Python
  packages, then starts it on `127.0.0.1:8787`, which only this computer can
  reach. It writes `~/.aar-gateway/` (a config file and the router's model
  file). It doesn't change your Claude Code settings and doesn't install a
  login service.
* **Reads what you type, locally.** Your message goes to that local router to
  choose a model. Claude Code then sends the request to Anthropic with your own
  login, as it always does. The router doesn't send your messages anywhere else.
* **Keeps a history on this computer.** When the history is on, the router
  keeps each message you type, with the model it used, and each subagent's
  task, in `~/.aar-gateway/`. It never leaves the machine. Turn it off with
  `aar-gateway history off`.
* **Sharing is off.** Anonymous usage sharing is off by default, and the plugin
  never turns it on.
* **Falls back if the router is down.** Every request then goes to the model
  you asked for.

Source, tests and the measurements behind the numbers:
https://github.com/wkukka1/agentic-llm-router

License: Apache-2.0.
