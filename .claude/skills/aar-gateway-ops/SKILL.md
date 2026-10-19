---
name: aar-gateway-ops
description: Operate AAR Gateway on this machine: check it, explain its savings, diagnose connection, thread and routing errors, and make the changes the user approves. Use for any question about the local gateway, its savings, or why Claude Code isn't reaching it.
---

# AAR Gateway operations

The gateway is a local process (`127.0.0.1:8787`) that Claude Code talks to. It
routes each message to the cheapest Claude model expected to answer well, and
keeps a request log at `~/.aar-gateway/requests.jsonl`. You operate it with the
`aar-gateway` command and the read-only script `scripts/query.py`.

**Read `reference/contracts.md` before running any command.** It lists every
command you may use, what it changes, and whether it needs the user's approval.

## Rules

0. **The command.** Always run `$HOME/.local/share/uv/tools/agentic-llm-router/bin/aar-gateway`
   (the installed copy). Never plain `aar-gateway`: the repo's `.venv` copy breaks when
   iCloud hides its `.pth` files. **One plain command per Bash call:** no `;`, `&&`, `||`,
   pipes, redirects, `$(...)`, or `echo` wrappers. If you need two results, make two calls.
1. **Read first.** Start with `aar-gateway doctor --json` and
   `aar-gateway stats`. Answer from those and the log before changing anything.
2. **Ask before changing anything.** Any command marked *confirm* in the
   contracts needs a yes from the user in this conversation, every time. Say what
   it will do, in one sentence, then wait.
3. **Never** kill processes by name pattern. Stop a gateway only with
   `aar-gateway stop`, which uses the pid file, or by a pid you read from
   `/health`.
4. **Never** edit `~/.claude/settings.json`, `~/.aar-gateway/config.yaml` or any
   token file by hand. Use the commands.
5. **Never** print, log or quote a token, an API key, or the contents of
   `~/.aar-gateway/share/token`.
6. **Never** read prompt text (`turns.jsonl`) unless the user asks for a specific
   prompt. The request log holds no prompt text; keep it that way.
7. **Never** send anything off the machine, and never run `share upload` unless
   the user asks for that upload.
8. **Never** delete logs or records except through `aar-gateway share delete`
   after confirmation.

## Report format

Start with one line: the verdict. Then the numbers that show it, from the JSON.
Then what you did, or what you propose and need approved. Keep it under 12 lines.

When giving savings, always use `vs_no_router` (the model the user asked for,
cache included) as the headline. Quote `vs_always_opus` only if asked, and say it
prices cheap requests as Opus, so it reads higher.

## Playbooks

Each one is a symptom, the checks to run, and the fix to propose.

**Claude Code shows "API Error: Connection refused" (ECONNREFUSED)**
- `doctor --json`: is the gateway check ok?
- `GET /health`. If it answers, the problem is elsewhere: check `ANTHROPIC_BASE_URL`
  in the project's settings.
- If it doesn't answer: the session should already have fallen back to Anthropic
  (the `bypassed` marker in `~/.aar-gateway/`). Propose `start` (confirm).
- If it keeps dying: `logs -n 80`. Look for "Traceback" or "address already in
  use". A second gateway on the same port can't start; say so.

**The gateway is running but slow or not answering**
- `GET /health` with a short timeout. No answer while the pid file names a live
  process means a stuck process. Propose `restart` (confirm). Do not kill it by name.

**"No thread state" (404)**
- Expected: the thread expired while idle. Claude Code resends the whole
  conversation. Not an error; `query.py errors` already excludes it.

**"differing field(s): model" (400 on a thread)**
- The gateway retries on the other Claude models and remembers the one that works.
  If it still fails, run `query.py threads` and report the count. It is a bug to
  report, not something to fix from here.

**429 / "busy" in the dashboard**
- Anthropic's rate limit on the plan. Nothing to fix. Say so and quote the count
  from `query.py errors`.

**Router not loaded**
- `doctor --json` shows the router check and its error. Propose
  `setup` (confirm; it rewrites nothing unless `--force`).

**"How much am I saving?"**
- `query.py savings --days 7`, then `query.py savings` for all time. Quote
  `vs_no_router`.

**A reroute cost more than no router**
- `query.py switches`. Explain: a switch rebuilds the cheaper model's cache. Say
  how many were costlier. Do not suggest hiding it.

**Upgrading to a new release**
- Propose the install command for the new tag (confirm). Then `restart` (confirm),
  then `doctor --json`.

**Windows or Linux differences**
- Windows: no start-at-login and no auto-restart. If the gateway crashed, the next
  message restarts it. Say so instead of promising otherwise.
- Linux: start-at-login uses systemd user units; check with `service status`.

**Sharing usage data**
- `share status --json`. Never show the token. Upload only on request (confirm).
