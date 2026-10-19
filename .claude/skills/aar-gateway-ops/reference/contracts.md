# Contracts: what the ops agent may run

Every command here is run as `$HOME/.local/share/uv/tools/agentic-llm-router/bin/aar-gateway <command>`
(the installed copy; `aar-gateway` on its own may be the repo's venv, which breaks under iCloud),
or `python3 .claude/skills/aar-gateway-ops/scripts/query.py <command>`. One command per call: no
`;`, `&&`, `||`, pipes, redirects, subshells, or `echo` wrappers.

Legend: **read** = changes nothing. **confirm** = ask the user first, every time.
**never** = not allowed for the agent at all.

## Read

| Command | Output | Notes |
|---|---|---|
| `aar-gateway doctor --json` | `{ok, checks: [{check, ok, detail, fix}]}` | exit 0 when all ok |
| `aar-gateway stats` | the `/stats` JSON totals | exit 1 and `{"running": false}` when down |
| `aar-gateway status --json` | running, router, pid, paths | |
| `aar-gateway logs -n N` | last N log lines (text) | N up to 200 |
| `aar-gateway config` | the config file path in use | |
| `aar-gateway service status` | installed or not | |
| `aar-gateway share status --json` | sharing on/off, counts, collection endpoint, last upload | |
| `GET http://127.0.0.1:8787/health` | status, pid, router | use `curl -m 3` |
| `GET http://127.0.0.1:8787/stats` | the same as `stats` | |
| `GET http://127.0.0.1:8787/api/preferences` | mode, pin, margin, roles | |
| `python query.py savings [--days N]` | vs no router, vs always Opus, by main agent and subagents | |
| `python query.py errors [--days N]` | failed requests by status and error type | expired threads excluded, as on the dashboard |
| `python query.py switches [--days N]` | switches by model pair, costlier count, cache written | |
| `python query.py threads [--days N]` | thread requests by mode, status, mismatches | |
| `python query.py session <id-prefix> [--days N]` | one session's requests, models, savings | |

## Confirm

| Command | What it changes | Notes |
|---|---|---|
| `aar-gateway start` | starts the gateway if it isn't answering | via the login service when installed |
| `aar-gateway stop` | stops the gateway via its pid file | never by name |
| `aar-gateway restart` | stop then start | the usual fix for a stuck gateway |
| `aar-gateway service install` / `uninstall` | the start-at-login entry | macOS and Linux only |
| `aar-gateway enable` / `disable` | Claude Code's user settings (every session) | `enable` refuses a copy in a temp folder |
| `aar-gateway setup [--force]` | router file and config | `--force` rewrites the config: confirm twice |
| `aar-gateway share on` / `off` | whether usage records are kept | |
| `aar-gateway share delete` | deletes records not yet sent | |
| `aar-gateway share upload` | sends records to the configured collection endpoint | only when the user asks |
| preference changes via `POST /api/preferences` | routing mode, pin, margin, allow-upgrade, subagent tiers | show the change first |

## Never

- `kill`, `pkill`, `killall` on anything matching `aar-gateway`, `router.gateway`
  or `python`. Use `aar-gateway stop` or a pid read from `/health`.
- Editing `~/.claude/settings.json`, `~/.aar-gateway/config.yaml`,
  or any file under `~/Library/LaunchAgents`.
- `share endpoint <url>` with an address the user did not ask for. The user sets the
  collection endpoint.
- Reading or printing `turns.jsonl`, prompts, file contents from a session, or
  `Authorization` headers.
- Any network request except to `127.0.0.1:8787`, and the upload in `share upload`
  when the user asked for it.
- `rm` on anything under `~/.aar-gateway`, `~/.claude`, or the repo.
