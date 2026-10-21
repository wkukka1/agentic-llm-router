// Aptic inside Claude Code: each turn's model switched from inside Claude Code
// by the local router (the `route` option, on by default); a pane with each
// turn's model, why, cache use and savings; and a line under each answer.
// The router being down never blocks anything: the mod then keeps the model
// Claude Code asked for and shows what it can count itself.

import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { CacheTotals, GatewayView, TurnRow } from '../types'

const GATEWAY = 'http://127.0.0.1:8787'
const PANE = 'aptic'
const EVERY_MS = 60_000

const empty: GatewayView = { isRunning: false, sessionSavedPct: null, allSavedPct: null, allRequests: 0, turns: [] }
const gateway = atom({ plugin: 'aptic', key: 'gateway' } as const, empty)
const cache = atom({ plugin: 'aptic', key: 'cache' } as const, { read: 0, written: 0, fresh: 0 })
const DOWN = "Aptic's router isn't running: /aptic starts it."

// Share of the prompt read from the cache, of everything the requests sent
export const hitRate = (u: CacheTotals): number | null => {
  const total = u.read + u.written + u.fresh
  return total ? Math.round((u.read / total) * 100) : null
}

export const pretty = (m: string | null | undefined): string =>
  (m || '?').replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

const k = (n: number): string => (n >= 1000 ? Math.round(n / 1000) + 'k' : String(n))

// A turn's usage against the model asked for, as the pane prints it
export const vsAsked = (pct: number | null): string =>
  pct == null ? '' : pct === 0 ? 'same usage' : pct > 0 ? `${pct}% less` : `${-pct}% more`

const kept = (t: TurnRow): boolean => t.models.every(m => m === t.requested_model)

// Why a turn ran where it did, in words (the gateway logs the router's own terms)
export const why = (t: TurnRow): string => {
  const r = (t.reason || '').replace(/^routed in Claude Code: /, '')
  if (/^cheapest within/.test(r) || r === 'router') {
    return kept(t) ? `router: needs ${pretty(t.requested_model)}` : `router: ${pretty(t.models.at(-1))} should do as well`
  }
  if (r.startsWith('continues a server-side thread')) return 'continues a thread, so stays on its model'
  if (r === "client's choice" || r === 'as asked' || r === 'same turn as the previous request') return 'your model, as asked'
  if (r === 'no eligible candidates') return 'no cheaper model fits this prompt'
  if (r.startsWith('side request')) return "Claude Code's own small requests"
  if (r.startsWith('fell back')) return 'the routed model failed, so your model answered'
  return r || 'your model, as asked'
}

// The model(s) that answered a turn and its cache use, for the line under the answer
export const turnLine = (t: CacheTotals & { requests: number; models: string[] }): string => {
  const hit = hitRate(t)
  return `${t.models.map(pretty).join(', ') || '?'} · cache ${hit === null ? '–' : hit + '% hit'} ` +
    `(${k(t.read)} read, ${k(t.written)} written) · ${t.requests} request${t.requests === 1 ? '' : 's'}`
}

async function getJson($: any, path: string, init?: any): Promise<any | null> {
  try {
    const res = await $.http.fetch(GATEWAY + path, init)
    return res.ok ? JSON.parse(res.text) : null
  } catch {
    return null
  }
}

async function refresh($: any): Promise<void> {
  const id = String(await $.session.id())
  const [stats, mine] = await Promise.all([
    getJson($, '/stats'),
    getJson($, '/api/session-requests?limit=40&session_id=' + encodeURIComponent(id)),
  ])
  const t = stats && stats.totals
  const view: GatewayView = {
    isRunning: Boolean(stats),
    sessionSavedPct: mine ? mine.saved_pct : null,
    allSavedPct: t && t.cost_if_requested_usd ? Math.round((1 - t.cost_usd / t.cost_if_requested_usd) * 100) : null,
    allRequests: t ? t.requests : 0,
    turns: mine && mine.turns ? mine.turns : [],
  }
  await update($, gateway, () => view)
  const hit = hitRate(await read($, cache))
  $.ui.status(
    (view.isRunning
      ? `Aptic: ${view.allSavedPct ?? '–'}% less plan usage · ${view.allRequests} requests`
      : DOWN) +
      (hit === null ? '' : ` · this session's cache: ${hit}% hit`),
  )
}

// How the plugin runs the gateway when it isn't running: uv fetches and caches it,
// so the user installs nothing by hand. Pinned to the release this plugin ships with.
export const GATEWAY_SPEC = 'aar-gateway @ git+https://github.com/wkukka1/agentic-llm-router@main#subdirectory=aar-gateway'

// Start the gateway if nothing answers: first-time setup (router file, config,
// nothing else changed), then a background start. Resolves to what happened.
async function ensureGateway($: any): Promise<'running' | 'started' | 'failed' | 'no-uv'> {
  if (await getJson($, '/health')) return 'running'
  const home = String((await $.env.get('HOME')) || (await $.env.get('USERPROFILE')) || '')
  const sep = home.includes('\\') ? '\\' : '/'
  // VS Code started from the dock often lacks ~/.local/bin on PATH: try uv's own install folder too
  for (const uvx of ['uvx', home + sep + '.local' + sep + 'bin' + sep + 'uvx']) {
    const run = (args: string[]) => $.process.run([uvx, '--from', GATEWAY_SPEC, 'aar-gateway', ...args], { timeoutMs: 600_000 })
    try {
      if (home && !(await $.fs.exists(home + sep + '.aar-gateway' + sep + 'config.yaml'))) {
        await run(['setup', '--no-enable', '--no-share', '--no-mod'])
      }
      await run(['start', '--quiet'])
    } catch {
      continue  // this uvx isn't there (or couldn't start): try the next
    }
    return (await getJson($, '/health')) ? 'started' : 'failed'
  }
  return 'no-uv'
}

export const register: Register = (on, options) => {
  const routeInClaudeCode = options?.route !== false  // on unless turned off: a plugin-only session has no proxy
  // set at session start: whether this session's requests go through the gateway's proxy
  // (it logs them itself) or straight to Anthropic (the mod reports them)
  let viaProxy = false
  let sessionModel = ''
  // the current main-loop turn: what it ran on and its cache use; its routed model when routing here
  let turn: (CacheTotals & { requests: number; models: string[]; model: string | null }) | null = null

  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'aptic', description: "Show Aptic's savings now (and start its router if it isn't running)" })
    await $.command.register({ name: 'aptic-pane', description: 'Open the Aptic pane' })
    void $.ui.open({ id: PANE, title: 'Aptic' })
    viaProxy = String((await $.env.get('ANTHROPIC_BASE_URL')) || '').includes('127.0.0.1:8787')
    sessionModel = String(await $.session.model())
    ensureGateway($).then(state => {
      if (state === 'no-uv') {
        $.ui.toast('Aptic needs uv to run its router: install it from https://docs.astral.sh/uv/ then restart Claude Code.')
      }
      return refresh($)
    }).catch(() => {})
    refresh($).catch(() => {})  // display only: a failed refresh shows the old numbers
    $.clock.every(EVERY_MS, () => refresh($).catch(() => {}))
    return next(e)
  })

  on('command.run', { command: 'aptic-pane' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Aptic', focus: true })
    return { text: 'Aptic pane opened.' }
  })

  on('command.run', { command: 'aptic' }, async $ => {
    const state = await ensureGateway($)
    await refresh($)
    const view = await read($, gateway)
    const hit = hitRate(await read($, cache))
    if (!view.isRunning) {
      return { text: state === 'no-uv' ? 'Aptic needs uv to run its router: https://docs.astral.sh/uv/' : "Aptic's router couldn't start." }
    }
    return {
      text: (state === 'started' ? "Started Aptic's router. " : '') +
        `Aptic: ${view.allSavedPct ?? '–'}% less plan usage · ${view.allRequests} requests` +
        (view.sessionSavedPct == null ? '' : ` · this session: ${view.sessionSavedPct}% less`) +
        (hit === null ? '' : ` · cache ${hit}% hit`),
    }
  })

  on('turn.start', async ($, e, next) => {
    if (e.agentId) return next(e)
    turn = { read: 0, written: 0, fresh: 0, requests: 0, models: [], model: null }
    sessionModel = String(await $.session.model())
    if (routeInClaudeCode && e.text) {
      // ask the gateway's router; no answer keeps the model Claude Code asked for
      const asked = sessionModel
      const picked = await getJson($, '/api/route', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: e.text, model: asked, session_id: String(await $.session.id()) }),
      })
      turn.model = picked && picked.model ? picked.model : null
    }
    return next(e)
  })

  // Each request: on the turn's routed model when routing here; count its cache use
  on('turn.step', async function* ($, e, next) {
    const mine = !e.agentId && turn
    const result = yield* next(mine && turn!.model ? { ...e, model: turn!.model } : e)
    const u: any = result && (result as any).usage
    if (mine && u && turn) {
      const add = { read: u.cache_read_input_tokens || 0, written: u.cache_creation_input_tokens || 0, fresh: u.input_tokens || 0 }
      turn.read += add.read
      turn.written += add.written
      turn.fresh += add.fresh
      turn.requests += 1
      if (u.model && !turn.models.includes(u.model)) turn.models.push(u.model)
      await update($, cache, c => ({ read: c.read + add.read, written: c.written + add.written, fresh: c.fresh + add.fresh }))
    }
    if (u && !viaProxy) {
      // straight to Anthropic: the gateway never saw this request, so report its usage for the savings and the pane
      getJson($, '/api/mod-record', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          session_id: String(await $.session.id()), agent_id: e.agentId, new_turn: !e.agentId && e.index === 0,
          requested_model: sessionModel || e.model, routed_model: u.model || (mine && turn ? turn.model : null) || e.model,
          input_tokens: u.input_tokens, output_tokens: u.output_tokens,
          cache_read_input_tokens: u.cache_read_input_tokens, cache_creation_input_tokens: u.cache_creation_input_tokens,
        }),
      }).catch(() => {})
    }
    return result
  })

  // A subagent's model by the kind of work, from the gateway (when routing here)
  on('agent.spawn', async ($, e, next) => {
    if (!routeInClaudeCode) return next(e)
    const picked = await getJson($, '/api/subagent-spawn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: String(await $.session.id()), subagent_type: e.subagentType, prompt: e.prompt, model: (e as any).model }),
    })
    return next(picked && picked.alias ? { ...e, model: picked.alias } : e)
  })

  on('turn.complete', async ($, e, next) => {
    if (e.agentId || !turn || !turn.requests) return next(e)
    const text = turnLine(turn)
    turn = null
    const result: any = await next(e)  // let Claude Code and other mods finish the turn first
    refresh($).catch(() => {})  // display only: a failed refresh shows the old numbers
    return { ...(result || {}), text: result && result.text ? result.text + '\n' + text : text }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    const view = await read($, gateway)
    const hit = hitRate(await read($, cache))
    // each turn takes two lines, plus one per subagent kind
    const room = Math.max(2, Math.floor(((e.props.scroll?.bodyRows ?? (e.viewport?.rows ?? 24) - 8) - 5) / 2))
    const turns = view.turns.slice(-room).reverse()
    return (
      <Box flexDirection="column">
        {!view.isRunning && <Text color="warning">{DOWN}</Text>}
        <Text weight="semibold">
          This session: {view.sessionSavedPct == null ? '–' : vsAsked(Math.round(view.sessionSavedPct)) + ' plan usage'}
          {hit === null ? '' : ` · cache ${hit}% hit`}
        </Text>
        <Text dimColor>
          All sessions: {view.allSavedPct == null ? '–' : vsAsked(view.allSavedPct)} · {view.allRequests} requests
        </Text>
        <Text dimColor>Routing {routeInClaudeCode ? 'in Claude Code' : 'by the proxy'} · newest turn first</Text>
        <Text> </Text>
        {turns.length === 0 && <Text dimColor>No turns yet in this session.</Text>}
        {turns.map((t: TurnRow, i: number) => {
          const moved = !kept(t)
          const total = t.cache_read + t.cache_written + t.fresh
          return (
            <Box key={`turn-${i}`} flexDirection="column" marginBottom={1}>
              <Text>
                <Text dimColor>{t.time} </Text>
                <Text weight="semibold">
                  {moved ? `${pretty(t.requested_model)} → ${t.models.map(pretty).join(', ')}` : pretty(t.requested_model)}
                </Text>
                {t.saved_pct != null && moved ? <Text color="success"> · {vsAsked(t.saved_pct)}</Text> : ''}
                <Text dimColor>
                  {' · '}{t.requests} request{t.requests === 1 ? '' : 's'}
                  {total ? ` · cache ${Math.round((t.cache_read / total) * 100)}%` : ''}
                </Text>
                {t.errors ? <Text color="error"> · {t.errors} failed</Text> : ''}
              </Text>
              <Text dimColor>      {why(t)}</Text>
              {t.subagents.map(a => (
                <Text dimColor>      ↳ {a.role} on {pretty(a.model)} · {a.requests} request{a.requests === 1 ? '' : 's'}</Text>
              ))}
            </Box>
          )
        })}
      </Box>
    )
  })
}
