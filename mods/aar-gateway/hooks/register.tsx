// AAR Gateway inside Claude Code: a pane with each request's model, cache use
// and savings; a line under each answer; and, with the `route` option on, the
// model switched per turn from inside Claude Code using the gateway's router.
// The gateway being down never blocks anything: the mod then keeps the model
// Claude Code asked for and shows what it can count itself.

import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { CacheTotals, GatewayView, RequestRow } from '../types'

const GATEWAY = 'http://127.0.0.1:8787'
const PANE = 'aar-gateway'
const EVERY_MS = 60_000

const empty: GatewayView = { isRunning: false, sessionSavedPct: null, allSavedPct: null, allRequests: 0, recent: [] }
const gateway = atom({ plugin: 'aar-gateway', key: 'gateway' } as const, empty)
const cache = atom({ plugin: 'aar-gateway', key: 'cache' } as const, { read: 0, written: 0, fresh: 0 })

// Share of the prompt read from the cache, of everything the requests sent
export const hitRate = (u: CacheTotals): number | null => {
  const total = u.read + u.written + u.fresh
  return total ? Math.round((u.read / total) * 100) : null
}

export const pretty = (m: string | null | undefined): string =>
  (m || '?').replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

const k = (n: number): string => (n >= 1000 ? Math.round(n / 1000) + 'k' : String(n))

// One request's cost against the model asked for, as the pane prints it
export const vsAsked = (r: RequestRow): string => {
  if (r.cost_usd == null || !r.cost_if_requested_usd) return ''
  const pct = Math.round((1 - r.cost_usd / r.cost_if_requested_usd) * 100)
  return pct === 0 ? 'same' : pct > 0 ? `-${pct}%` : `+${-pct}%`
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
    getJson($, '/api/session-requests?limit=60&session_id=' + encodeURIComponent(id)),
  ])
  const t = stats && stats.totals
  const view: GatewayView = {
    isRunning: Boolean(stats),
    sessionSavedPct: mine ? mine.saved_pct : null,
    allSavedPct: t && t.cost_if_requested_usd ? Math.round((1 - t.cost_usd / t.cost_if_requested_usd) * 100) : null,
    allRequests: t ? t.requests : 0,
    recent: mine ? mine.recent : [],
  }
  await update($, gateway, () => view)
  const hit = hitRate(await read($, cache))
  $.ui.status(
    (view.isRunning
      ? `AAR gateway: ${view.allSavedPct ?? '–'}% less plan usage than no router · ${view.allRequests} requests`
      : 'AAR gateway: not running (run `aar-gateway start`)') +
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
    await $.command.register({ name: 'aar-savings', description: 'Show the gateway savings and cache use now' })
    await $.command.register({ name: 'aar-pane', description: 'Open the AAR gateway pane' })
    void $.ui.open({ id: PANE, title: 'AAR gateway' })
    viaProxy = String((await $.env.get('ANTHROPIC_BASE_URL')) || '').includes('127.0.0.1:8787')
    sessionModel = String(await $.session.model())
    ensureGateway($).then(state => {
      if (state === 'no-uv') {
        $.ui.toast('AAR gateway needs uv to run its router: install it from https://docs.astral.sh/uv/ then restart Claude Code.')
      }
      return refresh($)
    }).catch(() => {})
    refresh($).catch(() => {})  // display only: a failed refresh shows the old numbers
    $.clock.every(EVERY_MS, () => refresh($).catch(() => {}))
    return next(e)
  })

  on('command.run', { command: 'aar-pane' }, async $ => {
    await $.ui.open({ id: PANE, title: 'AAR gateway', focus: true })
    return { text: 'AAR gateway pane opened.' }
  })

  on('command.run', { command: 'aar-savings' }, async $ => {
    await refresh($)
    const view = await read($, gateway)
    const hit = hitRate(await read($, cache))
    return {
      text: view.isRunning
        ? `AAR gateway: ${view.allSavedPct ?? '–'}% less plan usage than no router · ${view.allRequests} requests` +
          (view.sessionSavedPct == null ? '' : ` · this session: ${view.sessionSavedPct}%`) +
          (hit === null ? '' : ` · cache ${hit}% hit`)
        : 'AAR gateway: not running (run `aar-gateway start`)',
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
    const room = Math.max(3, (e.props.scroll?.bodyRows ?? (e.viewport?.rows ?? 24) - 8) - 4)
    const rows = view.recent.slice(-room).reverse()
    return (
      <Box flexDirection="column">
        {!view.isRunning && <Text dimColor>AAR gateway isn't running: run `aar-gateway start`.</Text>}
        <Text>
          This session: {view.sessionSavedPct == null ? '–' : view.sessionSavedPct + '% less plan usage'}
          {hit === null ? '' : ` · cache ${hit}% hit`}
        </Text>
        <Text dimColor>
          All sessions: {view.allSavedPct == null ? '–' : view.allSavedPct + '% less'} · {view.allRequests} requests ·
          routing {routeInClaudeCode ? 'in Claude Code' : 'by the gateway'}
        </Text>
        <Text> </Text>
        {rows.length === 0 && <Text dimColor>No requests yet in this session.</Text>}
        {rows.map((r: RequestRow) => {
          const total = (r.cache_read_input_tokens || 0) + (r.cache_creation_input_tokens || 0) + (r.input_tokens || 0)
          const rhit = total ? Math.round(((r.cache_read_input_tokens || 0) / total) * 100) + '%' : '–'
          const moved = r.routed_model !== r.requested_model
          return (
            <Text dimColor={!moved && !r.subagent}>
              {(r.ts || '').slice(11, 16)} {r.subagent ? '↳ ' + (r.agent_role || 'subagent') + ' ' : ''}
              {moved ? `${pretty(r.requested_model)} → ${pretty(r.routed_model)}` : pretty(r.routed_model)}
              {' · cache '}{rhit}{vsAsked(r) ? ' · ' + vsAsked(r) : ''}{r.status !== 200 ? ` · ${r.status}` : ''}
              {r.reason ? ' · ' + r.reason : ''}
            </Text>
          )
        })}
      </Box>
    )
  })
}
