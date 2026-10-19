// AAR Gateway mod: savings and cache use, inside Claude Code.
// Read-only: it reads the local gateway's stats and the token usage Claude Code
// reports for each request. It never starts or changes anything.

const STATS_URL = 'http://127.0.0.1:8787/stats'
const EVERY_MS = 60_000

// This session's main-conversation usage, and the current turn's
const session = { read: 0, written: 0, fresh: 0 }
let turn = null

// Share of the prompt read from the cache, of everything the requests sent
export function hitRate(u) {
  const total = u.read + u.written + u.fresh
  return total ? Math.round((u.read / total) * 100) : null
}

const k = (n) => (n >= 1000 ? Math.round(n / 1000) + 'k' : String(n))
const pretty = (m) => (m || '').replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

// Plan usage saved against the model the user asked for, from the gateway's totals
export function summarize(stats) {
  const t = stats && stats.totals
  if (!t || !t.cost_if_requested_usd) return 'AAR gateway: no priced requests yet'
  const pct = Math.round((1 - t.cost_usd / t.cost_if_requested_usd) * 100)
  return `AAR gateway: ${pct}% less plan usage than no router · ${t.requests} requests`
}

// The line under an answer: the model(s) that answered and the turn's cache use
export function turnLine(t) {
  const hit = hitRate(t)
  const models = [...t.models].map(pretty).join(', ')
  return `${models} · cache ${hit === null ? '–' : hit + '% hit'} (${k(t.read)} read, ${k(t.written)} written) · ${t.requests} request${t.requests === 1 ? '' : 's'}`
}

async function statusLine($) {
  const hit = hitRate(session)
  const cache = hit === null ? '' : ` · this session's cache: ${hit}% hit`
  try {
    const res = await $.http.fetch(STATS_URL)
    if (!res.ok) return 'AAR gateway: not answering (run `aar-gateway start`)' + cache
    return summarize(JSON.parse(res.text)) + cache
  } catch {
    return 'AAR gateway: not running (run `aar-gateway start`)' + cache
  }
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    // /aar-savings prints the same line on demand
    await $.command.register({ name: 'aar-savings', description: 'Show the gateway savings and cache use now' })
    const refresh = async () => $.ui.status(await statusLine($))
    refresh()
    $.clock.every(EVERY_MS, refresh)
    return next(e)
  })

  on('command.run', { command: 'aar-savings' }, async ($) => ({ text: await statusLine($) }))

  on('turn.start', async ($, e, next) => {
    turn = { read: 0, written: 0, fresh: 0, requests: 0, models: new Set() }
    return next(e)
  })

  // Each request to the model: count its cache use (the main conversation only)
  on('turn.step', async function* ($, e, next) {
    const result = yield* next(e)
    const u = result && result.usage
    if (u && !e.agentId && turn) {
      const add = { read: u.cache_read_input_tokens || 0, written: u.cache_creation_input_tokens || 0, fresh: u.input_tokens || 0 }
      for (const key of Object.keys(add)) { turn[key] += add[key]; session[key] += add[key] }
      turn.requests += 1
      if (u.model) turn.models.add(u.model)
    }
    return result
  })

  on('turn.complete', async ($, e, next) => {
    if (e.agentId || !turn || !turn.requests) return next(e)
    const text = turnLine(turn)
    turn = null
    $.ui.status(await statusLine($))
    return { text }
  })
}
