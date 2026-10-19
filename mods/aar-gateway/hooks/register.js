// AAR Gateway mod: savings from the local gateway, inside Claude Code.
// Read-only: it asks the gateway for its stats on localhost and shows them.
// If the gateway isn't running, it says so; it never starts or changes anything.

const STATS_URL = 'http://127.0.0.1:8787/stats'
const EVERY_MS = 60_000

// Plan usage saved against the model the user asked for, from the gateway's totals
function summarize(stats) {
  const t = stats && stats.totals
  if (!t || !t.cost_if_requested_usd) return 'AAR gateway: no priced requests yet'
  const pct = Math.round((1 - t.cost_usd / t.cost_if_requested_usd) * 100)
  return `AAR gateway: ${pct}% less plan usage than no router · ${t.requests} requests`
}

async function savingsLine($) {
  try {
    const res = await $.http.fetch(STATS_URL)
    if (!res.ok) return 'AAR gateway: not answering (run `aar-gateway start`)'
    return summarize(JSON.parse(res.text))
  } catch {
    return 'AAR gateway: not running (run `aar-gateway start`)'
  }
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    // /aar-savings prints the same line on demand
    await $.command.register({ name: 'aar-savings', description: 'Show the gateway savings now' })
    // The status line under the prompt, refreshed each minute
    const refresh = async () => $.ui.status(await savingsLine($))
    refresh()
    $.clock.every(EVERY_MS, refresh)
    return next(e)
  })

  on('command.run', { command: 'aar-savings' }, async ($) => {
    return { text: await savingsLine($) }
  })
}
