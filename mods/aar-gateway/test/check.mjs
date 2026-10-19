import assert from 'node:assert'
import { register, hitRate, turnLine, summarize } from '../hooks/register.js'
assert.equal(hitRate({ read: 90, written: 5, fresh: 5 }), 90)
assert.equal(hitRate({ read: 0, written: 0, fresh: 0 }), null)
assert.equal(summarize({ totals: { cost_usd: 70, cost_if_requested_usd: 100, requests: 5 } }), 'AAR gateway: 30% less plan usage than no router · 5 requests')
// drive one turn with two requests and a subagent request that must not count
const h = {}
register((name, a, b) => { h[name] = b ?? a })
const shown = []
const $ = { ui: { status: (t) => shown.push(t) }, http: { fetch: async () => { throw new Error('down') } } }
await h['turn.start']($, {}, async () => {})
for (const [u, agentId] of [[{ cache_read_input_tokens: 180000, cache_creation_input_tokens: 4000, input_tokens: 10, model: 'claude-haiku-5-5' }],
                            [{ cache_read_input_tokens: 184000, cache_creation_input_tokens: 500, input_tokens: 5, model: 'claude-haiku-5-5' }],
                            [{ cache_read_input_tokens: 1, cache_creation_input_tokens: 99999, input_tokens: 1, model: 'claude-opus-5-5' }, 'sub1']]) {
  const gen = h['turn.step']($, { agentId }, async function* () { return { usage: u } })
  let r = await gen.next(); while (!r.done) r = await gen.next()
}
const out = await h['turn.complete']($, {}, async () => {})
console.log(out.text)
assert.match(out.text, /^haiku 5\.5 · cache 99% hit \(364k read, 5k written\) · 2 requests$/)
assert.match(shown.at(-1), /not running.*this session.s cache: 99% hit/)
console.log('self-check passed')
