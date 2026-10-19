import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// Beneath the mod: the session, a gateway answering (or not), and the model
function world(on: On, gateway: 'up' | 'down', picked = 'claude-haiku-5-5') {
  const asked: string[] = []
  const sent: string[] = []
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('turn.start', async ($, e) => ({ turnId: e.turnId }))
  on('turn.complete', async () => ({ text: '' }))
  on('ui.status', async () => ({ value: undefined }))
  on('clock.every', async () => ({ value: { cancel() {} } }))
  on('http.fetch', async ($, e) => {
    asked.push(String(e.url).replace('http://127.0.0.1:8787', ''))
    if (gateway === 'down') throw new Error('connect ECONNREFUSED')
    const path = String(e.url)
    const body = path.includes('/api/route') ? { model: picked, reason: 'cheapest' }
      : path.includes('/stats') ? { totals: { requests: 10, cost_usd: 7, cost_if_requested_usd: 10 } }
      : { saved_pct: 30, recent: [] }
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } }
  })
  on('turn.step', async function* ($, e) {
    sent.push(e.model)
    return {
      turnId: e.turnId, index: e.index, answer: 'done', toolUses: [],
      usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 9000, cache_creation_input_tokens: 990, model: e.model },
    } as any
  })
  return { asked, sent }
}

async function oneTurn($: any, text: string, steps = 2) {
  await $.turn.start({ text, turnId: 't1' })
  for (let index = 0; index < steps; index++) {
    const stream = $.turn.step({ turnId: 't1', index, model: 'claude-opus-5-5', messageCount: 3 })
    for await (const _ of stream) { /* drain */ }
  }
  return $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 10, isAborted: false } as any)
}

test('with routing on, every request of the turn goes to the model the gateway picked', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  const done: any = await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-haiku-5-5', 'claude-haiku-5-5'])
  expect(w.asked).toContain('/api/route')
  expect(done.text).toMatch(/haiku 5\.5 · cache 90% hit/)
})

test('with routing off (the default), the model is left alone and the router is not asked', async ($, on) => {
  const w = world(on, 'up')
  await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-opus-5-5', 'claude-opus-5-5'])
  expect(w.asked).not.toContain('/api/route')
})

test('with the gateway down, routing on keeps the model asked for and the turn still ends', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'down')
  const done: any = await oneTurn($, 'rename this variable')
  expect(w.sent).toEqual(['claude-opus-5-5', 'claude-opus-5-5'])
  expect(done.text).toMatch(/opus 5\.5 · cache 90% hit \(18k read, 2k written\) · 2 requests/)
})

test('a subagent request is neither switched nor counted in the line', { options: { route: true } }, async ($, on) => {
  const w = world(on, 'up')
  await $.turn.start({ text: 'look this up', turnId: 't1' })
  const sub = $.turn.step({ turnId: 't9', index: 0, model: 'claude-opus-5-5', messageCount: 1, agentId: 'a1' })
  for await (const _ of sub) { /* drain */ }
  expect(w.sent).toEqual(['claude-opus-5-5'])
})
