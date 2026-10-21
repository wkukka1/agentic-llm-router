import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The words a drawn tree shows, one line per Text (a Text's runs are joined)
const words = (node: any): string =>
  typeof node === 'string' ? node
    : node.type === 'Text' ? (node.children ?? []).map(words).join('') + '\n'
    : (node.children ?? []).map(words).join('')

// The pane as the engine draws it: a session with two turns, one routed to Sonnet
// with a Haiku subagent, one kept on Opus
export function session(on: On) {
  const turns = [
    { time: '21:28', requested_model: 'claude-opus-5-5', models: ['claude-sonnet-5-5'], reason: 'routed in Claude Code: cheapest within 0.05 utility of the best',
      requests: 2, side_requests: 0, errors: 0, cache_read: 9000, cache_written: 990, fresh: 10, saved_pct: 50,
      subagents: [{ role: 'lookup', model: 'claude-haiku-5-5', requests: 2 }] },
    { time: '21:30', requested_model: 'claude-opus-5-5', models: ['claude-opus-5-5'], reason: "client's choice", requests: 1, side_requests: 0,
      errors: 0, cache_read: 0, cache_written: 0, fresh: 10, saved_pct: 0, subagents: [] },
  ]
  on('ui.render', async () => ({ type: 'Box' }) as any)  // the engine's own answer beneath the plugin
  on('http.fetch', async ($, e) => {
    const url = String(e.url)
    const body = url.includes('/stats') ? { totals: { requests: 12, cost_usd: 7, cost_if_requested_usd: 10 } }
      : url.includes('/api/session-requests') ? { saved_pct: 50, turns } : {}
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } }
  })
}

test('the pane draws each turn with its model, saving, reason and subagents', async ($, on) => {
  session(on)
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('env.get', async () => ({ value: undefined }))
  on('session.start', async ($, e, next) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.status', async () => ({ value: undefined }))
  mock.clock(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)  // the first refresh fills the view
  // env and http come from the test kit; the mounted pane reads the gateway's view
  const ui = await $.ui.mount({ plugin: 'aptic', surface: 'terminal', component: 'Pane', requestId: 'aptic', props: {}, viewport: { columns: 60, rows: 40 } } as any)
  const tree = words(await ui.drawn())
  const has = (pattern: RegExp) => expect(tree).toMatch(pattern)
  has(/This session: 50% less plan usage/)
  has(/All sessions: 30% less/)
  has(/21:28/)
  has(/Sonnet 5\.5|sonnet 5\.5/)
  has(/↳ lookup on haiku 5\.5 · 2/)
  has(/21:30/)
  await ui.unmount()
})
