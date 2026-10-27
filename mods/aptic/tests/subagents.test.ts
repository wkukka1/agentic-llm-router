import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { answered, spawnLine, subagentTotals } from '../hooks/register'
import type { SpawnRow, TurnRow } from '../types'

// The words a drawn tree shows, one line per Text (a Text's runs are joined)
const words = (node: any): string =>
  typeof node === 'string' ? node
    : node.type === 'Text' ? (node.children ?? []).map(words).join('') + '\n'
    : (node.children ?? []).map(words).join('')

const lookup: SpawnRow = { spawn_id: 'abc', role: 'lookup', model: 'claude-haiku-5-5', models: ['claude-haiku-5-5'], requests: 2,
  cost_usd: 0.001, returned: true, answer_tokens: 29, task_preview: 'Find where the router reads the cost weight',
  answer_preview: 'src/router/policy.py:88 reads cost_weight' }
const judge: SpawnRow = { spawn_id: 'agent-7', role: 'judge', model: 'claude-opus-5-5', models: ['claude-opus-5-5'], requests: 1,
  cost_usd: 0.02, returned: false, answer_tokens: null, task_preview: null, answer_preview: null }

const turns: TurnRow[] = [
  { time: '21:28', requested_model: 'claude-opus-5-5', models: ['claude-opus-5-5'], reason: "client's choice", requests: 2,
    side_requests: 0, errors: 0, cache_read: 9000, cache_written: 990, fresh: 10, saved_pct: 30, cost_usd: 0.1,
    subagent_cost_usd: 0.021, returned: 1, subagents: [lookup, judge] },
]

// A session whose turn spawned two subagents: one answered (seen by the gateway), one still working
function session(on: On) {
  on('ui.render', async () => ({ type: 'Box' }) as any)
  on('http.fetch', async ($, e) => {
    const url = String(e.url)
    const body = url.includes('/stats') ? { totals: { requests: 12, cost_usd: 7, cost_if_requested_usd: 10 } }
      : url.includes('/api/session-requests') ? { saved_pct: 30, turns } : {}
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } }
  })
  on('session.id', async () => ({ value: 'session-under-test' }))
  on('session.model', async () => ({ value: 'claude-opus-5-5' }))
  on('env.get', async () => ({ value: undefined }))
  on('session.start', async ($, e) => ({ cwd: (e as any).cwd }))
  on('command.register', async () => ({ value: undefined }))
  on('ui.open', async () => ({ value: { isPlaced: false } }))
  on('ui.status', async () => ({ value: undefined }))
  on('turn.complete', async ($, e) => ({ text: (e as any).answer ?? '' }))
  mock.clock(on)
}

const PANE = { plugin: 'aptic', component: 'Pane', requestId: 'aptic', props: {}, viewport: { columns: 70, rows: 60 } }

test('a subagent line says whether its answer came back', () => {
  expect(spawnLine(lookup, {})).toBe('↳ lookup on Haiku 5.5 · 2 req · ↩ answered ~29 tok')
  expect(spawnLine(judge, {})).toBe('↳ judge on Opus 5.5 · 1 req · working…')
  expect(spawnLine(judge, { 'agent-7': 1500 })).toBe('↳ judge on Opus 5.5 · 1 req · ↩ answered ~2k tok')
  // a spawn logged before answers were tracked has no status
  expect(spawnLine({ role: 'make', model: 'claude-sonnet-5-5', requests: 3 }, {})).toBe('↳ make on Sonnet 5.5 · 3 req')
  expect([answered(lookup, {}), answered(judge, {}), answered(judge, { 'agent-7': 4 })]).toEqual([true, false, true])
  expect(subagentTotals(turns, {})).toEqual({ spawned: 2, answered: 1, share: 21 })
})

test('the pane shows each turn\'s subagents, their tasks and the answers they returned', async ($, on) => {
  session(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface } as any)
    const tree = words(await ui.drawn())
    expect(tree).toMatch(/SUBAGENTS/)
    expect(tree).toMatch(/Spawned\n2/)
    expect(tree).toMatch(/Answered\n1\/2/)
    expect(tree).toMatch(/Share of cost\n21%/)
    expect(tree).toMatch(/↳ lookup on Haiku 5\.5 · 2 req · ↩ answered ~29 tok/)
    expect(tree).toMatch(/Task: Find where the router reads the cost weight/)
    expect(tree).toMatch(/Answer: src\/router\/policy\.py:88 reads cost_weight/)
    expect(tree).toMatch(/↳ judge on Opus 5\.5 · 1 req · working…/)
    await ui.unmount()
  }
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' } as any)
  await ui.press({ key: 'collapse' } as any)
  expect(words(await ui.drawn())).toMatch(/· 1 turn · 2 req · 2 subagents/)
  await ui.unmount()
})

test('a subagent that finishes in this session shows as answered, by its size alone', async ($, on) => {
  session(on)
  await $.session.start({ source: 'startup', cwd: '/tmp/project' } as any)
  await $.turn.complete({ turnId: 't9', agentId: 'agent-7', answer: 'x'.repeat(100), reason: 'answer', durationMs: 5,
    isAborted: false } as any)
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' } as any)
  const tree = words(await ui.drawn())
  expect(tree).toMatch(/↳ judge on Opus 5\.5 · 1 req · ↩ answered ~25 tok/)
  expect(tree).toMatch(/Answered\n2\/2/)
  expect(tree).not.toMatch(/xxxx/)  // the answer's text is never kept
  await ui.unmount()
})
