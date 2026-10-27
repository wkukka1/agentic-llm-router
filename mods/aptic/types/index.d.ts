// One turn as the pane shows it: what you typed, grouped with its tool loop and
// subagents (from the gateway's request log; no prompt text)
export type TurnRow = {
  time: string  // HH:MM, local time
  requested_model: string | null
  models: string[]
  reason: string | null
  requests: number
  side_requests: number
  errors: number
  cache_read: number
  cache_written: number
  fresh: number
  saved_pct: number | null
  cost_usd?: number
  subagents: SpawnRow[]
  returned?: number  // subagents' answers the main agent's requests carried back
  subagent_cost_usd?: number
}

// One subagent a turn spawned: its kind, the model(s) it ran on, and whether its
// answer came back to the main agent. The task and answer previews are there only
// when the gateway keeps prompt history; older gateways send role, model and requests
export type SpawnRow = {
  role: string
  model: string | null
  requests: number
  spawn_id?: string | null
  models?: string[]
  cost_usd?: number
  returned?: boolean
  answer_tokens?: number | null
  task_preview?: string | null
  answer_preview?: string | null
}

// Subagents that finished in this session, by agent id -> the size of the answer
// (about 4 characters a token); the size only, never the text
export type Returns = Record<string, number>

export type GatewayView = {
  isRunning: boolean
  sessionSavedPct: number | null
  allSavedPct: number | null
  allRequests: number
  turns: TurnRow[]
}

// This session's cache use, counted from the usage Claude Code reports per request
export type CacheTotals = { read: number; written: number; fresh: number }

// The pane's view choices, per user (copied to $.store so they outlast the session)
export type PanePrefs = { collapsed: boolean; oldestFirst: boolean; settingsOpen: boolean }

declare module 'claude-code' {
  interface PluginState {
    'aptic': { gateway: GatewayView; cache: CacheTotals; prefs: PanePrefs; returns: Returns }
  }
}
