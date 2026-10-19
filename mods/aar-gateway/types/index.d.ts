// One request as the pane shows it (from the gateway's request log; no prompt text)
export type RequestRow = {
  ts: string
  requested_model: string
  routed_model: string
  reason: string | null
  status: number
  subagent: boolean
  agent_role: string | null
  cache_read_input_tokens: number | null
  cache_creation_input_tokens: number | null
  input_tokens: number | null
  cost_usd: number | null
  cost_if_requested_usd: number | null
}

export type GatewayView = {
  isRunning: boolean
  sessionSavedPct: number | null
  allSavedPct: number | null
  allRequests: number
  recent: RequestRow[]
}

// This session's cache use, counted from the usage Claude Code reports per request
export type CacheTotals = { read: number; written: number; fresh: number }

declare module 'claude-code' {
  interface PluginState {
    'aar-gateway': { gateway: GatewayView; cache: CacheTotals }
  }
}
