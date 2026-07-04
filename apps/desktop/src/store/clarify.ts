import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

// A single clarify option. Mirrors the Python-side contract
// (tools/clarify_tool._normalize_choice): `label` is a non-empty string,
// `description` is a string that may be empty but is never null. The gateway
// normalizes every choice before it reaches the wire, so this end trusts the
// shape and reads `choice.label` / `choice.description` directly.
export interface ClarifyChoice {
  label: string
  description: string
}

export interface ClarifyRequest {
  requestId: string
  question: string
  context: string | null
  choices: ClarifyChoice[] | null
  sessionId: string | null
}

// Parse a wire/args `choices` value into the typed contract. The Python side is
// authoritative and only ever emits well-formed `{label, description}` dicts,
// so under normal operation this drops nothing — it is the same defensive
// filter the old code applied to bare strings, re-pointed at the dict shape.
// It deliberately does NOT accept a bare string as a label: there is no
// backward-compat path for the pre-redesign format (see the Python contract).
export function normalizeClarifyChoices(raw: unknown): ClarifyChoice[] | null {
  if (!Array.isArray(raw)) {
    return null
  }

  const choices: ClarifyChoice[] = []

  for (const item of raw) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const row = item as Record<string, unknown>

    if (typeof row.label !== 'string' || !row.label) {
      continue
    }

    choices.push({
      label: row.label,
      description: typeof row.description === 'string' ? row.description : ''
    })
  }

  return choices.length > 0 ? choices : null
}

// Pending clarify requests keyed by the runtime session id that raised them.
// Storing per-session (instead of one shared slot) lets a *background* session
// park its clarify request while the user is looking at a different chat, then
// resolve it once they switch over — without a second concurrent clarify
// clobbering the first. A request with no session id lands under the empty key.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $clarifyRequests = atom<Record<string, ClarifyRequest>>({})

// The clarify request for the currently-viewed session. The inline ClarifyTool
// only ever mounts inside the active session's transcript, so it reads this
// focus-scoped view rather than reaching into the whole map.
export const $clarifyRequest = computed(
  [$clarifyRequests, $activeSessionId],
  (requests, activeId) => requests[keyFor(activeId)] ?? null
)

export function setClarifyRequest(request: ClarifyRequest): void {
  $clarifyRequests.set({ ...$clarifyRequests.get(), [keyFor(request.sessionId)]: request })
}

export function clearClarifyRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $clarifyRequests.get()

  // Targeted clear when the caller knows the session (the common path from the
  // inline ClarifyTool answering its own request).
  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $clarifyRequests.set(next)

    return
  }

  // Fallback with no session hint: drop every entry matching the request id
  // (or clear all when none is given).
  const next: Record<string, ClarifyRequest> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      changed = true
    }
  }

  if (changed) {
    $clarifyRequests.set(next)
  }
}
