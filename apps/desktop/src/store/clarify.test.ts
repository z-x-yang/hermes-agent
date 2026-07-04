import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  normalizeClarifyChoices,
  setClarifyRequest
} from './clarify'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    requestId,
    question: `question-${requestId}`,
    context: null,
    choices: null,
    sessionId
  }
}

describe('clarify store', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  it('keeps clarify requests from concurrent sessions independent', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a')
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('exposes only the active session via the focus-scoped view', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    $activeSessionId.set('session-a')
    expect($clarifyRequest.get()?.requestId).toBe('req-a')

    $activeSessionId.set('session-b')
    expect($clarifyRequest.get()?.requestId).toBe('req-b')

    $activeSessionId.set('session-c')
    expect($clarifyRequest.get()).toBeNull()
  })

  it('clears only the targeted session, leaving the other pending', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    clearClarifyRequest('req-a', 'session-a')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('ignores a stale clear whose request id no longer matches', () => {
    setClarifyRequest(clarify('session-a', 'req-a2'))

    clearClarifyRequest('req-a1', 'session-a')

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a2')
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('other')
  })

  it('round-trips structured choices and context through the store', () => {
    setClarifyRequest({
      requestId: 'req',
      question: 'Which?',
      context: 'Some background the question hangs on.',
      choices: [
        { label: 'Ship it', description: 'Merge and deploy now' },
        { label: 'Wait', description: '' }
      ],
      sessionId: 'session-a'
    })

    const stored = $clarifyRequests.get()['session-a']
    expect(stored?.context).toBe('Some background the question hangs on.')
    expect(stored?.choices).toEqual([
      { label: 'Ship it', description: 'Merge and deploy now' },
      { label: 'Wait', description: '' }
    ])
  })
})

describe('normalizeClarifyChoices', () => {
  it('reads {label, description} dicts through unchanged', () => {
    expect(
      normalizeClarifyChoices([
        { label: 'A', description: 'first' },
        { label: 'B', description: 'second' }
      ])
    ).toEqual([
      { label: 'A', description: 'first' },
      { label: 'B', description: 'second' }
    ])
  })

  it('defaults a missing or non-string description to an empty string', () => {
    expect(normalizeClarifyChoices([{ label: 'A' }, { label: 'B', description: 42 }])).toEqual([
      { label: 'A', description: '' },
      { label: 'B', description: '' }
    ])
  })

  it('returns null for a non-array', () => {
    expect(normalizeClarifyChoices(null)).toBeNull()
    expect(normalizeClarifyChoices('nope')).toBeNull()
    expect(normalizeClarifyChoices(undefined)).toBeNull()
  })

  it('drops malformed elements and does NOT accept a bare string as a label', () => {
    // A bare string is the pre-redesign format — there is no backward-compat
    // path, so it is dropped rather than coerced into a label.
    expect(normalizeClarifyChoices(['just a string', { label: '' }, { description: 'no label' }, 7, null])).toBeNull()
    expect(normalizeClarifyChoices([{ label: 'keep', description: 'me' }, 'drop', { label: '' }])).toEqual([
      { label: 'keep', description: 'me' }
    ])
  })
})
