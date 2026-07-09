import { describe, expect, it } from 'vitest'

import { REASONING_EFFORT_OPTIONS, REASONING_EFFORT_VALUES } from './reasoning-effort-options'

describe('reasoning effort options', () => {
  it('keeps xhigh and max as distinct values and labels', () => {
    expect(REASONING_EFFORT_VALUES).toEqual(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    expect(REASONING_EFFORT_OPTIONS.slice(-2)).toEqual([
      { value: 'xhigh', labelKey: 'xhigh' },
      { value: 'max', labelKey: 'max' }
    ])
  })

  it('does not expose ultra yet', () => {
    expect(REASONING_EFFORT_VALUES).not.toContain('ultra')
  })
})
