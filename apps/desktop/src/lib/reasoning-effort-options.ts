export const REASONING_EFFORT_VALUES = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const

export const REASONING_EFFORT_OPTIONS = REASONING_EFFORT_VALUES.filter(value => value !== 'none').map(value => ({
  value,
  labelKey: value
}))
