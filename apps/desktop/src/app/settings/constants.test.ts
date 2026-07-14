import { describe, expect, it } from 'vitest'

import { FIELD_DESCRIPTIONS, FIELD_LABELS, SECTIONS } from './constants'
import { fieldCopyForSchemaKey } from './field-copy'

describe('desktop context-window setting', () => {
  it('binds the existing Context Window control to Evelyn’s independent window', () => {
    const modelSection = SECTIONS.find(section => section.id === 'model')

    expect(modelSection?.keys).toContain('compression.internal_context_length')
    expect(modelSection?.keys).not.toContain('model_context_length')
    expect(fieldCopyForSchemaKey(FIELD_LABELS, 'compression.internal_context_length')).toBe(
      'Context Window'
    )
    expect(
      fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, 'compression.internal_context_length')
    ).toContain('independent')
  })
})
