import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { Intro } from './intro'

describe('new-session intro branding', () => {
  it('renders Evelyn as the product wordmark', () => {
    render(<Intro seed={0} />)

    expect(screen.getByLabelText('EVELYN')).toBeTruthy()
    expect(screen.queryByText('HERMES AGENT')).toBeNull()
  })
})
