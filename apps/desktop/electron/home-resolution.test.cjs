const assert = require('node:assert/strict')
const path = require('node:path')
const test = require('node:test')

const { resolveEvelynHome } = require('./home-resolution.cjs')

function resolver({
  env = {},
  existing = [],
  homeDir = '/Users/alice',
  isWindows = false,
  registry = {},
  userDataOverride = ''
} = {}) {
  const pathApi = isWindows ? path.win32 : path.posix
  const present = new Set(existing.map(value => pathApi.normalize(value)))
  return resolveEvelynHome({
    directoryExists: value => present.has(pathApi.normalize(value)),
    env,
    homeDir,
    isWindows,
    normalizeHome: value => pathApi.normalize(value),
    readWindowsUserEnvVar: name => registry[name] || null,
    userDataOverride
  })
}

test('fresh POSIX install defaults to ~/.evelyn', () => {
  assert.equal(resolver(), path.normalize('/Users/alice/.evelyn'))
})

test('existing POSIX ~/.hermes is reused when ~/.evelyn is absent', () => {
  assert.equal(
    resolver({ existing: ['/Users/alice/.hermes'] }),
    path.normalize('/Users/alice/.hermes')
  )
})

test('POSIX ~/.evelyn wins when both default roots exist', () => {
  assert.equal(
    resolver({ existing: ['/Users/alice/.evelyn', '/Users/alice/.hermes'] }),
    path.normalize('/Users/alice/.evelyn')
  )
})

test('EVELYN_HOME precedes legacy HERMES_HOME', () => {
  assert.equal(
    resolver({ env: { EVELYN_HOME: '/data/evelyn', HERMES_HOME: '/data/hermes' } }),
    path.normalize('/data/evelyn')
  )
})

test('legacy HERMES_HOME remains accepted', () => {
  assert.equal(
    resolver({ env: { HERMES_HOME: '/data/hermes' } }),
    path.normalize('/data/hermes')
  )
})

test('fresh Windows install defaults to LOCALAPPDATA/evelyn', () => {
  assert.equal(
    resolver({ env: { LOCALAPPDATA: 'C:\\Users\\alice\\AppData\\Local' }, homeDir: 'C:\\Users\\alice', isWindows: true }),
    path.win32.normalize('C:\\Users\\alice\\AppData\\Local\\evelyn')
  )
})

test('Windows reuses LOCALAPPDATA/hermes before older home-dot-hermes', () => {
  const local = path.win32.normalize('C:\\Users\\alice\\AppData\\Local')
  const localLegacy = path.win32.join(local, 'hermes')
  const dotLegacy = path.win32.join(path.win32.normalize('C:\\Users\\alice'), '.hermes')
  assert.equal(
    resolver({
      env: { LOCALAPPDATA: local },
      existing: [localLegacy, dotLegacy],
      homeDir: 'C:\\Users\\alice',
      isWindows: true
    }),
    localLegacy
  )
})

test('Windows live EVELYN_HOME registry value precedes HERMES_HOME', () => {
  assert.equal(
    resolver({
      homeDir: 'C:\\Users\\alice',
      isWindows: true,
      registry: { EVELYN_HOME: 'D:\\Evelyn', HERMES_HOME: 'D:\\Hermes' }
    }),
    path.win32.normalize('D:\\Evelyn')
  )
})

test('Windows registry Evelyn beats a stale process Hermes value', () => {
  assert.equal(
    resolver({
      env: { HERMES_HOME: 'C:\\stale-hermes' },
      homeDir: 'C:\\Users\\alice',
      isWindows: true,
      registry: { EVELYN_HOME: 'D:\\current-evelyn' }
    }),
    path.win32.normalize('D:\\current-evelyn')
  )
})

test('desktop test sandbox keeps its isolated hermes-home child', () => {
  assert.equal(
    resolver({ userDataOverride: '/tmp/evelyn-desktop-test' }),
    path.normalize('/tmp/evelyn-desktop-test/hermes-home')
  )
})
