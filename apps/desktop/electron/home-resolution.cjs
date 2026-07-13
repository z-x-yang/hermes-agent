const path = require('node:path')

function nonEmpty(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

/**
 * Resolve the shared Evelyn data root without splitting legacy Hermes state.
 *
 * The dependency-injected shape keeps Windows and filesystem precedence
 * testable on every host; Electron main passes the real app/env helpers.
 */
function resolveEvelynHome({
  directoryExists,
  env = process.env,
  homeDir,
  isWindows = process.platform === 'win32',
  normalizeHome = value => path.resolve(value),
  readWindowsUserEnvVar = () => null,
  userDataOverride = ''
}) {
  const pathApi = isWindows ? path.win32 : path.posix

  // Resolve by variable identity, not by source. Explorer can retain a stale
  // process HERMES_HOME after `setx EVELYN_HOME ...`, so the live Evelyn
  // registry value must still beat every Hermes compatibility value.
  const explicitEvelyn = nonEmpty(env.EVELYN_HOME) ||
    (isWindows ? nonEmpty(readWindowsUserEnvVar('EVELYN_HOME')) : '')
  if (explicitEvelyn) return normalizeHome(explicitEvelyn)

  const explicitHermes = nonEmpty(env.HERMES_HOME) ||
    (isWindows ? nonEmpty(readWindowsUserEnvVar('HERMES_HOME')) : '')
  if (explicitHermes) return normalizeHome(explicitHermes)

  if (userDataOverride) {
    return normalizeHome(pathApi.join(pathApi.resolve(userDataOverride), 'hermes-home'))
  }

  if (isWindows) {
    const localBase = nonEmpty(env.LOCALAPPDATA) || pathApi.join(homeDir, 'AppData', 'Local')
    const preferred = pathApi.join(localBase, 'evelyn')
    const legacyLocal = pathApi.join(localBase, 'hermes')
    const legacyDot = pathApi.join(homeDir, '.hermes')

    if (directoryExists(preferred)) return normalizeHome(preferred)
    if (directoryExists(legacyLocal)) return normalizeHome(legacyLocal)
    if (directoryExists(legacyDot)) return normalizeHome(legacyDot)
    return normalizeHome(preferred)
  }

  const preferred = pathApi.join(homeDir, '.evelyn')
  const legacy = pathApi.join(homeDir, '.hermes')
  if (directoryExists(preferred)) return normalizeHome(preferred)
  if (directoryExists(legacy)) return normalizeHome(legacy)
  return normalizeHome(preferred)
}

module.exports = { resolveEvelynHome }
