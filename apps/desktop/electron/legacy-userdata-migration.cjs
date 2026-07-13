const fs = require('node:fs')
const path = require('node:path')

const LEGACY_DESKTOP_STATE_FILES = Object.freeze([
  'connection.json',
  'updates.json',
  'window-state.json',
  'active-profile.json',
  'native-theme.json',
  'translucency.json',
  'project-dir.json'
])
const LEGACY_DESKTOP_STATE_DIRECTORIES = Object.freeze(['Local Storage'])

function migrateLegacyDesktopState({
  currentUserData,
  legacyUserData,
  onWarning = message => console.warn(message)
}) {
  if (!currentUserData || !legacyUserData || path.resolve(currentUserData) === path.resolve(legacyUserData)) {
    return []
  }
  if (!fs.existsSync(legacyUserData)) return []

  fs.mkdirSync(currentUserData, { recursive: true })
  const copied = []
  for (const filename of LEGACY_DESKTOP_STATE_FILES) {
    const source = path.join(legacyUserData, filename)
    const destination = path.join(currentUserData, filename)
    if (!fs.existsSync(source) || fs.existsSync(destination)) continue
    try {
      fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL)
      copied.push(filename)
    } catch (error) {
      onWarning(`Could not migrate legacy Desktop state ${filename}: ${error.message}`)
    }
  }
  for (const dirname of LEGACY_DESKTOP_STATE_DIRECTORIES) {
    const source = path.join(legacyUserData, dirname)
    const destination = path.join(currentUserData, dirname)
    if (!fs.existsSync(source)) continue
    if (fs.existsSync(destination)) {
      onWarning(`Could not migrate legacy Desktop state: ${dirname} already exists in Evelyn userData`)
      continue
    }
    const stagingRoot = fs.mkdtempSync(path.join(currentUserData, '.evelyn-migrate-'))
    const staging = path.join(stagingRoot, dirname)
    try {
      fs.cpSync(source, staging, { recursive: true, force: false, errorOnExist: true })
      if (fs.existsSync(destination)) {
        onWarning(`Could not migrate legacy Desktop state: ${dirname} already exists in Evelyn userData`)
        continue
      }
      fs.renameSync(staging, destination)
      copied.push(dirname)
    } catch (error) {
      onWarning(`Could not migrate legacy Desktop state ${dirname}: ${error.message}`)
    } finally {
      try {
        fs.rmSync(stagingRoot, { recursive: true, force: true })
      } catch (cleanupError) {
        onWarning(`Could not clean legacy Desktop migration staging for ${dirname}: ${cleanupError.message}`)
      }
    }
  }
  return copied
}

function migrateLegacyDesktopStateForApp({ app, userDataOverride = false, onWarning = message => console.warn(message) }) {
  if (userDataOverride) return []
  try {
    return migrateLegacyDesktopState({
      currentUserData: app.getPath('userData'),
      legacyUserData: path.join(app.getPath('appData'), 'Hermes'),
      onWarning
    })
  } catch (error) {
    onWarning(`Could not migrate legacy Desktop state: ${error.message}`)
    return []
  }
}

module.exports = {
  LEGACY_DESKTOP_STATE_FILES,
  LEGACY_DESKTOP_STATE_DIRECTORIES,
  migrateLegacyDesktopState,
  migrateLegacyDesktopStateForApp
}
