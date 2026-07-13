const fs = require('node:fs')
const path = require('node:path')

function macosAppMigrationPaths(targetApp, exists = fs.existsSync) {
  if (!targetApp || path.basename(targetApp) !== 'Hermes.app') {
    return { installTarget: targetApp, legacyTarget: null, legacyBackup: null }
  }

  const installTarget = path.join(path.dirname(targetApp), 'Evelyn.app')
  const backupBase = `${targetApp}.evelyn-migrated-backup`
  let legacyBackup = backupBase
  for (let suffix = 1; exists(legacyBackup); suffix += 1) {
    legacyBackup = `${backupBase}.${suffix}`
  }
  return { installTarget, legacyTarget: targetApp, legacyBackup }
}

function macosAppMigrationConflict({ installTarget, legacyTarget }, exists = fs.existsSync) {
  if (!legacyTarget || !exists(installTarget)) return null
  return `Refusing to replace existing Evelyn app at ${installTarget} while migrating legacy app ${legacyTarget}`
}

module.exports = { macosAppMigrationPaths, macosAppMigrationConflict }
