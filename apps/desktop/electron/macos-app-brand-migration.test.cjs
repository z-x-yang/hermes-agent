const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { macosAppMigrationPaths, macosAppMigrationConflict } = require('./macos-app-brand-migration.cjs')

test('renames a legacy Hermes.app target to Evelyn and preserves the old bundle as backup', () => {
  const target = path.join('/Applications', 'Hermes.app')
  assert.deepEqual(macosAppMigrationPaths(target, () => false), {
    installTarget: path.join('/Applications', 'Evelyn.app'),
    legacyTarget: target,
    legacyBackup: `${target}.evelyn-migrated-backup`
  })
})

test('chooses a unique non-app backup without overwriting an earlier migration', () => {
  const target = path.join('/Applications', 'Hermes.app')
  const occupied = new Set([
    `${target}.evelyn-migrated-backup`,
    `${target}.evelyn-migrated-backup.1`
  ])
  assert.equal(
    macosAppMigrationPaths(target, candidate => occupied.has(candidate)).legacyBackup,
    `${target}.evelyn-migrated-backup.2`
  )
})

test('refuses to overwrite an existing Evelyn.app during legacy migration', () => {
  const target = path.join('/Applications', 'Hermes.app')
  const paths = macosAppMigrationPaths(target, () => false)
  assert.match(
    macosAppMigrationConflict(paths, candidate => candidate === paths.installTarget),
    /Refusing to replace existing Evelyn app/
  )
  assert.equal(macosAppMigrationConflict(paths, () => false), null)
})

test('keeps an Evelyn.app target in place', () => {
  const target = path.join('/Applications', 'Evelyn.app')
  assert.deepEqual(macosAppMigrationPaths(target), {
    installTarget: target,
    legacyTarget: null,
    legacyBackup: null
  })
})
