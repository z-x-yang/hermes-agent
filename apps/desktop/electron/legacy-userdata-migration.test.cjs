const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  migrateLegacyDesktopState,
  migrateLegacyDesktopStateForApp
} = require('./legacy-userdata-migration.cjs')

test('copies only missing allowlisted state and leaves legacy data untouched', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-userdata-migration-'))
  const legacyUserData = path.join(root, 'Hermes')
  const currentUserData = path.join(root, 'Evelyn')
  fs.mkdirSync(legacyUserData)
  fs.mkdirSync(currentUserData)
  fs.writeFileSync(path.join(legacyUserData, 'connection.json'), '{"legacy":true}')
  fs.writeFileSync(path.join(legacyUserData, 'updates.json'), '{"channel":"stable"}')
  fs.mkdirSync(path.join(legacyUserData, 'Local Storage', 'leveldb'), { recursive: true })
  fs.writeFileSync(path.join(legacyUserData, 'Local Storage', 'leveldb', '000003.log'), 'draft state')
  fs.writeFileSync(path.join(legacyUserData, 'Cookies'), 'not allowlisted')
  fs.writeFileSync(path.join(currentUserData, 'connection.json'), '{"current":true}')

  const copied = migrateLegacyDesktopState({ currentUserData, legacyUserData })

  assert.deepEqual(copied, ['updates.json', 'Local Storage'])
  assert.equal(fs.readFileSync(path.join(currentUserData, 'connection.json'), 'utf8'), '{"current":true}')
  assert.equal(fs.readFileSync(path.join(currentUserData, 'updates.json'), 'utf8'), '{"channel":"stable"}')
  assert.equal(fs.existsSync(path.join(currentUserData, 'Cookies')), false)
  assert.equal(
    fs.readFileSync(path.join(currentUserData, 'Local Storage', 'leveldb', '000003.log'), 'utf8'),
    'draft state'
  )
  assert.equal(fs.readFileSync(path.join(legacyUserData, 'updates.json'), 'utf8'), '{"channel":"stable"}')
  assert.equal(
    fs.readFileSync(path.join(legacyUserData, 'Local Storage', 'leveldb', '000003.log'), 'utf8'),
    'draft state'
  )

  fs.rmSync(root, { recursive: true, force: true })
})

test('does not merge Local Storage when Evelyn already has a LevelDB', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-userdata-conflict-'))
  const legacyUserData = path.join(root, 'Hermes')
  const currentUserData = path.join(root, 'Evelyn')
  fs.mkdirSync(path.join(legacyUserData, 'Local Storage', 'leveldb'), { recursive: true })
  fs.mkdirSync(path.join(currentUserData, 'Local Storage', 'leveldb'), { recursive: true })
  fs.writeFileSync(path.join(legacyUserData, 'Local Storage', 'leveldb', 'legacy.log'), 'legacy')
  fs.writeFileSync(path.join(currentUserData, 'Local Storage', 'leveldb', 'current.log'), 'current')
  const warnings = []

  const copied = migrateLegacyDesktopState({
    currentUserData,
    legacyUserData,
    onWarning: message => warnings.push(message)
  })

  assert.deepEqual(copied, [])
  assert.equal(fs.existsSync(path.join(currentUserData, 'Local Storage', 'leveldb', 'legacy.log')), false)
  assert.equal(
    fs.readFileSync(path.join(currentUserData, 'Local Storage', 'leveldb', 'current.log'), 'utf8'),
    'current'
  )
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /Local Storage already exists/)
  fs.rmSync(root, { recursive: true, force: true })
})

test('never deletes Local Storage created by a competing launch when staging copy fails', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-userdata-race-'))
  const legacyUserData = path.join(root, 'Hermes')
  const currentUserData = path.join(root, 'Evelyn')
  const destination = path.join(currentUserData, 'Local Storage')
  fs.mkdirSync(path.join(legacyUserData, 'Local Storage', 'leveldb'), { recursive: true })
  fs.mkdirSync(currentUserData)
  const originalCpSync = fs.cpSync
  fs.cpSync = () => {
    fs.mkdirSync(path.join(destination, 'leveldb'), { recursive: true })
    fs.writeFileSync(path.join(destination, 'leveldb', 'winner.log'), 'keep me')
    throw new Error('simulated staging copy failure')
  }
  const warnings = []
  try {
    assert.deepEqual(
      migrateLegacyDesktopState({
        currentUserData,
        legacyUserData,
        onWarning: message => warnings.push(message)
      }),
      []
    )
  } finally {
    fs.cpSync = originalCpSync
  }
  assert.equal(fs.readFileSync(path.join(destination, 'leveldb', 'winner.log'), 'utf8'), 'keep me')
  assert.match(warnings.join('\n'), /simulated staging copy failure/)
  fs.rmSync(root, { recursive: true, force: true })
})

test('derives the legacy Hermes directory from Electron appData', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-appdata-migration-'))
  const legacyUserData = path.join(root, 'Hermes')
  const currentUserData = path.join(root, 'Evelyn')
  fs.mkdirSync(legacyUserData)
  fs.writeFileSync(path.join(legacyUserData, 'active-profile.json'), '{"profile":"work"}')
  const app = {
    getPath(name) {
      return name === 'appData' ? root : currentUserData
    }
  }

  const copied = migrateLegacyDesktopStateForApp({ app })

  assert.deepEqual(copied, ['active-profile.json'])
  assert.equal(
    fs.readFileSync(path.join(currentUserData, 'active-profile.json'), 'utf8'),
    '{"profile":"work"}'
  )
  fs.rmSync(root, { recursive: true, force: true })
})

test('reports filesystem migration failures without crashing packaged startup', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-appdata-failure-'))
  const legacyUserData = path.join(root, 'Hermes')
  fs.mkdirSync(legacyUserData)
  fs.writeFileSync(path.join(legacyUserData, 'updates.json'), '{}')
  const warnings = []
  const originalMkdirSync = fs.mkdirSync
  fs.mkdirSync = () => {
    throw new Error('read-only destination')
  }
  try {
    assert.deepEqual(
      migrateLegacyDesktopStateForApp({
        app: {
          getPath(name) {
            if (name === 'appData') return root
            if (name === 'userData') return path.join(root, 'Evelyn')
            throw new Error(`unexpected path ${name}`)
          }
        },
        onWarning: message => warnings.push(message)
      }),
      []
    )
  } finally {
    fs.mkdirSync = originalMkdirSync
    fs.rmSync(root, { recursive: true, force: true })
  }
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /read-only destination/)
})

test('skips legacy migration for explicit sandbox userData overrides', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'evelyn-override-migration-'))
  const legacyUserData = path.join(root, 'Hermes')
  const currentUserData = path.join(root, 'sandbox')
  fs.mkdirSync(legacyUserData)
  fs.writeFileSync(path.join(legacyUserData, 'connection.json'), '{"real":true}')
  const app = {
    getPath(name) {
      return name === 'appData' ? root : currentUserData
    }
  }

  const copied = migrateLegacyDesktopStateForApp({ app, userDataOverride: true })

  assert.deepEqual(copied, [])
  assert.equal(fs.existsSync(path.join(currentUserData, 'connection.json')), false)
  fs.rmSync(root, { recursive: true, force: true })
})
