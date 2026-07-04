"""Discord one-click Notion task completion.

Submodules:
  detection     — pure helpers (link extraction, task detection, Status read/write)
  notion_client — async Notion REST client (lazy token load)
  tracker       — persistent page_id -> Discord message-location map
  registry      — module-level active-controller registry (breaks import cycle)
  buttons       — DynamicItem complete/undo button (restart-safe routing)
  controller    — orchestration (render / complete / undo / snooze / sync / thread)
  snooze        — persistent local snooze reminder store + preset time helpers
"""
