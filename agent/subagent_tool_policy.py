from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToolNamePolicy:
    allowed_names: Optional[frozenset[str]] = None
    denied_names: frozenset[str] = frozenset()

    def allows(self, name: str) -> bool:
        if name in self.denied_names:
            return False
        if self.allowed_names is not None and name not in self.allowed_names:
            return False
        return True


def apply_tool_policy_to_agent(agent, policy: ToolNamePolicy) -> None:
    """Persist and expose only tools permitted by a subagent capability policy."""
    agent._subagent_tool_policy = policy
    agent.tools = [
        definition
        for definition in list(getattr(agent, "tools", []) or [])
        if policy.allows(definition["function"]["name"])
    ]
    agent.valid_tool_names = {
        name
        for name in set(getattr(agent, "valid_tool_names", set()) or set())
        if policy.allows(name)
    }
    # Built-in subagent profiles are closed allowlists. Refreshing MCP schemas
    # later would otherwise restore definitions removed above.
    agent._skip_mcp_refresh = True


def tool_policy_block_message(agent, tool_name: str) -> Optional[str]:
    """Return a fail-closed runtime denial for a tool name, if configured."""
    policy = getattr(agent, "_subagent_tool_policy", None)
    if policy is None or policy.allows(tool_name):
        return None
    return (
        f"Tool {tool_name!r} is blocked by subagent capability policy. "
        "Do not work around this restriction or spawn another agent."
    )
