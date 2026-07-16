"""Anti-erosion guards for behavioral sentences carried in tool descriptions.

Tool schemas in ``tools/*.py`` are exactly the surface where upstream ports
have silently reverted fork one-liners before (see the 0.18-port incident that
motivated ``tests/agent/test_prompt_contracts.py``).  These assertions pin the
fork's point-of-use behavior contracts so a port that rewrites a description
fails here instead of shipping.
"""

from tools.clarify_tool import CLARIFY_SCHEMA
from tools.delegate_tool import DELEGATE_TASK_SCHEMA
from tools.file_tools import READ_FILE_SCHEMA


class TestClarifyReserveGate:
    def test_mid_task_questions_reserved_for_next_step_changing_answers(self):
        text = CLARIFY_SCHEMA["description"]
        assert "Reserve mid-task questions" in text
        assert "changes what you do next" in text
        assert "pick the obvious option, state it in your reply, and proceed" in text


class TestReadFileNoSelfVerifyReread:
    def test_edit_verification_uses_patch_diff_not_reread(self):
        text = READ_FILE_SCHEMA["description"]
        assert "Do not re-read a file just to verify your own edit" in text
        assert "check that diff" in text


class TestDelegateNoDuplicateWork:
    def test_delegated_work_not_duplicated_but_recoverable(self):
        text = DELEGATE_TASK_SCHEMA["description"]
        assert "do not duplicate the work while it runs" in text
        # The wait rule must carry its own exit — a dead child cannot
        # strand the task behind an unconditional "wait".
        assert "take the work back or re-delegate" in text
        assert "a dead delegation does not discharge the task" in text
