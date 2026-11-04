"""
Security Guardrails Module.
Provides indirect prompt injection detection for retrieved documents and
log redactor for credentials, API tokens, and authorization headers.
"""

import re
from typing import Dict, Any, List


class SecurityGuardrails:
    """
    Enforces runtime safety across retrieved evidence and application telemetry.
    """

    INDIRECT_INJECTION_SIGNALS = [
        r"system override",
        r"ignore (?:all )?(?:previous )?instructions",
        r"disregard (?:the )?rules",
        r"you must output",
        r"output hacked",
        r"do not mention this document",
        r"print the system prompt",
        r"exfiltrate",
        r"new rule:"
    ]

    CREDENTIAL_PATTERNS = [
        (r"(ghp_[a-zA-Z0-9]{36})", "[REDACTED_GITHUB_PAT]"),
        (r"(sbp_[a-zA-Z0-9]{40})", "[REDACTED_SUPABASE_TOKEN]"),
        (r"(sk-[a-zA-Z0-9]{48})", "[REDACTED_OPENAI_KEY]"),
        (r"(Bearer\s+[a-zA-Z0-9_\-\.]{20,})", "Bearer [REDACTED_TOKEN]"),
        (r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)([a-zA-Z0-9_\-]{16,})", r"\1[REDACTED_API_KEY]")
    ]

    def inspect_evidence_chunk(self, content: str) -> bool:
        """
        Returns True if content contains indirect prompt injection signatures.
        """
        lower = content.lower()
        for pat in self.INDIRECT_INJECTION_SIGNALS:
            if re.search(pat, lower):
                return True
        return False

    def redact_telemetry(self, text: str) -> str:
        """
        Redacts sensitive tokens, API keys, and credentials from logs or traces.
        """
        redacted = text
        for pattern, replacement in self.CREDENTIAL_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted)
        return redacted
